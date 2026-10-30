"""ProstNFound+ network (MedSAM image encoder with adapters, clinical prompts, mask and class decoders).

Our ProstNFound+ implementation, trimmed to the configuration Nebula uses: `build_prostnfound()` builds a MedSAM ViT-B
backbone with 256-d adapters, prompts age / PSA / PSA density, and the class decoder on.
"""

import logging
import typing as tp
from itertools import chain

import torch
from torch import nn

from .segment_anything.modeling.common import LayerNorm2d
from .segment_anything.modeling.mask_decoder import ClassDecoder
from .segment_anything.modeling.sam import Sam
from .segment_anything.modeling.transformer import TwoWayTransformer


class ProstNFound(nn.Module):

    BACKBONE_OPTIONS = [
        "sam",
        "medsam",
        "sam_med2d",
        "adapter_medsam",
        "adapter_sam",
        "adapter_sammed_2d",
    ]
    _WARNED_RESIZE = False

    def __init__(
        self,
        sam_backbone: Sam,
        floating_point_prompts: list[str] = [],
        discrete_prompts: list[str] = [],
        discrete_prompt_nvals: list[int] = [],
        use_sparse_cnn_patch_features: bool = False,
        use_sparse_cnn_patch_features_rf: bool = False,
        use_prostate_mask_prompt: bool = False,
        num_data_independent_prompts: int = 0,
        prompt_dropout: float = 0.0,  # dropout for prompt embeddings
        replace_patch_embed: bool = False,
        sparse_cnn_backbone_path: str | None = None,
        freeze_mask_decoder: bool = False,
        freeze_image_encoder: bool = False,
        freeze_cnn: bool = False,
        img_emb_dropout: float = 0.0,
        cnn_patches_whole_prostate: bool = False,
        pos_embed_cnn_patch: bool = True,
        pool_patch_features: bool | None = None,
        prompt_embedding_dim=256,
        use_class_decoder: bool = False,
        use_cnn_dense_prompt: bool = False,
        output_mode: tp.Literal["heatmaps", "classifier", "all"] = "all",
        upsample_image_encoder_features: tp.Literal["interpolate", "conv", None] = None,
        upsample_image_encoder_features_factor: int = 4,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.use_cnn_dense_prompt = use_cnn_dense_prompt
        self.floating_point_prompts = floating_point_prompts
        self.discrete_prompts = discrete_prompts
        self.discrete_prompt_nvals = discrete_prompt_nvals
        self.prompts = self.floating_point_prompts + self.discrete_prompts
        self.use_sparse_cnn_patch_features = use_sparse_cnn_patch_features
        self.use_sparse_cnn_patch_features_rf = use_sparse_cnn_patch_features_rf
        self.num_data_independent_prompts = num_data_independent_prompts
        self.use_prostate_mask_prompt = use_prostate_mask_prompt
        self.upsample_image_encoder_features = upsample_image_encoder_features

        if use_sparse_cnn_patch_features and use_sparse_cnn_patch_features_rf:
            raise ValueError("Both sparse_cnn_patch_features and sparse_cnn_patch_features_rf cannot be True")

        self.prompt_dropout = prompt_dropout
        self.replace_patch_embed = replace_patch_embed
        self.cnn_patches_whole_prostate = cnn_patches_whole_prostate
        self.pos_embed_cnn_patch = pos_embed_cnn_patch
        self.pool_patch_features = pool_patch_features
        if replace_patch_embed and sam_backbone != "sam_med2d":
            raise ValueError("replace_patch_embed is only supported for sam_med2d backbone")

        self.sparse_cnn_backbone_path = sparse_cnn_backbone_path
        self.medsam_model = sam_backbone
        self.img_emb_dropout = nn.Dropout(img_emb_dropout)

        if freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # ====================================================
        # BUILD PROMPT MODULES
        # ==================================================

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, prompt_embedding_dim))

        # floating point prompts
        self.floating_point_prompt_modules = torch.nn.ModuleDict()
        for prompt in self.floating_point_prompts:
            self.floating_point_prompt_modules[prompt] = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )

        # integer prompts
        self.integer_prompt_modules = torch.nn.ModuleDict()
        for prompt, num_categories in zip(self.discrete_prompts, self.discrete_prompt_nvals):
            self.integer_prompt_modules[prompt] = nn.Embedding(num_categories, prompt_embedding_dim)

        # data independent prompts
        if self.num_data_independent_prompts > 0:
            self.data_independent_prompts = nn.Parameter(torch.randn(1, num_data_independent_prompts, prompt_embedding_dim))

        # optional CNN branches of the original ProstNFound code, not used by ProstNFound+
        if self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf:
            raise NotImplementedError("sparse CNN patch features are not included (not used by ProstNFound+)")
        if self.use_cnn_dense_prompt:
            raise NotImplementedError("the CNN dense prompt is not included (not used by ProstNFound+)")
        else:
            self.cnn_for_dense_embedding = None

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))
        self.use_tc = False

        # ====================================================
        # Build additional modules
        # ==================================================

        self.class_decoder = (
            ClassDecoder(
                transformer_dim=256,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                num_cls_tokens=1,
                cls_output_dims=[2],
            )
            if use_class_decoder
            else None
        )

        if self.upsample_image_encoder_features == "conv":
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(prompt_embedding_dim, prompt_embedding_dim, kernel_size=2, stride=2),
                LayerNorm2d(prompt_embedding_dim),
                nn.GELU(),
                nn.ConvTranspose2d(prompt_embedding_dim, prompt_embedding_dim, kernel_size=2, stride=2),
                nn.GELU(),
            )
        elif self.upsample_image_encoder_features == "interpolate":
            self.upsample = nn.Upsample(
                scale_factor=upsample_image_encoder_features_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.upsample = None

    def forward(
        self,
        bmode=None,
        rf=None,
        prostate_mask=None,
        needle_mask=None,
        output_mode=None,
        output_patch_features=False,
        output_image_feats=False,
        # legacy positional arg name kept for backward compat
        image=None,
        rf_image=None,
        **kwargs,
    ):
        # backward compat: accept old arg names
        if bmode is None and image is not None:
            bmode = image
        if rf is None and rf_image is not None:
            rf = rf_image

        # filter kwargs to known prompts only; reshape (B,) → (B, 1) as needed
        prompts = {}
        for k, v in kwargs.items():
            if k in self.prompts and v is not None:
                if isinstance(v, torch.Tensor) and v.ndim == 1:
                    v = v[:, None]
                prompts[k] = v

        output_mode = output_mode or self.output_mode

        B, C, H, W = bmode.shape

        image_feats = self.medsam_model.image_encoder(bmode)
        image_feats = self.img_emb_dropout(image_feats)
        if self.upsample is not None:
            image_feats = self.upsample(image_feats)

        if self.use_prostate_mask_prompt:
            if prostate_mask is None or self.prompt_dropout > 0 and self.training and torch.rand(1) < self.prompt_dropout:
                mask = None
            else:
                B, C, H, W = prostate_mask.shape
                if H != 256 or W != 256:
                    prostate_mask = torch.nn.functional.interpolate(prostate_mask, size=(256, 256))
                mask = prostate_mask
        else:
            mask = None

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(None, None, mask)
        if (dense_embedding.shape[-2] != image_feats.shape[-2]) or (dense_embedding.shape[-1] != image_feats.shape[-1]):
            dense_embedding = torch.nn.functional.interpolate(
                dense_embedding,
                size=image_feats.shape[-2:],
            )

        dense_embedding = dense_embedding.repeat_interleave(len(bmode), 0)

        if self.use_cnn_dense_prompt:
            # extract dense prompt features from the image
            assert self.cnn_for_dense_embedding is not None
            cnn_emb = self.cnn_for_dense_embedding(bmode).permute(0, 3, 1, 2)
            dense_embedding = dense_embedding + cnn_emb

        sparse_embedding = sparse_embedding.repeat_interleave(len(bmode), 0)

        for prompt_name, prompt_value in prompts.items():
            if prompt_value is None or self.prompt_dropout > 0 and self.training and torch.rand(1) < self.prompt_dropout:
                # skip this prompt and use the 'null embedding' instead
                prompt_embedding = self.null_prompt.repeat_interleave(len(bmode), 0)
            else:
                if prompt_name in self.floating_point_prompts:
                    prompt_embedding = self.floating_point_prompt_modules[prompt_name](prompt_value)
                elif prompt_name in self.discrete_prompts:
                    prompt_embedding = self.integer_prompt_modules[prompt_name](prompt_value)
                else:
                    raise ValueError(f"Unknown prompt: {prompt_name}")

            prompt_embedding = prompt_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding], dim=1)

        if self.num_data_independent_prompts > 0:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if self.use_sparse_cnn_patch_features:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_bmode(bmode, needle_mask, prostate_mask)
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat([sparse_embedding, patch_cnn_sparse_embeddings], dim=1)

        if self.use_sparse_cnn_patch_features_rf:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_rf(rf, needle_mask, prostate_mask)
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat([sparse_embedding, patch_cnn_sparse_embeddings], dim=1)

        pe = self.medsam_model.prompt_encoder.get_dense_pe()
        if (pe.shape[-2] != image_feats.shape[-2]) or (pe.shape[-1] != image_feats.shape[-1]):
            pe = torch.nn.functional.interpolate(
                pe,
                size=image_feats.shape[-2:],
            )

        mask_decoder_result = self.medsam_model.mask_decoder(
            image_feats,
            pe,
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
            output_patch_features=output_patch_features,
        )
        if output_patch_features:
            mask_logits, iou, mask_decoder_patch_features = mask_decoder_result
        else:
            mask_logits, iou = mask_decoder_result[:2]
            mask_decoder_patch_features = None

        mask_logits = mask_logits / self.temperature[None, None, None, :] + self.bias[None, None, None, :]

        if self.class_decoder is not None:
            class_decoder_all_outputs = self.class_decoder(
                image_feats,
                pe,
                sparse_embedding,
                dense_embedding,
                output_hidden_states=True,
                output_patch_features=output_patch_features,
            )
            cls_outputs = class_decoder_all_outputs.pop("class_logits")
        else:
            cls_outputs = None
            class_decoder_all_outputs = None

        if output_mode == "heatmaps":
            return mask_logits
        elif output_mode == "classifier":
            assert cls_outputs is not None
            return cls_outputs[0]
        else:
            outputs = {
                "cancer_logits": mask_logits,
                "mask_logits": mask_logits,  # backward-compat alias
                "cls_outputs": cls_outputs,
                "image_level_classification_outputs": cls_outputs,  # compat alias
                "class_logits_list": cls_outputs,
                "iou": iou,
            }
            if class_decoder_all_outputs:
                outputs["class_decoder_hidden_states"] = class_decoder_all_outputs["class_token_hidden_states"]
                outputs["class_decoder_last_hidden_state"] = outputs["class_decoder_hidden_states"][-1]
                if "patch_features" in class_decoder_all_outputs:
                    outputs["class_decoder_patch_features"] = class_decoder_all_outputs["patch_features"]
            if mask_decoder_patch_features is not None:
                outputs["mask_decoder_patch_features"] = mask_decoder_patch_features
            if output_image_feats:
                outputs["image_feats"] = image_feats
            return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        # always keep cnn in eval mode - otherwise batch norm might interfere.
        if (
            (self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf)
            and self.sparse_cnn_backbone_path is not None
            and getattr(self, "patch_feature_cnn", None) is not None
        ):
            self.patch_feature_cnn.eval()
        return self

    def get_params_groups(self):

        encoder_parameters = [p for (k, p) in self.medsam_model.image_encoder.named_parameters() if "neck" not in k]

        warmup_parameters = []
        # warmup components from backbone
        warmup_parameters = chain(warmup_parameters, self.medsam_model.mask_decoder.parameters())
        warmup_parameters = chain(warmup_parameters, self.medsam_model.prompt_encoder.parameters())
        if hasattr(self.medsam_model.image_encoder, "neck"):
            warmup_parameters = chain(warmup_parameters, self.medsam_model.image_encoder.neck.parameters())
        # null prompt
        warmup_parameters = chain(warmup_parameters, [self.null_prompt])
        # floating point prompts
        for module in self.floating_point_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # data independent prompts
        for module in self.integer_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # data independent prompts
        if self.num_data_independent_prompts > 0:
            warmup_parameters = chain(warmup_parameters, [self.data_independent_prompts])
        if self.class_decoder is not None:
            warmup_parameters = chain(warmup_parameters, self.class_decoder.parameters())
        if self.cnn_for_dense_embedding is not None:
            warmup_parameters = chain(warmup_parameters, self.cnn_for_dense_embedding.parameters())
        if self.upsample is not None:
            warmup_parameters = chain(warmup_parameters, self.upsample.parameters())

        cnn_parameters = []

        return encoder_parameters, warmup_parameters, cnn_parameters

    @property
    def device(self):
        return next(self.parameters()).device


def build_prostnfound(
    medsam_checkpoint: str | None = None, prompts=("age", "psa", "approx_psa_density"), **kwargs
) -> ProstNFound:
    """The ProstNFound+ architecture used by Nebula. `medsam_checkpoint` (public MedSAM ViT-B weights) is only needed to train
    from scratch; a trained ProstNFound+ checkpoint replaces every weight. `prompts`: the clinical prompts (Compass's backbone
    uses age and PSA only)."""
    from .medsam_adapter import build_medsam_adapter

    backbone = build_medsam_adapter(adapter_dim=256, medsam_checkpoint=medsam_checkpoint)
    return ProstNFound(backbone, floating_point_prompts=list(prompts), use_class_decoder=True, **kwargs)
