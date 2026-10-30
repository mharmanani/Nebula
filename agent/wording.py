"""Wording produced by code, not by the language model: the risk on the Diagnostic Certainty Scale, the margin to the
threshold, the recommendation, the provenance footer, and the check for numbers in a report that no output contains."""

from __future__ import annotations

import re

FOOTER = (
    "Provenance: the risk and the decision come from the TabPFN-3 composition over the tool outputs (in context over previous "
    "patients); this text was written by the coordinating language model from those outputs."
)
RECOMMENDATION = {
    True: "I suggest targeting this region (targeted biopsy); the decision rests with the operator.",
    False: "I see no specific target here; the operator may take a systematic core at this location or skip it.",
}


def margin(p: float, t: float) -> str:
    return ("well above" if p >= 1.5 * t else "just above") if p >= t else ("well below" if p < t / 1.5 else "just below")


def risk_statement(p: float, t: float, where: str = "at this core", threshold_name: str = "targeting threshold") -> str:
    """Diagnostic Certainty Scale (most likely > 90%, likely 75-90%, may represent 25-75%, unlikely 5-25%, very unlikely < 5%),
    with the estimate and its margin to the threshold."""
    if p > 0.90:
        s = f"Clinically significant cancer {where} is most likely"
    elif p >= 0.75:
        s = f"Clinically significant cancer {where} is likely"
    elif p >= 0.25:
        s = f"The findings {where} may represent clinically significant cancer"
    else:
        s = f"Clinically significant cancer {where} is " + ("unlikely" if p >= 0.05 else "very unlikely")
    return f"{s} (estimated risk {round(100 * p)}%, {margin(p, t)} the {threshold_name})."


def threshold_note(p: float, t: float) -> str | None:
    """A risk above the threshold but below 25% is worded 'unlikely' yet targeted: say why."""
    if t <= p < 0.25:
        return (
            "The targeting threshold is set so that targets are suggested as often as operators biopsy, so a core can be worth "
            "targeting even when cancer is unlikely."
        )
    return None


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def unsupported_numbers(text: str, source: str) -> list[str]:
    """Numbers in `text` that occur in no output (tolerating lower precision, percentages of fractions and small integers)."""
    found = set(_NUMBER.findall(source))
    values = set()
    for t in found:
        v = float(t)
        values.update({round(v, d) for d in range(4)})
        values.update({round(100 * v, d) for d in range(3)})
    bad = []
    for t in _NUMBER.findall(text):
        v = float(t)
        if t in found or (abs(v) <= 10 and v.is_integer()) or any(abs(v - s) < 1e-9 for s in values):
            continue
        bad.append(t)
    return sorted(set(bad))
