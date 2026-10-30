"""The coordinator loop: one tool call per turn until `compose` (forced on what was collected if the turn budget runs out),
then a report written without tools. Tools that are not repeatable leave the menu once called; two turns without a tool
call end the loop."""

from __future__ import annotations

import json

from tools.registry import REPEATABLE, coordinator_tools

from .prompts import BUDGET_SPENT, NO_CALL, REPORT, REQUEST, SYSTEM
from .wording import FOOTER, unsupported_numbers

TURN_BUDGET = {"post": 30, "intra": 14}
REPORT_TOKENS = {"post": 900, "intra": 350}


def as_text(result) -> str:
    return json.dumps(result, default=str)


class Coordinator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, session) -> dict:
        wf, case = session.workflow, session.case
        tools = coordinator_tools(session)
        request = REQUEST[wf].format(case=case.case_id, center=case.center)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}]
        calls, trace, sources, no_call = [], [], [request], 0
        for turn in range(TURN_BUDGET[wf]):
            if session.final is not None or no_call >= 2:
                break
            menu = [n for n in tools if n not in calls or n in REPEATABLE]
            raw = self.llm.generate(messages, [tools[n].schema() for n in menu])
            text, call = self.llm.parse(raw)
            assistant = self.llm.assistant_turn(text, call)
            messages.append(assistant)
            if call is None:
                no_call += 1
                trace.append({"turn": turn, "raw": raw, "call": None})
                messages.append({"role": "user", "content": NO_CALL})
                continue
            name = call["name"]
            calls.append(name)
            result = (
                tools[name].run(**call["arguments"])
                if name in menu
                else {"error": f"'{name}' is not available now; available: {', '.join(menu)}"}
            )
            trace.append({"turn": turn, "raw": raw, "call": call, "result": result})
            sources.append(as_text(result))
            messages.append(self.llm.tool_turn(name, as_text(result), assistant))
        forced = session.final is None
        if forced:
            session.forcing = True
            result = session.call("compose")
            sources.append(as_text(result))
            messages.append({"role": "user", "content": BUDGET_SPENT + as_text(result)})
        messages.append({"role": "user", "content": REPORT[wf]})
        report = self.llm.parse(self.llm.generate(messages, None, REPORT_TOKENS[wf]))[0]
        return {
            "case": case.case_id,
            "core": session.aimed.core_id if session.aimed else None,
            "workflow": wf,
            "calls": calls,
            "final": session.final,
            "forced": forced,
            "findings": session.findings,
            "report": report,
            "footer": FOOTER,
            "unsupported_numbers": unsupported_numbers(report, "\n".join(sources)),
            "trace": trace,
        }
