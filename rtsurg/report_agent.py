"""LangGraph agent that turns a detector session into a reviewer-ready summary, with guardrails.

Why an agent here, and why a constrained one. A freezing-of-gait detector produces thousands of
per-frame probabilities per session; a clinical research team wants a short readable summary
("how many episodes, when, how confident, was the sensor OK"). A language model is good at the
prose and bad at being trusted with numbers or with clinical judgement. So the graph splits the job:

    analyze  (deterministic)  per-frame output -> FACTS: events, timings, confidence, sensor checks
    draft    (Claude)         FACTS -> markdown summary, told to copy numbers verbatim
    verify   (deterministic)  every number in the draft must exist in FACTS; no diagnosis,
                              treatment or medication language; required sections present
    revise   (Claude)         gets the verifier's findings, rewrites (at most ``MAX_REVISIONS``)
    fallback (deterministic)  template report if the model refuses, is unavailable, or never passes
    finalize                  appends a fixed disclaimer and the provenance of the text

The language model never sees raw signals and never computes anything; the verifier is the gate,
not the model's good behaviour. With no API key (or ``offline=True``) the template writer stands in
for Claude, so the graph and its tests run without network access.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import numpy as np

from .daphnet import _runs

MODEL = "claude-opus-5-5"
MAX_REVISIONS = 2
DISCLAIMER = ("_Automatically generated from a research prototype's output. Not a medical device, not "
              "validated for clinical use, and not a diagnosis or treatment recommendation. A qualified "
              "person must review the underlying data._")
SECTIONS = ["Overview", "Detected events", "Signal quality and confidence", "Notes for the reviewer"]
BANNED = [r"diagnos", r"prescri", r"medicat", r"\bdos(e|age|ing)\b", r"levodopa", r"\btreat", r"therap",
          r"\bshould (increase|decrease|change|adjust|start|stop)", r"progression", r"worsen", r"improv(ed|ing) (his|her|their) condition",
          r"\bsuffer"]

SYSTEM_PROMPT = f"""You write short session summaries of an automated freezing-of-gait DETECTOR for a \
clinical research team that will review the raw data themselves. You are summarising what the \
software output, not the patient's health.

Rules:
- Use only the facts in the FACTS JSON. Copy every number exactly as it is written there (same \
rounding, same units). Do not compute new numbers, percentages, averages or totals.
- Describe detector output neutrally ("the detector raised 12 alarms"). Never diagnose, never \
suggest treatment, medication, dosing or changes to care, and never speculate about disease state.
- If a sensor-quality or confidence flag is present, say plainly that those alarms deserve extra scrutiny.
- Use exactly these markdown sections, as level-3 headings: {", ".join(SECTIONS)}. About 150 to 250 words.
- Do not add a disclaimer; one is appended automatically."""


# ---------------------------------------------------------------------------------------
# Deterministic analysis
# ---------------------------------------------------------------------------------------
def _mmss(frames: int, rate: float) -> str:
    s = int(round(frames / rate))
    return f"{s // 60:02d}:{s % 60:02d}"


def stuck_fraction(x: np.ndarray, rate: float, min_s: float = 1.0) -> float:
    """Fraction of frames inside a run of >= ``min_s`` seconds where some channel did not change."""
    if x is None or len(x) < 2:
        return 0.0
    same = np.zeros(len(x), bool)
    same[1:] = (np.abs(np.diff(x, axis=0)) < 1e-9).any(1)
    n = int(round(min_s * rate))
    flag = np.zeros(len(x), bool)
    for s, e in _runs(same):
        if e - s >= n:
            flag[max(s - 1, 0):e] = True
    return float(flag.mean())


def analyze_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """FACTS for the writer. ``session`` holds: rate_hz, prob [T], alarm [T] (bool, after the
    validation-chosen threshold and debounce), and optionally labels [T], raw [T, F], model,
    threshold, debounce_s, latency_ms, subject."""
    rate = float(session["rate_hz"])
    prob = np.asarray(session["prob"], float)
    alarm = np.asarray(session["alarm"], bool)
    T = len(prob)
    from .fog import MERGE_GAP_S, pred_events
    ev = pred_events(alarm, int(round(MERGE_GAP_S * rate)))   # same event definition as the benchmark
    dur = [(e - s) / rate for s, e in ev]
    conf = np.maximum(prob, 1 - prob)
    f: Dict[str, Any] = {
        "subject": session.get("subject", "unknown"),
        "detector": session.get("model", "detector"),
        "decision_threshold": round(float(session.get("threshold", 0.5)), 2),
        "debounce_s": round(float(session.get("debounce_s", 0.0)), 2),
        "duration_min": round(T / rate / 60, 1),
        "frame_rate_hz": round(rate, 1),
        "n_alarms": len(ev),
        "alarms_per_min": round(len(ev) / max(T / rate / 60, 1e-9), 2),
        "total_alarm_time_s": round(float(alarm.sum()) / rate, 1),
        "alarm_time_pct": round(100 * float(alarm.mean()), 1),
        "longest_alarm_s": round(max(dur), 1) if dur else 0.0,
        "median_alarm_s": round(float(np.median(dur)), 1) if dur else 0.0,
        "first_alarm_onsets_mmss": [_mmss(s, rate) for s, _ in ev[:8]],
        "low_confidence_pct": round(100 * float((conf < 0.7).mean()), 1),
        "alarms_with_low_confidence": int(sum(conf[s:e].mean() < 0.7 for s, e in ev)),
    }
    if session.get("latency_ms") is not None:
        f["median_latency_ms_per_frame"] = round(float(session["latency_ms"]), 2)
    if session.get("raw") is not None:
        f["stuck_sensor_pct"] = round(100 * stuck_fraction(np.asarray(session["raw"]), rate), 1)
    if session.get("labels") is not None:          # research setting: compare with the annotation
        from .fog import event_counts, summarize_counts
        y = np.asarray(session["labels"], int)
        c = event_counts([alarm], [y], rate)
        s = summarize_counts(c, rate)
        f.update({"annotated_episodes": c["n_true"], "episodes_detected": c["detected"],
                  "episodes_missed": c["n_true"] - c["detected"], "false_alarms": c["false_alarms"],
                  "event_f1_pct": round(s["event_f1"], 1) if s["event_f1"] == s["event_f1"] else None,
                  "median_onset_delay_ms": round(s["median_delay_ms"]) if s["median_delay_ms"] == s["median_delay_ms"] else None})
    flags = []
    if f.get("stuck_sensor_pct", 0) > 1.0:
        flags.append("possible stuck sensor")
    if f["low_confidence_pct"] > 25:
        flags.append("many low-confidence frames")
    f["quality_flags"] = flags
    return f


# ---------------------------------------------------------------------------------------
# Verification (the gate)
# ---------------------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _allowed_numbers(facts: Dict[str, Any]) -> set:
    out = set()

    def add(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            for d in range(0, 3):
                out.add(f"{float(v):.{d}f}")
            if float(v).is_integer():
                out.add(str(int(v)))
        elif isinstance(v, str):
            out.update(_NUM.findall(v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)
    for k, v in facts.items():
        add(v)
    return out


def verify_report(text: str, facts: Dict[str, Any]) -> List[str]:
    """Problems with a draft (an empty list means it passes)."""
    issues = []
    allowed = _allowed_numbers(facts)
    clean = re.sub(r"\d{2}:\d{2}", lambda m: m.group(0) if m.group(0) in facts.get("first_alarm_onsets_mmss", [])
                   else "BADTIME", text)
    if "BADTIME" in clean:
        issues.append("a time stamp that is not in first_alarm_onsets_mmss")
    clean = re.sub(r"\d{2}:\d{2}", " ", clean)
    bad = sorted({n for n in _NUM.findall(clean) if n.lstrip("-") not in allowed})
    if bad:
        issues.append("numbers not present in FACTS (copy them exactly or remove them): " + ", ".join(bad))
    low = text.lower()
    hits = [p for p in BANNED if re.search(p, low)]
    if hits:
        issues.append("clinical-advice or diagnostic language is not allowed (matched: " + ", ".join(hits) + ")")
    missing = [s for s in SECTIONS if not re.search(rf"^#+\s*{re.escape(s)}\s*$", text, re.M | re.I)]
    if missing:
        issues.append("missing sections: " + ", ".join(missing))
    if facts.get("quality_flags") and not any(w in low for w in ("scrutiny", "caution", "check", "review")):
        issues.append("quality flags are present but the draft does not ask the reviewer to check them")
    return issues


# ---------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------
def template_report(f: Dict[str, Any]) -> str:
    """Deterministic writer: the offline stand-in for Claude and the fallback when it fails."""
    L = ["### Overview",
         f"Session of {f['duration_min']} min from subject {f['subject']}, analysed by {f['detector']} at "
         f"{f['frame_rate_hz']} Hz (decision threshold {f['decision_threshold']}, debounce {f['debounce_s']} s).", "",
         "### Detected events",
         f"The detector raised {f['n_alarms']} alarms ({f['alarms_per_min']} per minute), covering "
         f"{f['total_alarm_time_s']} s ({f['alarm_time_pct']} % of the session). The longest lasted "
         f"{f['longest_alarm_s']} s and the median {f['median_alarm_s']} s."]
    if f["first_alarm_onsets_mmss"]:
        L.append("First alarm onsets (mm:ss): " + ", ".join(f["first_alarm_onsets_mmss"]) + ".")
    if "annotated_episodes" in f:
        L.append(f"Against the study annotation: {f['episodes_detected']} of {f['annotated_episodes']} annotated "
                 f"episodes overlapped an alarm (missed: {f['episodes_missed']}); false alarms: {f['false_alarms']} "
                 f"(event F1 {f['event_f1_pct']} %).")
    L += ["", "### Signal quality and confidence",
          f"{f['low_confidence_pct']} % of frames had low model confidence; {f['alarms_with_low_confidence']} alarms "
          f"were mostly low-confidence."]
    if "stuck_sensor_pct" in f:
        L.append(f"Possible stuck-sensor segments covered {f['stuck_sensor_pct']} % of the session.")
    if "median_latency_ms_per_frame" in f:
        L.append(f"Median processing time was {f['median_latency_ms_per_frame']} ms per frame.")
    L += ["", "### Notes for the reviewer"]
    L.append(("Flags raised: " + "; ".join(f["quality_flags"]) + ". Please review the affected alarms with extra scrutiny.")
             if f["quality_flags"] else "No signal-quality flags were raised. Please review the alarms against the raw recording.")
    return "\n".join(L)


class TemplateLLM:
    """Offline writer with the same interface as ``ClaudeLLM``."""
    name = "template (offline)"

    def generate(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        facts = json.loads(messages[0]["content"].split("FACTS:\n", 1)[1])
        text = template_report(facts)
        return {"text": text, "content": [{"type": "text", "text": text}]}


class ClaudeLLM:
    """Claude via the official SDK. Server-side refusal fallback is enabled (``fallbacks="default"``);
    a request that is still refused, or any API error, returns None and the graph falls back to the
    template writer."""

    def __init__(self, model: str = MODEL, effort: str = "low", client=None):
        import anthropic
        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model, self.effort, self.name = model, effort, model

    def generate(self, messages):
        a = self.anthropic
        try:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=4000, system=SYSTEM_PROMPT, messages=messages,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError) as e:
            print(f"[agent] request rejected ({type(e).__name__}): {getattr(e, 'message', e)}")
            return None
        except (a.RateLimitError, a.APIStatusError, a.APIConnectionError) as e:
            print(f"[agent] API unavailable ({type(e).__name__}); using the template writer")
            return None
        if r.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        # keep every block (thinking included) so a revision turn replays the history unchanged
        return {"text": text, "content": [b.model_dump(exclude_none=True) for b in r.content]}


def default_llm(offline: bool = False):
    if offline or not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return TemplateLLM()
    return ClaudeLLM()


# ---------------------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------------------
class ReportState(TypedDict, total=False):
    session: Dict[str, Any]
    facts: Dict[str, Any]
    messages: List[Dict[str, Any]]
    draft: str
    issues: List[str]
    attempts: int
    source: str
    report: str
    trace: List[str]


def build_graph(llm=None):
    from langgraph.graph import END, START, StateGraph
    llm = llm or default_llm()

    def analyze(s: ReportState):
        f = analyze_session(s["session"])
        msg = {"role": "user", "content": "Write the session summary.\n\nFACTS:\n" + json.dumps(f, indent=1)}
        return {"facts": f, "messages": [msg], "attempts": 0, "trace": ["analyze"]}

    def draft(s: ReportState):
        out = llm.generate(s["messages"])
        if out is None:
            return {"draft": "", "issues": ["writer unavailable or refused"], "attempts": MAX_REVISIONS + 1,
                    "trace": s["trace"] + ["draft:failed"]}
        return {"draft": out["text"], "messages": s["messages"] + [{"role": "assistant", "content": out["content"]}],
                "attempts": s.get("attempts", 0) + 1, "trace": s["trace"] + ["draft"]}

    def verify(s: ReportState):
        if not s.get("draft"):
            return {"trace": s["trace"] + ["verify:skip"]}
        issues = verify_report(s["draft"], s["facts"])
        return {"issues": issues, "trace": s["trace"] + [f"verify:{'pass' if not issues else len(issues)}"]}

    def revise(s: ReportState):
        note = ("Your summary failed automatic checks. Rewrite the whole summary and fix:\n- "
                + "\n- ".join(s["issues"]))
        return {"messages": s["messages"] + [{"role": "user", "content": note}], "trace": s["trace"] + ["revise"]}

    def fallback(s: ReportState):
        return {"draft": template_report(s["facts"]), "source": "template (fallback)", "issues": [],
                "trace": s["trace"] + ["fallback"]}

    def finalize(s: ReportState):
        src = s.get("source") or getattr(llm, "name", "llm")
        head = f"## Detector session summary: {s['facts']['subject']}\n\n"
        tail = f"\n\n{DISCLAIMER}\n\n_Text: {src}; every number was checked against the detector output._"
        return {"report": head + s["draft"].strip() + tail, "source": src, "trace": s["trace"] + ["finalize"]}

    def route(s: ReportState):
        if not s.get("issues"):
            return "finalize"
        return "revise" if s.get("attempts", 0) <= MAX_REVISIONS else "fallback"

    g = StateGraph(ReportState)
    for name, fn in [("analyze", analyze), ("draft", draft), ("verify", verify), ("revise", revise),
                     ("fallback", fallback), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze"); g.add_edge("analyze", "draft"); g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route, {"finalize": "finalize", "revise": "revise", "fallback": "fallback"})
    g.add_edge("revise", "draft"); g.add_edge("fallback", "finalize"); g.add_edge("finalize", END)
    return g.compile()


def write_report(session: Dict[str, Any], llm=None) -> ReportState:
    return build_graph(llm).invoke({"session": session})
