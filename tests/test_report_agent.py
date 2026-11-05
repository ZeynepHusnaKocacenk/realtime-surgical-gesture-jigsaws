"""The LangGraph report agent: facts, verification gate, revise loop and fallbacks (no network)."""
import numpy as np
import pytest

pytest.importorskip("langgraph")

from rtsurg.report_agent import (DISCLAIMER, TemplateLLM, analyze_session, template_report, verify_report,
                                 write_report)

RATE = 32.0


def _session(stuck=False):
    rng = np.random.default_rng(0)
    T = int(RATE * 120)
    labels = np.zeros(T, int); labels[640:960] = 1; labels[2000:2200] = 1
    prob = np.clip(0.05 + 0.85 * labels + rng.normal(0, 0.05, T), 0, 1)
    prob[3000:3040] = 0.8                                            # one false alarm
    raw = rng.normal(size=(T, 9))
    if stuck:
        raw[1000:1200, 1] = raw[1000, 1]
    return {"rate_hz": RATE, "prob": prob, "alarm": prob >= 0.5, "labels": labels, "raw": raw,
            "subject": "S99", "model": "TCN-causal", "threshold": 0.5, "debounce_s": 0.0, "latency_ms": 0.42}


def test_facts_match_the_signal():
    f = analyze_session(_session())
    assert f["n_alarms"] == 3 and f["annotated_episodes"] == 2 and f["episodes_detected"] == 2
    assert f["false_alarms"] == 1 and f["first_alarm_onsets_mmss"][0] == "00:20"
    assert f["quality_flags"] == []
    assert "possible stuck sensor" in analyze_session(_session(stuck=True))["quality_flags"]


def test_template_report_passes_its_own_verifier():
    for stuck in (False, True):
        f = analyze_session(_session(stuck))
        assert verify_report(template_report(f), f) == []


def test_verifier_catches_invented_numbers_times_and_clinical_advice():
    f = analyze_session(_session())
    good = template_report(f)
    assert any("not present in FACTS" in i for i in verify_report(good + "\nAbout 42 more were likely.", f))
    assert any("time stamp" in i for i in verify_report(good + "\nAnother at 13:37.", f))
    assert any("clinical" in i for i in verify_report(good + "\nThe dose should be adjusted.", f))
    assert any("missing sections" in i for i in verify_report(good.replace("### Overview", "Overview:"), f))


class _Scripted:
    """Fake writer: returns the scripted drafts in order, None means a refusal."""
    name = "scripted"

    def __init__(self, drafts):
        self.drafts, self.calls = list(drafts), []

    def generate(self, messages):
        self.calls.append(messages)
        d = self.drafts.pop(0)
        return None if d is None else {"text": d, "content": [{"type": "text", "text": d}]}


def test_graph_revises_a_hallucinated_draft_then_accepts_the_fix():
    f = analyze_session(_session())
    good = template_report(f)
    llm = _Scripted([good + "\nRoughly 97 % of freezes were caught.", good])
    out = write_report(_session(), llm)
    assert out["trace"] == ["analyze", "draft", "verify:1", "revise", "draft", "verify:pass", "finalize"]
    assert out["source"] == "scripted" and DISCLAIMER in out["report"] and "97" not in out["report"]
    second = llm.calls[1]                               # the revision is appended, history is untouched
    assert [m["role"] for m in second] == ["user", "assistant", "user"] and "97" in second[2]["content"]


def test_graph_falls_back_to_template_when_the_model_never_passes_or_refuses():
    bad = "### Overview\nThe patient needs a higher levodopa dose."
    out = write_report(_session(), _Scripted([bad, bad, bad]))
    assert out["source"] == "template (fallback)" and "levodopa" not in out["report"]
    out = write_report(_session(), _Scripted([None]))
    assert out["source"] == "template (fallback)" and out["trace"][-2:] == ["fallback", "finalize"]


def test_offline_writer_runs_end_to_end():
    out = write_report(_session(), TemplateLLM())
    assert out["trace"][-2:] == ["verify:pass", "finalize"] and "### Detected events" in out["report"]
