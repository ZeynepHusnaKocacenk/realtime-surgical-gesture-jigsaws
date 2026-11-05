"""Build a single static page (docs/index.html) from results/*, for GitHub Pages.

Only real results are shown. JIGSAWS sections say "awaiting dataset access" unless a non-synthetic
results.json exists for that task; synthetic-data runs are never displayed as results.

    python make_report.py
"""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
RES, DOCS = ROOT / "results", ROOT / "docs"
REPO = "https://github.com/ZeynepHusnaKocacenk/realtime-surgical-gesture-jigsaws"

CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#5b6475;--line:#d9dde5;--card:#f6f7f9;--acc:#1f5fbf}
@media (prefers-color-scheme:dark){:root{--bg:#10141b;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2a3140;--card:#171c26;--acc:#7fb0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.7rem;line-height:1.25}h2{margin-top:2.2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}
h3{margin-top:1.6rem}a{color:var(--acc)}.status{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0}
.tablewrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;font-size:.85rem;min-width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:var(--card)}figure{margin:16px 0}img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--mut);font-size:.85rem}.pending{color:var(--mut);font-style:italic}code{background:var(--card);padding:1px 5px;border-radius:4px}
"""


def md_to_html(md: str, shift: int = 1) -> str:
    """Tiny converter for the markdown these scripts emit: headings, tables, paragraphs, bullets."""
    out, lines, i = [], md.splitlines(), 0
    inline = lambda t: re.sub(r"`([^`]+)`", r"<code>\1</code>", re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", html.escape(t)))
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#")); out.append(f"<h{min(n + shift, 6)}>{inline(ln[n:].strip())}</h{min(n + shift, 6)}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|[-| :]+\|", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[1:]
            out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                                                           for r in body) + "</tbody></table></div>")
            continue
        elif ln.startswith("* "):
            items = []
            while i < len(lines) and lines[i].startswith("* "):
                items.append(inline(lines[i][2:])); i += 1
            out.append("<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"); continue
        elif ln.strip():
            para = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "* ")):
                i += 1; para.append(lines[i])
            out.append(f"<p>{inline(' '.join(para))}</p>")
        i += 1
    return "\n".join(out)


def figure(src: Path, caption: str) -> str:
    if not src.exists():
        return ""
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, DOCS / "figures" / src.name)
    return f'<figure><img src="figures/{src.name}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>'


def read_md(path: Path, drop_title: bool = True) -> str:
    t = path.read_text()
    return "\n".join(t.splitlines()[1:]) if drop_title else t


def jigsaws_section(task: str, folder: str) -> str:
    d = RES / folder
    rj = d / "results.json"
    if rj.exists():
        meta = json.load(open(rj)).get("meta", {})
        if not meta.get("synthetic", True):
            body = md_to_html(read_md(d / "results.md"), shift=2)
            figs = "".join(figure(f, f"JIGSAWS {task}: {f.stem.replace('_', ' ')}") for f in sorted((d / "figures").glob("*.png")))
            return f"<h3>{task}</h3>{body}{figs}"
    return f'<h3>{task}</h3><p class="pending">Awaiting dataset access. No results are shown for this task.</p>'


def main():
    DOCS.mkdir(exist_ok=True)
    daph = RES / "daphnet"
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real-time gesture and freeze detection</title><style>{CSS}</style></head><body><main>
<h1>From Offline to the Operating Room: real-time recognition on wearable and surgical kinematics</h1>
<div class="status"><b>Status.</b> JIGSAWS results pending dataset access; real results currently on Daphnet Freezing of Gait.
Code and full README: <a href="{REPO}">{REPO.replace('https://', '')}</a>. Nothing on this page is a clinical claim.</div>
<p>One real-time framework for two problems: causal sequence models that must answer within a fixed time budget as
data streams in. Offline accuracy is compared with what a live system can achieve, with latency, calibration,
abstention and sensor-fault robustness measured on real data.</p>"""]
    if (RES / "daphnet_summary.md").exists():
        parts.append("<h2>Headline: freezing-of-gait detection with uncertainty</h2>")
        parts.append(md_to_html(read_md(RES / "daphnet_summary.md"), shift=2))
        parts.append(figure(RES / "daphnet_summary.png", "Online event F1 with 95% CIs over the 8 held-out patients"))
    demo = DOCS / "demo"
    if any(demo.glob("stream_*.png")):
        parts.append("<h2>Real-data streaming demo</h2>")
        for img in sorted(demo.glob("stream_*.png")):
            parts.append(f'<figure><img src="demo/{img.name}" alt="streamed patient" loading="lazy"><figcaption>'
                         f'Held-out patient streamed frame by frame (demo_daphnet.py)</figcaption></figure>')
        for md in sorted(demo.glob("session_*.md")):
            parts.append(md_to_html(md.read_text(), shift=1))
    parts.append("<h2>Daphnet Freezing of Gait (real data): full benchmark</h2>")
    if (daph / "results.md").exists():
        parts.append(md_to_html(read_md(daph / "results.md"), shift=2))
        parts.append(figure(daph / "figures" / "offline_vs_online.png", "Event F1, offline vs. online, with the Freeze Index baseline"))
        parts.append(figure(daph / "figures" / "per_subject_f1.png", "Online event F1 per held-out subject"))
        parts.append(figure(daph / "figures" / "recall_vs_false_alarms.png", "Online operating points"))
        parts.append(figure(daph / "figures" / "latency_vs_history.png", "Per-update latency against elapsed recording time"))
    else:
        parts.append('<p class="pending">Results not generated yet.</p>')
    parts.append("<h2>Calibration, abstention and sensor-fault robustness</h2>")
    if (daph / "robustness.md").exists():
        parts.append(md_to_html(read_md(daph / "robustness.md"), shift=2))
        parts.append(figure(daph / "figures" / "robustness.png", "Freeze event F1 against fault severity"))
        parts.append(figure(daph / "figures" / "abstention.png", "Coverage against event-level performance"))
    else:
        parts.append('<p class="pending">Results not generated yet.</p>')
    parts.append("<h2>Deployment (design proposal)</h2>")
    if (RES / "deployment.md").exists():
        parts.append(md_to_html(read_md(RES / "deployment.md"), shift=2))
        parts.append(figure(RES / "deployment_latency.png", "Streaming latency by runtime"))
    else:
        parts.append('<p class="pending">Results not generated yet.</p>')
    parts.append("<h2>JIGSAWS surgical gesture and skill recognition</h2>")
    parts += [jigsaws_section("Suturing", "jigsaws_suturing"), jigsaws_section("Knot Tying", "jigsaws_knot_tying"),
              jigsaws_section("Needle Passing", "jigsaws_needle_passing")]
    parts.append(f'<h2>Reproduce</h2><p>Every setting lives in <code>configs/*.yaml</code>; <code>./run_all.sh</code> runs all '
                 f'configs and rebuilds this page. See the <a href="{REPO}">README</a>.</p></main></body></html>')
    (DOCS / "index.html").write_text("\n".join(parts))
    (DOCS / ".nojekyll").write_text("")
    print(f"[done] {DOCS / 'index.html'}")


if __name__ == "__main__":
    main()
