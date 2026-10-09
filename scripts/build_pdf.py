"""Сборка PDF отчёта и слайдов. `python scripts/build_pdf.py` собирает оба документа.

Отчёт: исходник docs/report/report.tex, компилятор tectonic (XeLaTeX-движок, пакеты и шрифты докачивает сам);
результат копируется в docs/report.pdf и site/docs/report.pdf.
Слайды: docs/slides.md → HTML → headless Chrome; разделитель `---` = новый слайд, формат страницы 16:9;
результат docs/slides.pdf и site/docs/slides.pdf.
`python scripts/build_pdf.py docs/slides.md out.pdf` собирает один Markdown-файл через Chrome, как раньше.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("chrome"),
]

CSS_REPORT = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 20pt; margin: 0 0 6pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; font-size: 8.8pt; margin: 6pt 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; background: #f6f7f9; padding: 0 2pt; }
pre { background: #f6f7f9; padding: 6pt; font-size: 8.5pt; overflow-x: auto; white-space: pre-wrap; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #cbd5e1; margin: 6pt 0; padding: 2pt 10pt; color: #444; }
.pagebreak { page-break-after: always; }
"""

CSS_SLIDES = """
@page { size: 297mm 167mm; margin: 0; }
:root { --ink: #262626; --muted: #808080; --accent: #148F2B; --plate: #F5F6F7; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: var(--ink); }
section.slide { position: relative; width: 297mm; height: 167mm; box-sizing: border-box; padding: 16mm 16mm 20mm;
  page-break-after: always; overflow: hidden; display: flex; flex-direction: column; }
section.slide:last-child { page-break-after: auto; }
h1 { font-size: 34pt; line-height: 1.1; margin: 0 0 6mm; font-weight: 700; }
h2 { font-size: 24pt; line-height: 1.15; margin: 0 0 8mm; font-weight: 700; max-width: 230mm; }
p { font-size: 15pt; line-height: 1.4; margin: 0 0 4mm; }
.points p { margin: 0 0 5mm; max-width: 170mm; }
.points p:last-child { margin-bottom: 0; }
.split { display: grid; grid-template-columns: 4fr 6fr; gap: 10mm; flex: 1; min-height: 0; align-items: start; }
.split .points p { max-width: none; }
.split.even { grid-template-columns: 1fr 1fr; }
.points p.label { font-size: 12pt; color: var(--muted); font-weight: 600; margin-bottom: 3mm; }
section.slide > table { max-width: 230mm; }
.fig img { display: block; width: 100%; height: auto; max-height: 105mm; object-fit: contain; object-position: left top; }
.wide { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: flex-end; }
.wide img { display: block; max-width: 100%; max-height: 64mm; object-fit: contain; object-position: left bottom; }
.caption { font-size: 11pt; color: var(--muted); margin: 2mm 0 0; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8mm; }
.stat { background: var(--plate); border-radius: 24px; padding: 10mm 9mm; min-height: 60mm; box-sizing: border-box; }
.num { font-size: 48pt; line-height: 1; font-weight: 700; color: var(--accent); margin: 0 0 4mm; }
.stat p { font-size: 15pt; line-height: 1.35; margin: 0; }
.stat p.num, .plate p.num { font-size: 54pt; line-height: 1; margin: 0 0 6mm; }
.plate { background: var(--plate); border-radius: 24px; padding: 7mm 8mm; margin-bottom: 6mm; }
.plate p { font-size: 13pt; margin: 0; }
table { border-collapse: collapse; font-size: 12.5pt; width: 100%; }
th { text-align: left; font-weight: 600; color: #808080; font-size: 11pt; padding: 0 4mm 2.5mm 0; border-bottom: 1px solid #d9d9d9; }
td { padding: 2.4mm 4mm 2.4mm 0; border-bottom: 1px solid #ececec; vertical-align: top; }
.title-meta { margin-top: auto; }
.title-meta p { font-size: 15pt; margin: 0 0 2mm; }
.links p { font-size: 12pt; color: var(--muted); margin: 0 0 1.5mm; }
.sub { font-size: 18pt; color: var(--muted); }
.title-grid { margin-top: auto; display: grid; grid-template-columns: 1fr 100mm; gap: 10mm; align-items: end; }
.title-grid .title-meta { margin-top: 0; }
.title-right { display: flex; flex-direction: column; gap: 4mm; align-items: flex-end; }
.ill { background: #0f1512; border-radius: 24px; overflow: hidden; }
.ill img { display: block; width: 100%; height: 100%; object-fit: cover; }
.hero { width: 100mm; height: 40mm; }
.qrs { display: flex; gap: 4mm; }
.qr { background: #fff; border-radius: 24px; border: 1px solid #e6e8ea; text-align: center; box-sizing: border-box; }
.qr img { display: block; margin: 0 auto; }
.qr p { color: var(--muted); margin: 0; font-weight: 600; }
.qrs.small .qr { padding: 2mm 2mm 2.5mm; } .qrs.small .qr img { width: 32mm; height: 32mm; } .qrs.small .qr p { font-size: 9.5pt; }
.qrs.big { gap: 6mm; } .qrs.big .qr { padding: 4mm 4mm 5mm; } .qrs.big .qr img { width: 52mm; height: 52mm; } .qrs.big .qr p { font-size: 13pt; margin-top: 1mm; }
.s2 { display: grid; grid-template-columns: 1fr 88mm; gap: 8mm; flex: 1; min-height: 0; }
.s2 .side { display: flex; flex-direction: column; gap: 6mm; }
.s2 .side .stat { min-height: 0; flex: 1; padding: 6mm 7mm; }
.s2 .side .stat p.num { font-size: 36pt; margin: 0 0 3mm; }
.s2 .side .stat p { font-size: 12pt; }
.stat.big { min-height: 0; padding: 8mm 10mm; }
.stat.big p.lead { font-size: 13pt; color: var(--muted); margin: 0 0 3mm; } .stat.big p.lead b { color: var(--accent); font-size: 22pt; margin-right: 2mm; }
.tlist { list-style: none; margin: 0; padding: 0; }
.tlist li { display: flex; justify-content: space-between; font-size: 15pt; padding: 2.4mm 0; border-bottom: 1px solid #e3e5e8; }
.tlist li:last-child { border-bottom: 0; } .tlist b { color: var(--accent); }
p.label { font-size: 12pt; color: var(--muted); font-weight: 600; margin-bottom: 3mm; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5mm; }
.tile { background: var(--plate); border-radius: 24px; padding: 5mm 6mm; min-height: 40mm; box-sizing: border-box; }
.tile p { font-size: 12.5pt; line-height: 1.3; margin: 0; }
.tile p.tn { font-size: 28pt; font-weight: 700; color: var(--accent); line-height: 1; margin: 0 0 2.5mm; }
.tile.acc { background: #E6F4E9; } .tile.note { background: #fff; border: 1px solid #e3e5e8; } .tile.note p { color: var(--muted); font-size: 11pt; }
.tiles + .caption { margin-top: 6mm; font-size: 12.5pt; }
.layers { display: grid; grid-template-columns: 1fr 1fr 1fr 8mm 1.15fr; gap: 5mm; align-items: stretch; }
.lc { border-radius: 24px; padding: 6mm; min-height: 82mm; display: flex; flex-direction: column; }
.lc p { font-size: 12pt; line-height: 1.3; margin: 0 0 2mm; }
.lc p.lh { font-size: 15pt; font-weight: 700; margin-bottom: 2mm; }
.lc p.rule { font-size: 11.5pt; color: #262626; }
.lc p.deg { margin-top: auto; margin-bottom: 0; color: var(--muted); font-size: 11pt; } .lc p.deg b { font-size: 26pt; color: var(--accent); margin-right: 1.5mm; }
.c1 { background: #E6F4E9; } .c2 { background: #EAF2FB; } .c3 { background: #FBF1E4; }
.lc.snf { background: var(--plate); padding: 4mm; } .lc.snf .ill { height: 40mm; margin-bottom: 3mm; } .lc.snf p { padding: 0 2mm; }
.arrow { align-self: center; font-size: 26pt; color: var(--accent); text-align: center; }
.layers + .caption { margin-top: 6mm; font-size: 12.5pt; }
.tt { display: flex; align-items: center; gap: 3mm; } .tt img { width: 14mm; height: 9mm; object-fit: cover; border-radius: 8px; background: #0f1512; }
td { vertical-align: middle; }
.cases { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; flex: 1; min-height: 0; }
.case { background: var(--plate); border-radius: 24px; padding: 8mm 9mm; display: flex; flex-direction: column; justify-content: center; }
.case p { font-size: 14pt; margin: 0; line-height: 1.3; } .case p.lh { font-size: 14pt; font-weight: 700; margin-bottom: 2mm; }
.case p.cn { font-size: 40pt; font-weight: 700; color: var(--accent); line-height: 1.05; margin-bottom: 2mm; }
.endgrid { display: grid; grid-template-columns: 1fr auto; gap: 12mm; align-items: start; }
.endcol .points + .points { margin-top: 8mm; }
.legend + .layers .lc { min-height: 68mm; }
.legend { font-size: 12pt; color: var(--muted); margin: -4mm 0 5mm; max-width: 250mm; }
.takeaway { font-size: 16pt; font-weight: 700; margin: 6mm 0 0; }
.foot-note { font-size: 11pt; color: var(--muted); margin: 2mm 0 0; }
.wide.big img { max-height: 92mm; }
.dyn { display: grid; grid-template-columns: 1fr 1fr; gap: 3mm 8mm; flex: 1; min-height: 0; align-items: start; margin-top: -4mm; }
.dyn .points { grid-column: 1 / 3; } .dyn .points p { font-size: 13pt; max-width: none; margin-bottom: 1.5mm; }
.dyn .fig img { max-height: 84mm; margin: 0 auto; }
.tlist li.grp { font-size: 11pt; color: var(--muted); font-weight: 600; border-bottom: 0; padding: 3mm 0 0; }
.concl p { font-size: 17pt; margin-bottom: 5mm; } .concl p.label { font-size: 12pt; }
.hero.light { width: 100mm; height: auto; background: none; } .hero.light img { display: block; width: 100%; height: auto; }
.qa p.big { font-size: 22pt; line-height: 1.35; margin: 0 0 10mm; max-width: 250mm; } .qa p.label { margin-bottom: 2mm; }
.caption.clr { font-size: 14pt; color: var(--ink); }
.defs { font-size: 14pt; margin: -3mm 0 5mm; max-width: 260mm; } .defs + .layers .lc { min-height: 56mm; }
.sub2 { font-size: 14pt; margin: 1mm 0 0; }
.eta { font-size: 14pt; color: var(--muted); } .eta b { font-size: 30pt; color: var(--accent); margin-right: 2mm; }
.foot { position: absolute; left: 16mm; right: 16mm; bottom: 8mm; display: flex; justify-content: space-between;
  font-size: 9.5pt; color: var(--muted); }
"""

MATHJAX = '<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]]}};</script>' \
          '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'


def md_to_html(text: str, slides: bool = False) -> str:
    ext = ["tables", "fenced_code", "toc", "attr_list", "pymdownx.arithmatex"] + (["md_in_html"] if slides else [])
    return markdown.markdown(text, extensions=ext,
                             extension_configs={"pymdownx.arithmatex": {"generic": True}})


def build_html(md_path: pathlib.Path, slides: bool) -> str:
    text = md_path.read_text(encoding="utf-8")
    if slides:
        parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
        foot = "Pulsar, конкурс СберИндекса 2026"
        body = "\n".join(
            f'<section class="slide">{md_to_html(p, slides=True)}'
            f'<div class="foot"><span>{foot}</span><span>{i}</span></div></section>'
            for i, p in enumerate(parts, 1))
        css = CSS_SLIDES
    else:
        body = md_to_html(text)
        css = CSS_REPORT
    base = md_path.parent.resolve().as_uri() + "/"
    return f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><base href="{base}"><style>{css}</style>{MATHJAX}</head><body>{body}</body></html>'


def chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and pathlib.Path(c).exists():
            return c
    sys.exit("Chrome/Chromium не найден: установите или укажите путь в CHROME_CANDIDATES")


def to_pdf(md_path: pathlib.Path, pdf_path: pathlib.Path, slides: bool = False) -> None:
    html = build_html(md_path, slides)
    with tempfile.TemporaryDirectory() as td:
        h = pathlib.Path(td) / "doc.html"
        h.write_text(html, encoding="utf-8")
        cmd = [chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
               f"--print-to-pdf={pdf_path.resolve()}", h.resolve().as_uri()]
        subprocess.run(cmd, check=True, capture_output=True)
    print(f"→ {pdf_path} ({pdf_path.stat().st_size / 1e6:.2f} МБ)")


def build_report() -> None:
    tex = ROOT / "docs/report/report.tex"
    tectonic = shutil.which("tectonic")
    if not tectonic:
        sys.exit("tectonic не найден: отчёт собирается из docs/report/report.tex. "
                 "Установите tectonic (https://tectonic-typesetting.github.io, например `brew install tectonic`) "
                 "или соберите report.tex любым XeLaTeX.")
    subprocess.run([tectonic, "--chatter", "minimal", tex.name], cwd=tex.parent, check=True)
    built = tex.with_suffix(".pdf")
    for dst in (ROOT / "docs/report.pdf", ROOT / "site/docs/report.pdf"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built, dst)
    print(f"→ docs/report.pdf, site/docs/report.pdf ({built.stat().st_size / 1e6:.2f} МБ)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = pathlib.Path(sys.argv[1])
        dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
        to_pdf(src, dst, slides="slides" in src.name)
    else:
        build_report()
        slides = ROOT / "docs/slides.md"
        to_pdf(slides, ROOT / "docs/slides.pdf", slides=True)
        (ROOT / "site/docs").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs/slides.pdf", ROOT / "site/docs/slides.pdf")
