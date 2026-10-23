import argparse
import json
import math
import random
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color


# figure code reference from VLM4Rec
def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "outputs/suite-dev-20261020-01"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/pdf/retrieval-paper-figures.pdf"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads((args.run / "balanced_summary.json").read_text())["results"]
    analysis = json.loads((args.run / "analysis.json").read_text())
    labels = {
        "python": "Python cosine",
        "flat": "FAISS exact",
        "hnsw": "HNSW",
        "ivfpq": "IVF-PQ",
        "bm25_reference": "BM25 reference",
        "bm25s": "BM25S",
        "hybrid": "Hybrid RRF",
        "rerank": "Cross-encoder rerank",
        "gated_rerank": "Gated rerank",
        "query2doc_original": "Query2doc original",
        "query2doc_concise": "Query2doc concise",
        "lamer_original": "LameR original",
        "lamer_concise": "LameR concise",
        "lamer_rrf": "LameR + RRF",
    }
    if set(results) != set(labels):
        parser.error("Figures 1-3 require the full isolated suite")
    results = {method: results[method] for method in labels}
    balanced = json.loads((args.run / "balanced_summary.json").read_text())
    question_scores = analysis["methods"]["flat"]["query_quality"]
    question_count = len(question_scores)
    answerable_count = sum(
        row["recall@5"] is not None for row in question_scores.values()
    )
    pass_count = len(balanced["complete_repeats"])
    GRAY = "#AAA69F"
    BLUE = "#607F8C"
    SAGE = "#7FA397"
    CLAY = "#C28B77"
    RUST = "#9E6C5B"
    GOLD = "#B19A68"
    INK = "#474747"
    GRID = "#DFDFDA"
    colors = {k: GRAY for k in results}
    colors.update(
        flat=BLUE,
        bm25s=GOLD,
        hybrid=SAGE,
        query2doc_original=CLAY,
        query2doc_concise=CLAY,
        lamer_original=RUST,
        lamer_concise=SAGE,
    )
    pdf = canvas.Canvas(str(args.output), pagesize=(1100, 720))
    pdf.setTitle("Enterprise retrieval - development benchmark figures")

    def draw_text(x, y, s, size=11, col=INK, align="left"):
        pdf.setFillColor(HexColor(col))
        pdf.setFont("Helvetica", size)
        getattr(
            pdf,
            {
                "left": "drawString",
                "center": "drawCentredString",
                "right": "drawRightString",
            }[align],
        )(x, y, s)

    def line(x, y, u, v, col=GRID, width=0.7):
        pdf.setStrokeColor(HexColor(col))
        pdf.setLineWidth(width)
        pdf.line(x, y, u, v)

    def circle(x, y, col, r=4):
        pdf.setFillColor(HexColor(col))
        pdf.circle(x, y, r, fill=1, stroke=0)

    def header(title, sub):
        draw_text(550, 681, title, 23, align="center")
        draw_text(550, 654, sub, 11, align="center")

    def footer(a, b):
        draw_text(40, 51, a, 10)
        draw_text(40, 33, b, 10)
        draw_text(1060, 15, f"Source: {args.run.name}", 8, align="right")

    def confidence_interval(k, metric):
        vals = [
            q[metric]
            for q in analysis["methods"][k]["query_quality"].values()
            if q[metric] is not None
        ]
        rng = random.Random(42)
        means = sorted(
            sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(5000)
        )
        return means[124], means[4874]

    main = [
        "bm25s",
        "flat",
        "hybrid",
        "query2doc_original",
        "query2doc_concise",
        "lamer_original",
        "lamer_concise",
    ]

    def draw_quality():
        header(
            "Figure 1. Retrieval quality with uncertainty",
            f"Same {answerable_count} answerable questions | Higher is better",
        )
        for j, (metric, title) in enumerate(
            [("recall@5", "Recall@5"), ("ndcg@5", "nDCG@5")]
        ):
            x = 205 + j * 460
            w = 380
            draw_text(x + w / 2, 606, title, 16, align="center")
            for t in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                xx = x + w * t
                line(xx, 155, xx, 580)
                draw_text(xx, 136, f"{t:.1f}", 10, align="center")
            for i, k in enumerate(main):
                y = 548 - i * 56
                v = results[k][metric]
                if j == 0:
                    draw_text(x - 14, y - 3, labels[k], 11, align="right")
                pdf.setFillColor(HexColor(colors[k]))
                pdf.rect(x, y - 13, w * v, 26, fill=1, stroke=0)
                a, b = confidence_interval(k, metric)
                line(x + w * a, y, x + w * b, y, INK, 1)
                line(x + w * a, y - 4, x + w * a, y + 4, INK)
                line(x + w * b, y - 4, x + w * b, y + 4, INK)
                draw_text(x + w * v, y + 19, f"{v:.3f}", 10, align="center")
            draw_text(x + w / 2, 108, "Mean score", 12, align="center")
        footer(
            f"Bars: query-averaged scores over {pass_count} passes. Whiskers: 95% query-bootstrap intervals (5,000 resamples).",
            "Intervals describe uncertainty in each mean; use paired differences to assess improvements. No held-out evaluation is shown.",
        )
        pdf.showPage()

    def draw_overview():
        header(
            "Figure 2. All variants: retrieval quality and latency",
            "Quality heatmap uses one shared 0-1 scale | Latency uses a logarithmic scale",
        )
        x = 225
        y0 = 573
        rh = 31
        cw = 117
        for j, t in enumerate(["Recall@5", "MRR@5", "nDCG@5"]):
            draw_text(x + (j + 0.5) * cw, 603, t, 13, align="center")
        start = HexColor("#F3EFE7")
        end = HexColor("#718F89")
        for i, k in enumerate(results):
            y = y0 - i * rh
            draw_text(x - 14, y - 4, labels[k], 11, align="right")
            for j, metric in enumerate(["recall@5", "mrr@5", "ndcg@5"]):
                v = results[k][metric]
                pdf.setFillColor(
                    Color(
                        start.red + (end.red - start.red) * v,
                        start.green + (end.green - start.green) * v,
                        start.blue + (end.blue - start.blue) * v,
                    )
                )
                pdf.rect(x + j * cw, y - rh / 2, cw, rh, fill=1, stroke=0)
                draw_text(
                    x + (j + 0.5) * cw,
                    y - 4,
                    f"{v:.3f}",
                    11,
                    "#FFFFFF" if v > 0.73 else INK,
                    "center",
                )
            if i in [3, 5, 8]:
                line(x, y - rh / 2, x + 3 * cw, y - rh / 2, "#FFFFFF", 2)
        # heatmap key
        for j in range(100):
            v = j / 99
            pdf.setFillColor(
                Color(
                    start.red + (end.red - start.red) * v,
                    start.green + (end.green - start.green) * v,
                    start.blue + (end.blue - start.blue) * v,
                )
            )
            pdf.rect(x + j * 3.51, 106, 3.6, 8, fill=1, stroke=0)
        draw_text(x, 89, "0", 9)
        draw_text(x + 3 * cw, 89, "1", 9, align="right")
        draw_text(x + 1.5 * cw, 89, "Quality score; darker = higher", 9, align="center")
        lx = 650
        w = 305
        fx = lambda v: lx + w * (math.log10(v) + 1) / 5
        draw_text(lx + w / 2, 603, "Total retrieval latency (ms)", 13, align="center")
        for t in [0.1, 1, 10, 100, 1000, 10000]:
            xx = fx(t)
            line(xx, 145, xx, 590)
            draw_text(xx, 126, f"{t:g}", 9, align="center")
        for i, k in enumerate(results):
            y = y0 - i * rh
            m = results[k]["mean_ms"]
            p95 = results[k]["p95_ms"]
            line(fx(m), y, fx(p95), y, colors[k], 2)
            circle(fx(m), y, colors[k])
            pdf.setStrokeColor(HexColor(colors[k]))
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.circle(fx(p95), y, 3.5, stroke=1, fill=1)
            draw_text(1058, y - 3, f"{m:,.2f}", 10, align="right")
        draw_text(1058, 603, "Mean ms", 11, align="right")
        circle(688, 98, BLUE)
        draw_text(700, 94, "Mean", 10)
        pdf.setStrokeColor(HexColor(BLUE))
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.circle(775, 98, 3.5, stroke=1, fill=1)
        draw_text(787, 94, "p95", 10)
        footer(
            f"Quality: {answerable_count} answerable questions. Timing: {question_count} questions x {pass_count} complete passes.",
            "Latency includes encoding, expansion and reranking where used; excludes model startup, index construction and final answer generation.",
        )
        pdf.showPage()

    def draw_tradeoffs():
        header(
            "Figure 3. Quality-latency tradeoffs",
            "Upper-left is better | Arrows show original-to-concise prompt changes",
        )
        for panel, metric, yrange, title in [
            (0, "ndcg@5", (0.70, 0.90), "Ranking quality"),
            (1, "recall@5", (0.88, 1.0), "Relevant-document retrieval"),
        ]:
            x = 95 + panel * 530
            y = 195
            w = 415
            h = 365
            lo, hi = yrange
            fx = lambda v: x + w * (math.log10(v) + 1) / 5
            fy = lambda v: y + h * (v - lo) / (hi - lo)
            draw_text(x + w / 2, 602, title, 16, align="center")
            for t in [0.1, 1, 10, 100, 1000, 10000]:
                xx = fx(t)
                line(xx, y, xx, y + h)
                draw_text(xx, y - 20, f"{t:g}", 10, align="center")
            for i in range(5):
                v = lo + (hi - lo) * i / 4
                yy = fy(v)
                line(x, yy, x + w, yy)
                draw_text(x - 10, yy - 3, f"{v:.2f}", 10, align="right")
            draw_text(
                x + w / 2,
                y - 47,
                "Mean total retrieval latency (ms, log scale)",
                11,
                align="center",
            )
            pdf.saveState()
            pdf.translate(x - 53, y + h / 2)
            pdf.rotate(90)
            draw_text(0, 0, "nDCG@5" if panel == 0 else "Recall@5", 12, align="center")
            pdf.restoreState()
            for a, b in [
                ("query2doc_original", "query2doc_concise"),
                ("lamer_original", "lamer_concise"),
            ]:
                ax, ay = fx(results[a]["mean_ms"]), fy(results[a][metric])
                bx, by = fx(results[b]["mean_ms"]), fy(results[b][metric])
                line(ax, ay, bx, by, colors[a], 1.5)
                ang = math.atan2(by - ay, bx - ax)
                for da in [-0.45, 0.45]:
                    line(
                        bx,
                        by,
                        bx - 9 * math.cos(ang + da),
                        by - 9 * math.sin(ang + da),
                        colors[a],
                        1.5,
                    )
            offsets = (
                {
                    "bm25s": (10, -17),
                    "flat": (-13, -22),
                    "hybrid": (-10, 17),
                    "query2doc_original": (-15, -28),
                    "query2doc_concise": (-20, -29),
                    "lamer_original": (3, -17),
                    "lamer_concise": (-8, 20),
                }
                if panel == 0
                else {
                    "bm25s": (9, -22),
                    "flat": (-45, -32),
                    "hybrid": (14, -13),
                    "query2doc_original": (-48, -12),
                    "query2doc_concise": (-78, -25),
                    "lamer_original": (-50, 20),
                    "lamer_concise": (-100, 12),
                }
            )
            short = {
                "bm25s": "BM25S",
                "flat": "FAISS exact",
                "hybrid": "Hybrid",
                "query2doc_original": "Q2doc original",
                "query2doc_concise": "Q2doc concise",
                "lamer_original": "LameR original",
                "lamer_concise": "LameR concise",
            }
            for k in main:
                xx, yy = fx(results[k]["mean_ms"]), fy(results[k][metric])
                circle(xx, yy, colors[k], 4.3)
                dx, dy = offsets[k]
                line(xx, yy, xx + dx, yy + dy, colors[k], 0.5)
                draw_text(
                    xx + dx,
                    yy + dy,
                    short[k],
                    9,
                    col=colors[k],
                    align="left" if dx >= 0 else "right",
                )
        footer(
            "Points are measured configurations, not a continuous parameter sweep. Arrows connect paired prompt variants only.",
            "Small quality differences remain uncertain. Latency is not a measurement of dollar cost.",
        )

    draw_quality()
    draw_overview()
    draw_tradeoffs()
    pdf.save()
    print(args.output)


if __name__ == "__main__":
    main()
