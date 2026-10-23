import argparse
import json
import math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


# figure code reference from VLM4Rec
def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "outputs/combinations-dev-20261020-01"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/pdf/retrieval-figure-4-combinations.pdf",
    )
    args = parser.parse_args()
    balanced = json.loads((args.run / "balanced_summary.json").read_text())
    results = balanced["results"]
    meta = json.loads((args.run / "metadata.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "bm25s": "BM25S",
        "flat": "FAISS exact",
        "hybrid": "Hybrid",
        "lamer_concise": "Concise LameR",
        "hybrid_lamer_concise": "Hybrid + LameR",
        "hybrid_lamer_gated": "Selective hybrid + LameR",
    }
    colors = {
        "bm25s": "#B19A68",
        "flat": "#607F8C",
        "hybrid": "#7FA397",
        "lamer_concise": "#9E6C5B",
        "hybrid_lamer_concise": "#C28B77",
        "hybrid_lamer_gated": "#7D8264",
    }
    order = [
        "bm25s",
        "flat",
        "hybrid",
        "lamer_concise",
        "hybrid_lamer_concise",
        "hybrid_lamer_gated",
    ]
    if set(order) != set(results):
        parser.error("Figure 4 requires the six-method combination suite")
    question_count = len(meta["selected_questions"])
    analyzed = json.loads((args.run / "analysis.json").read_text())
    answerable_count = sum(
        row["recall@5"] is not None
        for row in analyzed["methods"]["flat"]["query_quality"].values()
    )
    pdf = canvas.Canvas(str(args.output), pagesize=(1100, 760))
    pdf.setTitle("Figure 4 - Combined retrieval tradeoffs")

    def text(x, y, s, size=11, col="#474747", align="left"):
        pdf.setFillColor(HexColor(col))
        pdf.setFont("Helvetica", size)
        getattr(
            pdf,
            {
                "left": "drawString",
                "right": "drawRightString",
                "center": "drawCentredString",
            }[align],
        )(x, y, s)

    def line(x, y, u, v, col="#DFDFDA", width=0.7):
        pdf.setStrokeColor(HexColor(col))
        pdf.setLineWidth(width)
        pdf.line(x, y, u, v)

    text(550, 720, "Figure 4. Combining retrieval optimizations", 23, align="center")
    text(
        550,
        695,
        "Upper-left is better | Baselines rerun in the same experiment | Arrows: hybrid to expanded to selective",
        11,
        align="center",
    )
    for panel, (metric, title, low, high) in enumerate(
        [
            ("ndcg@5", "Ranking quality", 0.70, 0.95),
            ("recall@5", "Relevant-document retrieval", 0.85, 1.0),
        ]
    ):
        x = 90 + panel * 535
        y = 315
        w = 405
        h = 305
        fx = lambda value: x + w * (math.log10(value) + 1) / 5
        fy = lambda value: y + h * (value - low) / (high - low)
        text(x + w / 2, 650, title, 16, align="center")
        for t in [0.1, 1, 10, 100, 1000, 10000]:
            xx = fx(t)
            line(xx, y, xx, y + h)
            text(xx, y - 19, f"{t:g}", 10, align="center")
        for i in range(6):
            value = low + (high - low) * i / 5
            yy = fy(value)
            line(x, yy, x + w, yy)
            text(x - 9, yy - 3, f"{value:.2f}", 10, align="right")
        text(
            x + w / 2,
            y - 46,
            "Mean total retrieval latency (ms, log scale)",
            11,
            align="center",
        )
        pdf.saveState()
        pdf.translate(x - 50, y + h / 2)
        pdf.rotate(90)
        text(0, 0, "nDCG@5" if panel == 0 else "Recall@5", 12, align="center")
        pdf.restoreState()
        for left, right in [
            ("hybrid", "hybrid_lamer_concise"),
            ("hybrid_lamer_concise", "hybrid_lamer_gated"),
        ]:
            ax, ay = fx(results[left]["mean_ms"]), fy(results[left][metric])
            bx, by = fx(results[right]["mean_ms"]), fy(results[right][metric])
            line(ax, ay, bx, by, colors[right], 1.3)
            theta = math.atan2(by - ay, bx - ax)
            for off in [-0.5, 0.5]:
                line(
                    bx,
                    by,
                    bx - 8 * math.cos(theta + off),
                    by - 8 * math.sin(theta + off),
                    colors[right],
                    1.3,
                )
        positions = []
        for i, k in enumerate(order):
            xx, yy = fx(results[k]["mean_ms"]), fy(results[k][metric])
            pdf.setFillColor(HexColor(colors[k]))
            pdf.circle(xx, yy, 4.3, fill=1, stroke=0)
            # Numeric labels use collision-aware callouts; dots retain exact measured coordinates.
            dx, dy = 0, 15
            while any(
                abs(xx + dx - px) < 18 and abs(yy + dy - py) < 18
                for px, py in positions
            ):
                dx += 18
                dy += 3
            positions.append((xx + dx, yy + dy))
            line(xx, yy, xx + dx, yy + dy - 3, colors[k], 0.5)
            text(xx + dx, yy + dy, str(i + 1), 11, colors[k], align="center")
    # Compact shared legend, including measured means for precise reading.
    for i, k in enumerate(order):
        x = 80 + (i % 3) * 340
        y = 232 - (i // 3) * 49
        pdf.setFillColor(HexColor(colors[k]))
        pdf.circle(x, y + 3, 4, fill=1, stroke=0)
        text(x + 12, y, f"{i+1}. {labels[k]}", 11)
        text(
            x + 12,
            y - 17,
            f"{results[k]['mean_ms']:,.2f} ms  |  Recall {results[k]['recall@5']:.3f}  |  nDCG {results[k]['ndcg@5']:.3f}",
            9,
        )
        text(x + 12, y - 30, f"p95 latency: {results[k]['p95_ms']:,.2f} ms", 9)
    gate = results["hybrid_lamer_gated"]["gate_rate"]
    text(
        40,
        116,
        f"Selective expansion fired on {gate:.1%} of requests; otherwise it returned the original hybrid ranking.",
        11,
    )
    text(
        40,
        94,
        f"{len(balanced['complete_repeats'])} complete passes; {balanced['analyzed_requests']:,} requests. Quality: {answerable_count} answerable queries; timing: {question_count} queries.",
        10,
    )
    text(
        40,
        73,
        "Warm-model sequential timings include encoding and expansion. Native prompt caching may occur; final answer generation is excluded.",
        10,
    )
    text(
        40,
        52,
        "These are development observations, not established generalization gains. Paired bootstrap intervals are saved in analysis.json.",
        10,
    )
    text(
        1060,
        26,
        f"Source: {args.run.name} | {meta['data']['chunks']:,} chunks",
        9,
        align="right",
    )
    pdf.save()
    print(args.output)


if __name__ == "__main__":
    main()
