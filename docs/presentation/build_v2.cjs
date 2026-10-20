// Builds docs/presentation/Team13-Midterm-v2.pptx (2 slides + speaker notes).
// Usage from the repository root: npm install --no-save pptxgenjs && node docs/presentation/build_v2.cjs
// Numbers come from results/midterm/retrieval_eval_20261019-023503/summary.json.
const path = require('node:path');
const pptxgen = require('pptxgenjs');

const INK = '1F2A30', MUTED = '5B6770', TEAL = '0F7C80', TEAL_TINT = 'E3F1F1';
const GRAY_TINT = 'F1F3F4', AMBER = 'D98E04', LINE = 'B8C2C8', WHITE = 'FFFFFF';

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625 in
pres.title = 'Team 13 Midterm: Cost-aware adaptive retrieval';
pres.author = 'Team 13';
pres.theme = {headFontFace: 'Calibri', bodyFontFace: 'Calibri'};

const text = (slide, value, opts) => slide.addText(value, {isTextBox: true, margin: 0, fontFace: 'Calibri', color: INK, valign: 'top', ...opts});

function header(slide, kicker, title, subtitle) {
  text(slide, kicker, {x: 0.5, y: 0.32, w: 9, h: 0.25, fontSize: 11, bold: true, color: TEAL, charSpacing: 1});
  text(slide, title, {x: 0.5, y: 0.58, w: 9, h: 0.55, fontSize: 28, bold: true});
  text(slide, subtitle, {x: 0.5, y: 1.13, w: 9, h: 0.3, fontSize: 14, color: MUTED});
}

// ---------- Slide 1: problem, method, differentiation ----------
const s1 = pres.addSlide();
s1.background = {color: WHITE};
header(s1, 'TEAM 13  ·  CSCI 544 MIDTERM', 'Match retrieval effort to the question',
  'Enterprise support assistant: a cited answer from company documents, or a refusal when evidence is weak');

const steps = [
  {n: '1', title: 'Check verified cache', sub: 'Repeat? Reuse the answer', built: false},
  {n: '2', title: 'Retrieve + rerank', sub: 'MiniLM search built today', built: true},
  {n: '3', title: 'Check the evidence', sub: 'Weak? Search once more', built: false},
  {n: '4', title: 'Answer or refuse', sub: 'Cite sources, or decline', built: false},
];
const boxW = 2.0, boxH = 1.05, gap = 0.33, boxY = 1.75, x0 = 0.5;
steps.forEach((step, i) => {
  const x = x0 + i * (boxW + gap);
  s1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y: boxY, w: boxW, h: boxH, rectRadius: 0.08, objectName: `step-${step.n}`,
    fill: {color: step.built ? TEAL : WHITE},
    line: {color: step.built ? TEAL : LINE, width: 1.25, dashType: step.built ? 'solid' : 'dash'},
  });
  const fg = step.built ? WHITE : INK;
  text(s1, step.n, {x: x + 0.15, y: boxY + 0.12, w: 0.3, h: 0.3, fontSize: 14, bold: true, color: step.built ? WHITE : TEAL});
  text(s1, step.title, {x: x + 0.15, y: boxY + 0.38, w: boxW - 0.3, h: 0.3, fontSize: 14, bold: true, color: fg});
  text(s1, step.sub, {x: x + 0.15, y: boxY + 0.68, w: boxW - 0.3, h: 0.3, fontSize: 11, color: step.built ? WHITE : MUTED});
  if (i < steps.length - 1) {
    s1.addShape(pres.shapes.LINE, {x: x + boxW + 0.05, y: boxY + boxH / 2, w: gap - 0.1, h: 0,
      line: {color: MUTED, width: 1.5, endArrowType: 'triangle'}});
  }
});
text(s1, [
  {text: '■ ', options: {color: TEAL}}, {text: 'built      ', options: {color: MUTED}},
  {text: '□ ', options: {color: MUTED}}, {text: 'planned', options: {color: MUTED}},
], {x: 0.5, y: 2.92, w: 4, h: 0.25, fontSize: 11});

const cardY = 3.4, cardH = 1.65, cardW = 4.38;
s1.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: 0.5, y: cardY, w: cardW, h: cardH, rectRadius: 0.08, fill: {color: GRAY_TINT}, line: {color: GRAY_TINT}, objectName: 'prior-work'});
text(s1, 'PRIOR WORK', {x: 0.72, y: cardY + 0.2, w: 4, h: 0.25, fontSize: 11, bold: true, color: MUTED, charSpacing: 1});
text(s1, [
  {text: 'Adaptive-RAG ', options: {bold: true}}, {text: '(Jeong et al., 2024): routes by question complexity', options: {breakLine: true}},
  {text: 'Self-RAG ', options: {bold: true}}, {text: '(Asai et al., 2024): learns when to retrieve, critiques itself'},
], {x: 0.72, y: cardY + 0.52, w: cardW - 0.44, h: 1.0, fontSize: 14, paraSpaceAfter: 6});

const ourX = 0.5 + cardW + 0.24;
s1.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: ourX, y: cardY, w: cardW, h: cardH, rectRadius: 0.08, fill: {color: TEAL_TINT}, line: {color: TEAL_TINT}, objectName: 'our-angle'});
text(s1, 'OUR ANGLE', {x: ourX + 0.22, y: cardY + 0.2, w: 4, h: 0.25, fontSize: 11, bold: true, color: TEAL, charSpacing: 1});
text(s1, [
  {text: 'They decide when to retrieve. We decide '},
  {text: 'how much effort', options: {bold: true}},
  {text: ' an enterprise question deserves (reuse, retry once, or refuse) and measure quality against latency and cost.'},
], {x: ourX + 0.22, y: cardY + 0.52, w: cardW - 0.44, h: 1.0, fontSize: 14});

s1.addNotes(`Speaking target: 55 seconds.

Support teams want fast, correct answers from company documents, but running the full retrieval pipeline on every question is slow and costly, and answering from weak evidence is risky.

Our idea is to match retrieval effort to the question. First, check a cache of verified answers: a repeat question reuses its answer. Otherwise, retrieve and rerank evidence. Then check that evidence. If it is weak, search once more, at most once. Finally, answer with citations, or refuse if the evidence is still insufficient.

Today the retrieval step is built: a local MiniLM search over company documents. The other steps are planned.

Prior work such as Adaptive-RAG routes by question complexity, and Self-RAG learns when to retrieve and critiques itself. Both decide when to retrieve. We decide how much effort an enterprise question deserves, including reusing an answer or refusing, and we measure the quality-versus-cost tradeoff. Cost savings are still a hypothesis.

Sources:
Jeong et al., Adaptive-RAG, NAACL 2024: https://aclanthology.org/2024.naacl-long.389/
Asai et al., Self-RAG, ICLR 2024: https://arxiv.org/abs/2310.11511`);

// ---------- Slide 2: takeaway and remaining work ----------
const s2 = pres.addSlide();
s2.background = {color: WHITE};
header(s2, 'WHAT WE HAVE LEARNED', 'The right document is not always the answer',
  'MiniLM baseline on an EnterpriseRAG-Bench subset: 100 questions, 290 documents, 3,358 passages');

const types = ['All questions', 'Constrained', 'Basic', 'Intra-doc reasoning', 'Semantic'];
const hit5 = [84.4, 100, 91.4, 86.7, 64.0];
s2.addChart(pres.charts.BAR, [{name: 'Hit@5', labels: types, values: hit5}], {
  x: 0.4, y: 1.55, w: 4.7, h: 2.15, barDir: 'bar', objectName: 'hit5-by-type',
  chartColors: [INK, LINE, LINE, LINE, AMBER],
  catAxisOrientation: 'maxMin', valAxisHidden: true, valAxisMaxVal: 115, valAxisMinVal: 0,
  valGridLine: {style: 'none'}, catGridLine: {style: 'none'},
  catAxisLabelColor: INK, catAxisLabelFontSize: 11, catAxisLabelFontFace: 'Calibri', catAxisLineShow: false,
  showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0"%"', dataLabelColor: INK,
  dataLabelFontSize: 11, dataLabelFontFace: 'Calibri', barGapWidthPct: 45,
  showTitle: true, title: 'Gold document in top 5 (MRR 0.67)', titleFontSize: 12, titleColor: MUTED, titleFontFace: 'Calibri',
  showLegend: false,
});

s2.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: 0.5, y: 3.85, w: 4.55, h: 1.15, rectRadius: 0.08, fill: {color: GRAY_TINT}, line: {color: GRAY_TINT}, objectName: 'example'});
text(s2, [
  {text: 'Example: ', options: {bold: true}},
  {text: 'the correct email thread ranked #1, but its best passage starts just after the answer (a five-day runbook turnaround). A document hit is not yet usable evidence.'},
], {x: 0.7, y: 4.0, w: 4.15, h: 0.9, fontSize: 13});

text(s2, 'BEFORE THE FINAL REPORT', {x: 5.45, y: 1.6, w: 4, h: 0.25, fontSize: 11, bold: true, color: TEAL, charSpacing: 1});
const next = [
  'Keyword + MiniLM search with a reranker',
  'Grounded answers with citations, plus refusal',
  'Evidence check, verified cache, and one retry',
  'Plot answer quality against latency and cost',
  'Grow to ~300 questions, incl. multi-document',
];
next.forEach((item, i) => {
  const y = 2.0 + i * 0.52;
  text(s2, String(i + 1), {x: 5.45, y, w: 0.3, h: 0.4, fontSize: 14, bold: true, color: TEAL});
  text(s2, item, {x: 5.8, y, w: 3.75, h: 0.4, fontSize: 14});
});

text(s2, 'Exploratory: splits pooled; 90 answerable questions scored, 10 unanswerable excluded. Answer quality and cost savings are not measured yet.',
  {x: 0.5, y: 5.17, w: 9, h: 0.25, fontSize: 10, color: MUTED});

s2.addNotes(`Speaking target: 65 seconds, then about one minute of questions.

We built a reproducible retrieval baseline on a subset of EnterpriseRAG-Bench: 100 questions and 290 documents, split into 3,358 passages and indexed with MiniLM. For 84 percent of the 90 answerable questions, the correct document is in the top five. MRR is 0.67. Constrained and basic questions do well, but semantic questions, the paraphrased ones, drop to 64 percent. A rerun reproduced every rank exactly.

Our main takeaway: finding the right document is not the same as finding the answer. In one example the correct email thread ranked first, but its best passage starts just after the five-business-day detail the question asks for. So retrieval scores alone overstate readiness, and we need to check evidence before answering.

Before the final report: hybrid keyword plus MiniLM search with a reranker for the semantic gap; grounded answers with citations and refusal; the evidence check, verified cache, and one retry; then a plot of answer quality against latency and cost. We will also grow to about 300 questions, including multi-document ones.

Results: https://github.com/AdithNG/adaptive-enterprise-retrieval/blob/midterm-presentation/results/midterm/retrieval_eval_20261019-023503/summary.json
Dataset: https://github.com/onyx-dot-app/EnterpriseRAG-Bench

Q&A:
What is Hit@5? The share of answerable questions with a correct document among the top five ranked documents.
What is MRR? The average of 1/rank of the first correct document. It is not the average rank.
Why 290 documents? 90 gold documents plus 200 related documents chosen as hard negatives. Full-corpus performance is not established.
Have you reduced cost? Not yet. This run measures retrieval only; adaptive comparisons come next.
Do you evaluate refusal? Not yet. Ten unanswerable questions are logged but excluded from retrieval scores.
Held-out results? Test Hit@5 is 12/15, a small split. Treat it as descriptive; no tuning was done.
Multi-document questions? None in the current subset; we plan to add them.
Failure example: qst_0187 (harmful-output warnings over two evaluation windows): the correct document ranks sixth, behind general incident guidance.`);

const out = path.join(__dirname, 'Team13-Midterm-v2.pptx');
pres.writeFile({fileName: out}).then(() => console.log(`Wrote ${out}`));
