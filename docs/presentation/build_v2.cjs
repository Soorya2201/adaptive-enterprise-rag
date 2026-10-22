// Builds docs/presentation/Team13-Midterm-v2.pptx (2 slides + speaker notes).
// Usage from the repository root: npm install --no-save pptxgenjs && node docs/presentation/build_v2.cjs
// Slide 2 numbers come from Ty's feat-retrieval-optimizations run (experiments/optimizations).
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
  {n: '2', title: 'Hybrid retrieval', sub: 'Vector + keyword search', built: true},
  {n: '3', title: 'Check the evidence', sub: 'Disagree? Search again', built: true},
  {n: '4', title: 'Answer or refuse', sub: 'Cite sources, or decline', built: true},
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

Our idea is to match retrieval effort to the question. First, check a cache of verified answers: a repeat question reuses its answer. Otherwise, retrieve evidence with hybrid search, combining vector and keyword search. Then check that evidence: if the two searches disagree, search once more, at most once. Finally, answer with citations, or refuse if the evidence is still insufficient.

Today steps 2 to 4 are built as prototypes. The verified cache is planned.

Prior work such as Adaptive-RAG routes by question complexity, and Self-RAG learns when to retrieve and critiques itself. Both decide when to retrieve. We decide how much effort an enterprise question deserves, including reusing an answer or refusing, and we measure the quality-versus-cost tradeoff. Cost savings are still a hypothesis.

Sources:
Jeong et al., Adaptive-RAG, NAACL 2024: https://aclanthology.org/2024.naacl-long.389/
Asai et al., Self-RAG, ICLR 2024: https://arxiv.org/abs/2310.11511`);

// ---------- Slide 2: takeaway and remaining work ----------
// Numbers: Ty's combination run (feat-retrieval-optimizations, experiments/optimizations figure 4).
const s2 = pres.addSlide();
s2.background = {color: WHITE};
header(s2, 'WHAT WE HAVE LEARNED', 'Search harder only when it is needed',
  'Extra LLM-assisted search runs only when keyword and vector search disagree on the top document');

const rows = [
  {name: 'Vector search only', ndcg: '0.807', ms: '30 ms'},
  {name: 'Hybrid: vector + keyword', ndcg: '0.846', ms: '30 ms'},
  {name: '+ extra search, every question', ndcg: '0.825', ms: '1,320 ms'},
  {name: '+ extra search, when needed', ndcg: '0.855', ms: '592 ms', best: true},
];
const tx = 0.5, tw = 4.75, colN = tx + 2.85, colL = tx + 3.75, rowH = 0.48, ty = 1.95;
text(s2, 'Ranking quality', {x: colN - 0.35, y: 1.62, w: 0.95, h: 0.25, fontSize: 10, bold: true, color: MUTED, align: 'right'});
text(s2, 'Latency', {x: colL, y: 1.62, w: 0.9, h: 0.25, fontSize: 10, bold: true, color: MUTED, align: 'right'});
rows.forEach((row, i) => {
  const y = ty + i * rowH;
  if (row.best) {
    s2.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: tx - 0.1, y: y - 0.06, w: tw + 0.1, h: rowH, rectRadius: 0.06, fill: {color: TEAL_TINT}, line: {color: TEAL_TINT}, objectName: 'best-row'});
  } else if (i > 0) {
    s2.addShape(pres.shapes.LINE, {x: tx, y: y - 0.06, w: tw - 0.1, h: 0, line: {color: GRAY_TINT, width: 1}});
  }
  const style = {fontSize: 14, bold: !!row.best, color: row.best ? TEAL : INK};
  text(s2, row.name, {x: tx + 0.05, y: y + 0.04, w: 2.7, h: 0.3, ...style});
  text(s2, row.ndcg, {x: colN, y: y + 0.04, w: 0.6, h: 0.3, align: 'right', ...style});
  text(s2, row.ms, {x: colL, y: y + 0.04, w: 0.9, h: 0.3, align: 'right', ...style});
});

s2.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: 0.5, y: 4.0, w: 4.65, h: 1.0, rectRadius: 0.08, fill: {color: GRAY_TINT}, line: {color: GRAY_TINT}, objectName: 'takeaway'});
text(s2, [
  {text: 'Takeaway: ', options: {bold: true}},
  {text: 'targeting the extra search (46% of questions) cuts latency by more than half versus always searching, and ranks better.'},
], {x: 0.7, y: 4.15, w: 4.25, h: 0.75, fontSize: 13});

text(s2, 'END-TO-END PROTOTYPE', {x: 5.6, y: 1.62, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: TEAL, charSpacing: 1});
text(s2, [
  {text: 'A local Qwen3-4B writes cited answers or refuses. Each answer takes up to 6 s, and '},
  {text: '~96% of that is the LLM writing', options: {bold: true}},
  {text: ', so reusing verified answers saves the most.'},
], {x: 5.6, y: 1.95, w: 3.95, h: 1.0, fontSize: 13});

text(s2, 'BEFORE THE FINAL REPORT', {x: 5.6, y: 3.12, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: TEAL, charSpacing: 1});
const next = [
  'Confirm on validation and test questions',
  'Score answers and refusals on all questions',
  'Verified cache for repeat questions',
  'Grow to ~300 questions, incl. multi-doc',
];
next.forEach((item, i) => {
  const y = 3.45 + i * 0.4;
  text(s2, String(i + 1), {x: 5.6, y, w: 0.3, h: 0.35, fontSize: 14, bold: true, color: TEAL});
  text(s2, item, {x: 5.92, y, w: 3.65, h: 0.35, fontSize: 14});
});

text(s2, 'Retrieval: 62 dev questions, Qwen3-Embedding-0.6B, nDCG@5 (1 = perfect). Answering: 3 rehearsal questions. Not yet validated on held-out data.',
  {x: 0.5, y: 5.12, w: 9, h: 0.3, fontSize: 10, color: MUTED});

s2.addNotes(`Speaking target: 65 seconds, then about one minute of questions.

We tested this idea on our development set of 62 answerable questions. Ranking quality here is nDCG at 5, where 1 means the correct document is ranked first.

Plain vector search scores 0.807. Adding keyword search to make it hybrid raises that to 0.846, at the same 30 milliseconds. Next we tried an extra search step, where a small local LLM rewrites the query using the first results. Doing that on every question is slow, about 1.3 seconds, and it actually ranks slightly worse, 0.825.

So we made it adaptive: we run the extra search only when keyword search and vector search disagree on the top document. That happens on 46 percent of questions. The result is the best ranking, 0.855, at 592 milliseconds, less than half the latency of always searching. That is our main takeaway: spending effort only where the question needs it can be both cheaper and better.

We also have an end-to-end prototype: a local Qwen3 4B model writes an answer with citations, or refuses when the passages do not support one. In our rehearsal questions, about 96 percent of each answer's time was the model writing it, three and a half to six seconds, versus well under a second for retrieval. So the biggest remaining saving is reusing verified answers, which is our next step.

Before the final report we confirm these development results on validation and test questions, score answers and refusals on every question, add the verified cache, and grow to about 300 questions, including multi-document ones.

Answering source: John's feat-rag-metrics branch, demoplan.md (offline rehearsal on an Apple M4 Pro: qst_0026 answered with two valid citations in 6.21 s, 6.05 s generation; the Gmail password question and qst_0046 refused in 3.96 s and 3.64 s).
Retrieval source: Ty's optimization experiments, branch feat-retrieval-optimizations, experiments/optimizations (figure 4, combinations-dev-20261020-01: 8 passes, 3,312 requests).

Q&A:
What is nDCG@5? A score for the top-5 ranking: 1 if the correct document is first, lower the further down it appears, 0 if it is missing.
Recall@5? Hybrid 0.919; extra search always 0.935; only when needed 0.935. The gain is mostly in ranking order, not in finding new documents.
Why not a reranker? We tried the ms-marco MiniLM cross-encoder. It lowered nDCG@5 from 0.807 to 0.781, likely because it was trained on web search, not enterprise documents.
What is the extra search? LameR: an LLM (Llama 3.2 3B, run locally) drafts likely answers from the top keyword results, and we search again with them.
Is the gain significant? Paired bootstrap intervals were computed; the run is small (62 questions) and on the development split only, so we present it as an observation.
Why is this different from the earlier MiniLM numbers? These runs use Qwen3-Embedding-0.6B and a separate experiment harness, so the numbers are not directly comparable.
Does refusal work? A first check: the off-topic Gmail question was refused. qst_0046 was also refused because its retrieved passages missed the 5-business-day answer, a retrieval limit rather than a model error. Citations and quotes are checked by code; semantic correctness is not yet verified at scale.
Is this cost or only latency? Latency only, on an M4 Max laptop with warm models. Hardware and energy cost are not measured; there are no paid API calls.`);

const out = path.join(__dirname, 'Team13-Midterm-v2.pptx');
pres.writeFile({fileName: out}).then(() => console.log(`Wrote ${out}`));
