# Revised dissertation outline — MED-US

**Chapter titles are fixed as required.** Subsections are updated to the final
class-level project. Word budget assumes ~18,000 words for the body — confirm against
your handbook and rescale proportionally.

Legend: **[E]** = evidence already exists in the repository · **[W]** = writing and
argument only · **[H]** = needs a human decision before writing (see the context pack,
§14).

---

## Front matter (unnumbered)

Title page · Abstract (~300 words) · Acknowledgements (supervisor; Aditya and Pragati
for the distributed reference training **[H]**) · Contents · List of Figures · List of
Tables · Abbreviations and Notation — define `D_f`, `D_r`, `W_0`, `W_ref`, `C*`, `S`,
`ACC_r`, `ACC_f` exactly once, here.

> **Abstract framing.** Lead with the structure precondition and the honest headline,
> not the accuracy: a gradient-free multi-objective search that holds retention within
> 0.35 of the state of the art, forgets substantially less well, and identifies *when*
> weight-editing unlearning can work at all.

---

## 1. Introduction and Problem Statement — 1,200 words

| § | Title | Notes |
|---|---|---|
| 1.1 | The right to erasure and the cost of exact retraining | GDPR Art. 17 as the obligation; retraining a ResNet-18 as the naive price. **[W]** |
| 1.2 | Approximate unlearning by post-hoc weight editing | The deployment case: one pass over the weights, no optimiser state, no retain-set training loop. **[W]** |
| 1.3 | From an instance-level failure to a class-level hypothesis | The predecessor's 10,534 strategies, best `S` = 1.158 against ~932 for retraining; the activation measurement (0.55% vs 1.00% chance) that explained it; the falsifiable prediction that follows. **State the prediction here, before testing it.** **[E]** |
| 1.4 | Problem statement | Gradient-free single-class unlearning, formulated as a multi-objective search over layer-wise weight-editing strategies. Name the three axes — match the reference on `D_f`, preserve `D_r`, edit as little as possible — without yet giving formulae. **[W]** |
| 1.5 | Research questions | RQ1 structure as precondition · RQ2 competitiveness against the gradient-free SOTA · RQ3 what predicts per-class success · RQ4 what one constrained gradient step adds, and what it costs. **[W]** |
| 1.6 | Contributions | Four claims, each naming the chapter that discharges it. Include the negative and partial ones. **[W]** |
| 1.7 | Structure of this dissertation | **[W]** |

---

## 2. Background and Foundations (General SOTA) — 2,000 words

Foundations a reader needs before the literature can be criticised. Keep it a
*foundation* chapter, not a second literature review — the critique lives in Chapter 3.

| § | Title | Notes |
|---|---|---|
| 2.1 | **Machine unlearning: definitions and setting** | |
| 2.1.1 | Exact unlearning and partition-based training | SISA and why it is a training-time commitment. **[W]** |
| 2.1.2 | Approximate unlearning: what is guaranteed and what is only measured | **[W]** |
| 2.1.3 | Class-level versus instance-level forget sets | Not a cosmetic distinction — it is the axis this whole project turns on. **[W]** |
| 2.1.4 | Retraining as gold standard: the reference model `W_ref` | And the framing that follows: **the target is the gap to `W_ref`, not zero accuracy.** A model at 0% has learned to avoid the answer, which is its own detectable signature. **[E]** |
| 2.2 | **Deep networks and where class information lives** | |
| 2.2.1 | ResNet-18 and the CIFAR variant | 3×3 stride-1 stem, no max-pool, and why the torchvision stem destroys 32×32 inputs. **[E]** |
| 2.2.2 | Layer groups as the addressable unit | Six groups; `layer4` alone holds 75.12% of 11,173,962 parameters. This number reappears as the justification for a *relative* edit cost. **[E]** |
| 2.2.3 | BatchNorm: state, not weights | Running statistics are buffers. Foreshadows §4.8.2. **[E]** |
| 2.3 | **Gradient-free model-editing primitives** | Masking, pruning, damping, clipping, quantisation, noise, reset — as a family, with the activation-aware importance criterion (Wanda) that makes selection possible without a derivative. **[E]** |
| 2.4 | **Multi-objective optimisation foundations** | |
| 2.4.1 | Pareto dominance and non-dominated fronts | **[W]** |
| 2.4.2 | NSGA-II: non-dominated sorting, crowding distance, elitism | **[W]** |
| 2.4.3 | Integer-coded genomes, and why SBX and polynomial mutation do not apply | Sets up the from-scratch implementation in §4.6. **[E]** |
| 2.5 | **Evaluating unlearning** | |
| 2.5.1 | Accuracy-based: `ACC_r`, `ACC_f`, and the composite `ACC_r × (1 − ACC_f)` | **Corrected definition — MIA is not a term in the composite.** **[E]** |
| 2.5.2 | Membership inference as an evaluation metric | Introduce it here as a *metric*, never as an objective. **[E]** |
| 2.5.3 | Selectivity `S`, and the instance-level ceiling of 1.158 | **[E]** |

---

## 3. Literature Review and Critical Analysis — 3,200 words

**This chapter must synthesise and criticise, not summarise.** Two cross-cutting
critiques (§3.2, §3.3) are what lift it out of the "list of papers" band; write those
before §3.1 if it helps.

| § | Title | Notes |
|---|---|---|
| 3.1 | **Families of unlearning methods, critically compared** | One comparison table, then argument per family. |
| 3.1.1 | Retraining and partition-based approaches | **[W]** |
| 3.1.2 | Gradient fine-tuning: NegGrad, NegGrad+, SalUn, Boundary Unlearning | **[W]** |
| 3.1.3 | Distillation: SCRUB, Bad Teaching, and ZRF | **[W]** |
| 3.1.4 | Impair–repair: UNSIR | Nearest published neighbour on `ACC_f` (10.89 ± 8.79) and, like this work, has wide per-class spread. **[E]** |
| 3.1.5 | Gradient-free weight editing: SSD, Kodge et al. | **The family MED-US belongs to.** **[E]** |
| 3.2 | **Cross-cutting critique I — protocol fragmentation** | Datasets, forget rates, MIA definitions, and *which classes each paper actually reports* all differ. This is why a common harness is needed, and it earns §5.2 in advance. Source table already exists in `claudedocs/research_anchor_paper_20260827.md`. **[E]** |
| 3.3 | **Cross-cutting critique II — what a membership-inference number certifies** | The SVC-on-ground-truth-confidence attack, LiRA/U-LiRA, and the saturation argument: Retraining pinned at 100.00, SCRUB at `ACC_f` 0.00 with MIA 0.00, and this project's own AUC near chance on the same models. **A genuine methodological contribution that costs only writing.** **[E]** |
| 3.4 | **Search-based and evolutionary model editing** | Evolutionary methods in pruning, compression and architecture search; the gap where unlearning should be. **[W]** |
| 3.5 | **The anchor study, and what adopting it costs** | Kodge et al. (TMLR 2024): method, Table 1, and the two protocol commitments adopting it forces — report their metric definitions, and sweep all ten classes because their numbers are ten-class means. **[E]** |
| 3.6 | **Gap analysis and positioning of MED-US** | No prior work treats operator choice as a multi-objective search, and none reports a *precondition* for weight-editing unlearning. That pair is the gap. **[W]** |

---

## 4. Multi-Objective Evolutionary Design of Unlearning Strategies (MED-US) — 3,400 words

**Every design choice justified against an alternative that was rejected.** This is
where a marker looks for independent decision-making.

| § | Title | Notes |
|---|---|---|
| 4.1 | **Design rationale and the alternatives rejected** | Three separate decisions — gradient-free, post-hoc, multi-objective — argued on their own terms rather than as one package. **[W]** |
| 4.2 | **The precondition: forget-specific structure** | The per-channel activation-contrast instrument and its null control (two disjoint halves of `D_r`). **Belongs in methodology, not results** — it is an instrument built here, and it is applied again in §5.10. **[E]** |
| 4.3 | **The search space** | |
| 4.3.1 | Layer groups: `stem, layer1…layer4, fc` → `L = 6` | Disjointness and coverage invariants. **[E]** |
| 4.3.2 | Chromosome encoding `x = (b, g, s, d_g, d_s)` | Five integer vectors of length 6; 30 genes; gene-major flat layout. **[E]** |
| 4.3.3 | The decoding rule, and latent genes | Frozen groups keep their genes as latent material a later mutation can recover; the cost is that distinct genomes decode to the same strategy, which the canonical cache form handles. **[E]** |
| 4.3.4 | Size of the search space | 3.87 × 10¹⁴ genomes at `max_level = 2`. **This is the justification for search over enumeration** and the number a random-search ablation would be measured against. **[E]** |
| 4.4 | **The operator library** | |
| 4.4.1 | Channel A — editor operators: `MASK`, `PRUNE`, `RANDOM_PRUNE` | **[E]** |
| 4.4.2 | Channel B — smoother operators: `DAMP`, `NOISE`, `CLIP`, `QUANTIZE`, `RESET` | **[E]** |
| 4.4.3 | Ordinal intensity ladders, and why `max_level = 2` | **[E]** |
| 4.4.4 | Operators excluded at library level: `REINIT`, `SIGN_FLIP` | Excluded from the *library*, not disabled by config, so no config edit can restore them. Independent-judgement evidence. **[E]** |
| 4.4.5 | Connection selection: `class_contrast` vs `magnitude` vs `random` | The Wanda-style criterion `\|W_ij\| · (rms_f − rms_r)` from forward passes only; `PRUNE` and `RANDOM_PRUNE` as deliberate data-free controls. **[E]** |
| 4.5 | **The three objectives** | **Replaces the outdated "maximise unlearning / maximise utility / minimise leakage" formulation entirely.** |
| 4.5.1 | `f1 = JS(P_ref(D_f) ‖ P_cand(D_f))` — minimised, bounded by ln 2 | Why JS over KL: symmetric, bounded, finite everywhere. **[E]** |
| 4.5.2 | `f2 = L_r` — retain cross-entropy, minimised | **[E]** |
| 4.5.3 | `f3 = ‖θ − θ₀‖₂ / ‖θ₀‖₂` — relative edit cost, minimised | Relative because `layer4` holds 75.12% of the parameters; weights only, because BN buffers are not edits. **[E]** |
| 4.5.4 | Why `ACC_f` is not optimised directly, and why MIA is not an objective | **[W]** |
| 4.5.5 | Why `f3` is an edit cost and not a second reference term | The measured Spearman against `f2`: +0.36 for the parameter norm vs +0.74 for the old KL. A three-objective search that was really two. **[E]** |
| 4.6 | **NSGA-II for an integer-coded genome** | Implemented from scratch (Deb et al. 2002) — **not pymoo, which is not a dependency**. Uniform crossover p = 0.9, random-reset mutation p = 0.10, binary tournament, elitist (μ + λ). Population 10, 50 generations, seed 42, 510 evaluations per class. Two details worth citing: the zero-range guard in crowding distance, and that min-max normalisation cannot change the fronts. **[E]** |
| 4.7 | **Two-tier fidelity and the selection of `C*`** | Subset evaluation during search (64 / 256, batch cap 3), full-fidelity re-measurement of the front afterwards. `C*` maximises the anchor composite — one rule for all ten. **Disclose that selection reads the same test quantities that are later reported.** **[E] [H]** |
| 4.8 | **The hybrid variant — a separate method** | |
| 4.8.1 | One clipped ascent step on `D_f`, one repair step on `D_r` | **[E]** |
| 4.8.2 | Why BatchNorm must be frozen | The silent failure: an unfrozen attempt passed every weight-based guard while eight batches of `D_r` re-estimated the running statistics and undid the operator edit. Buffer movement is the only check that catches it. **Strong methodology-chapter material.** **[E]** |
| 4.8.3 | Six acceptance checks, and why the hybrid is reported separately, permanently | **[E]** |
| 4.9 | **Implementation and reproducibility** | Self-contained per-class configs; sha256 provenance; byte-for-byte verification when merging a measured row into a committed table; resumable search including RNG state; 65 passing tests. Short, but it is what makes every later number credible. **[E]** |

---

## 5. Experimental Study — 4,400 words

**The largest chapter.** Report in §5.5–5.11, interpret in §5.13.

| § | Title | Notes |
|---|---|---|
| 5.1 | **Experimental setup** | CIFAR-10, ResNet-18 CIFAR variant, splits 5,000 / 45,000 / 1,000 / 9,000, hardware (GTX 1650 local; Tesla T4 for reference training), software versions. **[E]** |
| 5.2 | **Reference and baseline models** | `W_0` (test acc 0.9479) and the ten `W_ref`. Training protocol, selection rule, validation, checksums, distributed training across three people and seven Kaggle bundles. **State plainly that no published baseline was re-implemented.** **[E] [H]** |
| 5.3 | **Metrics as implemented** | `ACC_r`, `ACC_f`, composite, anchor MIA, own MIA AUC, selectivity `S`. Both MIA definitions side by side, with the reason two exist. **[E]** |
| 5.4 | **Protocol and controls** | One seed and what a seed does and does not vary; the `C*` selection rule; the `S = nan` defect and its guard. **[E]** |
| 5.5 | **Result 1 — class structure exists (RQ1)** | 84.1–91.2% vs 0.55% instance-level against a 1.00% null. **And the follow-on null: structure does not predict difficulty, r = −0.04.** Report both here; interpret in §5.13. **[E]** |
| 5.6 | **Result 2 — pure MED-US across ten classes (RQ2)** | Pareto fronts and their geometry; per-class `C*`; the aggregate 93.84 / 12.55 / 82.09 / 92.54; selectivity against the instance-level ceiling; zero failures. **[E]** |
| 5.7 | **Result 3 — comparison against the anchor benchmark** | The full table with the reported/measured column. Retention competitive, forgetting not. The harness-agreement evidence. **[E]** |
| 5.8 | **Result 4 — the hybrid variant (RQ4)** | Nine of nine accepted; BN buffer movement exactly 0.000000; −5.00 `ACC_f` for −0.12 `ACC_r`; nine of ten improve on the composite. **[E]** |
| 5.9 | **Result 5 — operator analysis** | `MASK` in all ten; full frequency; the claim's limit without an ablation. **[E]** |
| 5.10 | **Result 6 — the truck case study** | Its own subsection. Worst on every metric; largest refinement gain; the prediction distribution (68.40% → automobile under `W_ref`); the mutual similarity with automobile; and the two nulls that rule out the easy explanations. **[E]** |
| 5.11 | **Computational cost** | ~8 min search + ~10 min full fidelity per class; 510 evaluations; 37–58% cache hits; no retain-set training loop, no optimiser state. Report the *shape* of the cost, not a wall-clock race. **[E]** |
| 5.12 | **Threats to validity** | Internal: selection-on-test, a single `W_0`. External: one dataset, one architecture. Construct: what MIA certifies. Conclusion: ten classes, one seed. **[E]** |
| 5.13 | **Critical discussion** | Where the marks are. What the results mean; what MED-US buys (no gradients, no retain-set loop, an explicit Pareto front rather than a single operating point) and what it costs; the honest positioning against the anchor; and the reflection on what worked and what failed. **[W]** |

---

## 6. Conclusion — 1,100 words

| § | Title | Notes |
|---|---|---|
| 6.1 | Answers to the research questions | RQ1–RQ4 in the order asked, each with its verdict — including the partial answer to RQ2 and the null for RQ3. **[W]** |
| 6.2 | Contributions restated against the evidence | **[W]** |
| 6.3 | Limitations | Distinct from §5.12 — what the *method* cannot do, rather than what the *study* might have got wrong. **[E]** |
| 6.4 | Future work | Random search at equal budget; baselines in this harness; extra search seeds; a stronger attack in place of the saturated MIA; the untried per-class predictor (overlap between edited channels and a retained neighbour's channels); second architecture and dataset. **[E]** |
| 6.5 | Closing reflection | What was learned about *when* weight-editing unlearning can work, and about reporting a negative result honestly. **[W] [H]** |

---

## 7. Declaration on the Use of AI

Not written yet. Bullet notes in `ai_declaration_notes.md`. Must be explicit,
specific and honest.

---

## 8. References

Every citation traceable to a paper actually read. The anchor and seven comparison
papers are recorded with links in `claudedocs/research_anchor_paper_20260827.md`.
**Do not cite pymoo — it is not used.** Cite Deb et al. (2002) for NSGA-II, and the
Wanda paper for the activation-aware importance criterion.

---

## Appendices

| | Content |
|---|---|
| A | Per-class full results — ten anchor-metric tables, pure and hybrid |
| B | The ten selected `C*` chromosomes (30 integers each) and one full per-class YAML |
| C | Reference training: logs, validation verdicts, sha256 table |
| D | Reproducibility — the artefact manifest; what is and is not in the repository |
| E | The remaining seven Pareto-front figures at full size |
| F | Ethics and data statement **[H]** |

---

## Word budget

| chapter | words | share |
|---|---:|---:|
| 1. Introduction and Problem Statement | 1,200 | 6.7% |
| 2. Background and Foundations | 2,000 | 11.1% |
| 3. Literature Review and Critical Analysis | 3,200 | 17.8% |
| 4. MED-US | 3,400 | 18.9% |
| 5. Experimental Study | 4,400 | 24.4% |
| 6. Conclusion | 1,100 | 6.1% |
| Discussion absorbed into 5.13 and 6 | — | — |
| Slack / front matter / declaration | 2,700 | 15.0% |
| **Total** | **18,000** | |

Deliberately lopsided towards Chapters 3, 4 and 5 — 61% between them — because the
marking criteria weight critical synthesis, methodological justification and systematic
evaluation, and because the results are mixed enough that the *reading* of them is the
contribution.
