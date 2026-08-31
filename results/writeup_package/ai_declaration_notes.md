# Notes for the AI-use declaration

**This is not the declaration.** These are working notes for Chapter 7. Write the final
text yourself, in your own voice, once the dissertation is finished and you know the
full extent of the use.

**Two principles.** Be specific — a vague declaration reads as evasive, and a specific
one reads as confident. And be accurate — do not overstate the assistance to seem
scrupulous, or understate it to seem independent. Under-declaring is an academic
integrity risk; over-declaring gives away work that is genuinely yours.

---

## 1. Where AI was used

### Coding assistance
- Implementation support across the experimental codebase: the operator library, the
  chromosome encoding and decoder, the NSGA-II implementation, the evaluation harness,
  the class-split utilities, and the analysis and plotting scripts.
- Drafting of experiment-driver scripts (`run_class_sweep.py`, `report_anchor_metrics.py`,
  `refine_candidate.py`, `build_*` summary and figure builders).
- Assistance with the Kaggle packaging and reference-import tooling used to distribute
  reference-model training.
- **State the working pattern honestly:** the architecture, the experimental design and
  the acceptance criteria were specified by me; AI assisted with implementation within
  those specifications.

### Debugging support
- Diagnosing the BatchNorm failure in the refinement step, where an earlier attempt
  passed every weight-based guard while `D_r` batches silently re-estimated the running
  statistics and undid the operator edit. **The decision to add an explicit
  buffer-movement check as a sixth acceptance criterion was mine.**
- Diagnosing the `selectivity_S = nan` defect, where a plain `max()` returned the first
  `nan` row and reported the unedited model as most selective. The finite-value guard and
  the decision to re-run the three affected diagnostic rows were mine.
- Routine debugging: environment and CUDA build issues, dataloader determinism, encoding
  and path handling on Windows.

### Drafting and editing assistance
- Structuring the dissertation outline and the chapter/section plan.
- Drafting and editing prose; improving clarity, flow and consistency of terminology.
- Producing the planning documents in `results/writeup_package/` — the context pack,
  table/figure plan, revised outline and writing order.
- Generating captions and summary tables from the committed result files.
- **Be precise here about the final chapters:** state whether AI assisted with drafting
  the dissertation text itself, and to what extent. Do not leave this ambiguous. If AI
  drafted passages that you then revised, say so.

### Analysis and figure production
- Writing the plotting scripts that produced the seven write-up figures and the ten
  Pareto-front figures.
- Computing derived quantities from committed artefacts: the inter-class similarity
  matrix, operator frequencies, and the correlation coefficients reported in the
  class-structure analysis.
- **All underlying measurements come from experiments I designed and ran.**

---

## 2. What remains mine

Say these plainly. They are the substance of the contribution.

- **The research direction.** The decision to abandon instance-level unlearning after the
  exhaustive search failed, to measure *why* it failed, and to redirect the project to
  class-level forgetting on the strength of a falsifiable prediction.
- **The experimental design.** The three objectives and their formulation; the choice of
  a gradient-free, post-hoc, multi-objective approach; the operator library and the
  decision to exclude `REINIT` and `SIGN_FLIP` at library level rather than by
  configuration; the retention of `PRUNE` and `RANDOM_PRUNE` as data-free controls.
- **The choice of anchor study** and the decision to adopt its measurement protocol,
  including the consequence that all ten classes had to be swept.
- **The decision to keep pure and hybrid permanently separate**, and to refuse to report
  the hybrid's stronger numbers as the project's headline result.
- **The reference-model selection rule** — `D_r_test` accuracy with loss as tie-breaker,
  with `D_f_test` deliberately excluded from selection — and the reasoning that full-test
  selection is both diluted and backwards.
- **All interpretation of results**, including the decision to report the null correlation
  between class structure and per-class difficulty as a finding rather than omitting it.
- **All verification.** Every number in the dissertation was checked against the committed
  CSV and Markdown artefacts.
- **Final judgement on every claim made, and full responsibility for the work.**

---

## 3. Numbers and verification — state this explicitly

- Every quantitative result originates from experiments run in this project, not from any
  AI system.
- Results are stored as committed artefacts under `results/`, principally
  `results/writeup_package/` and `results/literature_alignment/`.
- Headline numbers were re-read from their source CSVs and cross-checked against the
  rendered Markdown summaries.
- The summary builder recomputes the pure-method aggregate and **exits non-zero if it
  disagrees with the committed table**, so a drifted number cannot be published silently.
- When measured rows were merged into existing tables, preserved rows were copied as raw
  CSV text and compared **byte for byte** after writing, restoring a backup on any
  mismatch.
- The eight literature rows in the benchmark comparison are transcribed from the anchor
  paper's Table 1, are labelled as reported rather than measured in every rendering, and
  were **not** produced by any AI system.
- 65 automated tests cover the objectives, operators, class splits and anchor metrics; all
  pass.

---

## 4. Points to decide before writing the final text

| # | Decision |
|---|---|
| D1 | **Check your school's exact wording requirements.** Some require a named tool, version and date range; some require a per-chapter breakdown; some require a specific declaration form. This note is content, not format. |
| D2 | **Decide the granularity for dissertation prose.** Chapter-by-chapter, or a single overall statement? Chapter-by-chapter is more defensible if the level of assistance varied. |
| D3 | **Name the tool(s) and period of use.** Include any other AI assistance used earlier in the project, not only recent work. |
| D4 | **Decide how to describe the human–AI working pattern.** "I specified, AI implemented, I verified" is accurate for the code — check it is accurate for the writing too, and say something different if it is not. |
| D5 | **Confirm the reference-training attribution separately.** Reference models for eight of the ten classes were trained by two collaborators on Kaggle. That is human collaboration, not AI assistance — it belongs in the acknowledgements and in §5.2, and should not be blurred into this declaration. |
| D6 | **Ask your supervisor** whether the department expects the declaration to cover code as well as text. Practice varies. |

---

## 5. Keep a running log from now on

Reconstructing AI use at the end is harder and less accurate than recording it as you go.
A short running note per session — date, what was worked on, what assistance was used,
what you decided yourself — makes the final declaration a summarising exercise rather
than a memory exercise, and gives you something concrete if you are ever asked to
elaborate.

---

## 6. Tone

Neither defensive nor apologetic. AI use is permitted; what is assessed is your
understanding, your judgement and your ownership of the work. A declaration that
specifies exactly what was assisted and exactly what was decided demonstrates all three
better than a minimal one.
