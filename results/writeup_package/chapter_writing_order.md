# Chapter writing order

**Recommended order — evidence first, argument outward, framing last.**

| order | chapter | why now | words | rough time |
|---:|---|---|---:|---|
| 1st | **5. Experimental Study** | Closest to work already done; every table and 10 of 11 figures exist | 4,400 | 4–5 days |
| 2nd | **4. MED-US** | You have just written what the method *did*; now write what it *is* | 3,400 | 3–4 days |
| 3rd | **3. Literature Review and Critical Analysis** | You now know exactly which critiques your results support | 3,200 | 4–5 days |
| 4th | **2. Background and Foundations** | Include only what Chapters 3–5 actually rely on | 2,000 | 2 days |
| 5th | **1. Introduction and Problem Statement** | Promise only what the dissertation delivers | 1,200 | 1–2 days |
| 6th | **6. Conclusion** | Answer questions whose answers are now on paper | 1,100 | 1 day |
| 7th | **7. Declaration on the Use of AI** | Needs the full record of how AI was actually used | — | ½ day |
| 8th | **8. References** | Finalise once every citation has been used in anger | — | ½ day |
| last | **Abstract** | Written last, always — it summarises a finished document | ~300 | ½ day |

Front matter (contents, figure and table lists) is generated after everything else.

---

## Why this beats writing from Chapter 1 forward

**1. Chapter 1 makes promises the rest of the dissertation must keep.** Written first, it
is written from intentions, and every later chapter that diverges leaves an inconsistency
you must hunt down. Written fifth, the contributions list in §1.6 is a *description* of
what Chapters 4 and 5 contain — including the honest ones. You cannot accurately promise
"we identify what predicts per-class difficulty" before discovering that the answer is a
null (r = −0.04).

**2. The results are mixed, and the mixture determines the framing.** MED-US is 12.52
points behind the anchor on `ACC_f`. That single fact decides how Chapter 1 must be
written: the spine cannot be "we beat the state of the art", it has to be "when does
weight-editing unlearning work at all". Writing the introduction before internalising
that risks a framing the evidence will not carry — and a mismatched introduction is one
of the most visible weaknesses a marker can find.

**3. Chapter 5 is where the marks concentrate, so it gets your best energy.** It is the
largest chapter (24% of the body), it carries systematic evaluation and critical
analysis, and it has the most existing material. Writing it first means it is drafted
while you are fresh and gets the most revision passes. The common failure mode is the
opposite: three polished early chapters and a rushed results chapter.

**4. Methodology is easier to justify once you have described its consequences.** §4.5.5
argues that `f3` had to be an edit cost rather than a second reference term. That argument
is far easier to write after Chapter 5 has shown the fronts the objectives actually
produced. The same holds for §4.4.4 (excluded operators), §4.7 (two-tier fidelity) and
§4.8.2 (BatchNorm freezing) — each is a design decision whose *justification* is clearest
once its *effect* is on paper.

**5. The literature review should be shaped by your findings, not the reverse.** This is
the difference between the good and excellent bands. Two of your strongest critiques —
protocol fragmentation (§3.2) and MIA saturation (§3.3) — are things you can only argue
with conviction *after* re-implementing the anchor's protocol and seeing its MIA read
92–100 while your own AUC sits near chance. Written third, §3.3 becomes a critique
grounded in your own measurements. Written first, it would be a summary of what other
people have said.

**6. Background stays proportionate.** Written fourth, Chapter 2 contains exactly what
Chapters 3–5 depend on and nothing else. Written first, it becomes a textbook chapter —
the classic way to spend 3,000 words on material no later chapter uses, while the results
chapter runs short.

**7. It front-loads the risk.** If something is wrong — a number that will not reconcile,
a figure that misleads, a claim without support — you find it in week one, while there is
time to fix or disclose it. The `C*` selection-on-test issue is exactly this kind of
problem: cheap to address early, awkward to discover the week before submission.

---

## The one exception

**Write §1.5 (research questions) as a rough draft before starting Chapter 5**, on a
single page, and keep it beside you. Not the chapter — just RQ1–RQ4. Chapter 5's sections
map onto them (§5.5 → RQ1, §5.6–5.7 → RQ2, §5.5/§5.10 → RQ3, §5.8 → RQ4), and having them
written down keeps the results chapter organised around questions rather than around the
order the experiments happened to run in.

Expect to revise them when you write Chapter 1 properly. RQ3 in particular will change:
its honest form is "does forget-specific structure predict per-class difficulty?" with
the answer "no", not "what predicts it?" with an answer you do not have.

---

## Within Chapter 5, a suggested internal order

Not strictly top to bottom. Write the results sections while the evidence is in front of
you, then the framing sections around them:

1. **§5.6** (pure results) — the central table, the spine of the chapter
2. **§5.7** (benchmark) — while the comparison table is open
3. **§5.8** (hybrid) — the three dumbbell figures make this fast
4. **§5.5** (class structure) — needs the RQ1/RQ3 framing to be settled first
5. **§5.10** (truck) — the richest section; write it when you have momentum
6. **§5.9** (operators), **§5.11** (cost) — short, mechanical
7. **§5.1–5.4** (setup, references, metrics, protocol) — easiest to write last, once you
   know precisely which details the results sections actually referred to
8. **§5.12** (threats), **§5.13** (critical discussion) — last, and give §5.13 real time.
   It is where the reflection marks are.

---

## Practical notes

- **Draft in Markdown, convert at the end.** Do not fight LaTeX or Word formatting while
  the argument is still moving. Generate the final format once the content is stable.
- **Copy every number from the CSV, never from memory or from a summary file.** Sources
  are listed in `dissertation_writing_context_pack.md` §4.
- **Keep a running list of claims made**, and check each against the key claims register
  (context pack §7) before submission. A claim not in that register is not yet supported.
- **Log AI use as you go**, in the running notes described in `ai_declaration_notes.md`.
  Reconstructing it at the end is both harder and less accurate.
- **Re-verify every number at final draft.** The context pack was verified on 2026-08-31;
  verify again before submission.
