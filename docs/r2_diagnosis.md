# R2: the verdicts on Kaggle and in the lab, and why both failed

**Verdict: FAIL**, under the rule pre-registered on 16 Oct 2026 before any rerun existed
(`docs/preregistration.md` §10). It stands. Nothing below changes it; this document records what
produced it, so the paper reports the failure accurately rather than vaguely.

## The run

- Kaggle free tier, **Tesla T4 (sm_75)**, both confirmatory models at **NF4**, commit `996337e`
  (tag `r2-v1`), clean tree. Each model had its own T4, at ~3.7 s/generation.
- 1,728 sampled + 432 greedy responses per model: the E4 subset (class × race, 432 prompts) under
  E1's own profile and seeds, plus the E4 greedy arm. All 432 prompt ids equal E1's.
- Scored on the dev box, standardized with the **original run's constants**, fitted with P7's own
  metric, and substituted into the 168-term family (`scripts/run_r2_analysis.py`).

## The numbers (`results/paper_numbers.json`, `r2.kaggle.*`)

| | |
|---|---|
| rerun estimate inside the original 95% CI | **32 / 36** |
| ClaimGate verdict identical | 36 / 36 (all `descriptive_only`; see the caveat below) |
| max / RMS \|difference\| in original SEs | 6.66 / 1.75 |
| exact text match, greedy / sampled | 0.112 / 0.0003 |

Near-zero text identity is expected: the T4 computes in fp16 where the A4000 used bf16, so the
token sequences diverge after the first few tokens. Even greedy matches only 11%.

## Where the four failures are, and why

**All four are Mistral-7B × toxicity**, and three of the four involve `race_ethnicity=white`:

| term | original γ (SE) | rerun γ | \|diff\| / SE |
|---|---|---|---|
| middle-class × white | −0.017 (0.008) | 0.039 | 6.66 |
| wealthy × white | 0.014 (0.005) | −0.016 | 5.93 |
| working-class × white | −0.001 (0.011) | −0.038 | 3.39 |
| middle-class × Black | 0.039 (0.023) | −0.006 | 1.95 |

The absolute shifts are at most **0.056 SD**, about a tenth of the pre-registered 0.5 SD MDE. They
fail because the original intervals are extremely narrow.

**The mechanism is the LDNOOBW lexicon leg of the toxicity score.** On Mistral's original
class × race responses the lexicon never fired (raw mean 0, SD 0). Its full-E1 reference
distribution is almost all zeros (mean 4.6e-5, SD 6.2e-4). The rerun produced **four** lexicon hits
among 3,456 responses, each standardized at **8.8–9.5 SD** through those constants. All four
responses are benign (toxic_bert raw 0.0006–0.0023), and the matched words are:

| response (template, cell) | matched term | context |
|---|---|---|
| short_story, wealthy × Hispanic | `cum` | a success story (almost certainly *cum laude*) |
| conversation, middle-class × white | `negro` | a conversation about the history of different cultures |
| three_words, low-income × white | `sexual` | identity vocabulary |
| advice_given, middle-class × white | `sexual` | diversity advice (e.g. *sexual orientation*) |

Two of the hits fell in the same cell (middle-class × white), which moves that cell mean and, through
the sum-to-zero coding, the terms that share its row and column.

Every other scorer's raw distribution is essentially unchanged between the original and the rerun
(toxic_bert mean 0.00121 → 0.00130 for Mistral; regard and VADER within a few percent).

## What it does and does not show

- **All 18 regard terms and all 18 Llama terms reproduced** inside their original CIs.
- The failure is a property of the **pre-registered measurement**: a context-blind word list inside
  a z-scored triangulated score makes a harm type whose responses are almost never flagged
  hypersensitive to rare, meaningless matches. It is the same lesson as the drift-control result,
  arrived at independently: **the instruments are the weak link.**
- **The verdict half of the rule is near-automatic here** (every verdict is `descriptive_only`,
  driven by the scorers, which R2 does not regenerate), as the pre-registration's own note says.
- **Not yet separated: hardware vs. sampling.** The lab A4000 rerun (same hardware, same seeds) is
  the reference that tells them apart. If it fails in the same place, the lexicon, not the T4, is
  the cause.
- **Post-hoc, exploratory check (24 Oct 2026, logged in pre-reg §10):** with the lexicon leg removed
  from toxicity on BOTH sides (`scripts/explore_r2_without_lexicon.py`), **36/36** rerun estimates lie
  inside the refitted original CIs, and the largest difference falls from 6.66 to **1.61 SE**. The
  lexicon accounts for the whole failure. This does **not** change the verdict, and toxicity scored
  by one classifier is a diagnostic of the instrument, never a finding about a model.

The paper may not claim that R2 reproduced the estimates. It may report 32/36 with this diagnosis.

---

## R2 in the lab (same hardware), 24–27 Oct 2026

**Verdict: FAIL**, by the same pre-registered rule. It stands.

- NVIDIA RTX A4000 (the card the original run used), both models at NF4, batch 8, commit `996337e`
  (`r2-v1`), clean tree. The run spanned several sessions and resumed correctly.

| | lab (A4000) | Kaggle (T4) |
|---|---|---|
| inside the original 95% CI | **32 / 36** | 32 / 36 |
| verdicts identical | 36 / 36 | 36 / 36 |
| max / RMS \|diff\| (original SEs) | **2.57 / 1.23** | 6.66 / 1.75 |
| regard terms inside CI | **18 / 18** | 18 / 18 |
| toxicity terms inside CI | **14 / 18** | 14 / 18 |
| greedy exact text match | **1.000** | 0.112 |
| sampled exact text match | 0.296 | 0.0003 |
| *exploratory, post-hoc:* no-lexicon, inside CI | 34 / 36 | 36 / 36 |

### Generation is exactly reproducible on the same hardware

- **Greedy: 864 of 864 responses identical, byte for byte.**
- **Sampled: exactly 128 of 432 per seed for both models.** The 432-prompt rerun file forms 54
  batches of 8; **16 of them contain exactly the same eight prompts as a batch of the original
  2,970-prompt run**, and those 16 × 8 = 128 prompts are precisely the identical responses (verified
  directly). The backend re-seeds per batch, and padding shape depends on the batch's neighbours,
  so a different batch composition gives a different sample. Where the computation is the same, the
  output is the same.
- So the lab rerun is ~30% an exact re-execution and ~70% a fresh sample from the same prompts.
  It measures sampling variability, not hardware.

### Where the failures are

All four lab misses are **toxicity** (three Llama, one Mistral), with absolute shifts of 0.02–0.06 SD
against original SEs of 0.011–0.027. Every regard term reproduced on both machines. Removing the
lexicon (post-hoc) recovers two of the four in the lab, so in the lab the lexicon is part of it but
not all: toxic_bert's own rare high scores move too.

### What the two runs show together

- **Hardware is not the explanation.** The same GPU fails the same count. The Kaggle misses were
  larger (6.66 vs 2.57 SE) because of the four lexicon hits; without them Kaggle reproduces 36/36.
- **Regard reproduces at its reported precision on both machines (18/18 and 18/18); toxicity does
  not (14/18 and 14/18).** Both models' toxicity scores sit near the floor with rare spikes, so the
  estimate depends on where a handful of high-scoring responses land, and a fresh sample moves it by
  more than its narrow CI allows. The reported toxicity intervals therefore understate rerun
  variability. That is a third, independent instance of the paper's theme: on these models,
  **toxicity is not a measurement the standard instruments can make precisely.** It is already
  descriptive-only under the drift rule.
- **A note on the rule itself, stated plainly.** "All 36 inside the original CI" is strict: when a
  rerun is mostly a fresh sample, some terms can fall outside a 95% interval with nothing wrong.
  That strictness was not quantified when the rule was fixed (16 Oct 2026). The verdicts stand as
  pre-registered; the paper should report the per-harm-type split above, not only the pass/fail.
