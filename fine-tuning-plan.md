# Transferring DiffBrush to German handwriting — line-level synthetic data for form-field extraction (DRAFT full plan)

This repository extends the ICCV-2025 paper **DiffBrush** ([arXiv 2508.03256](https://arxiv.org/abs/2508.03256),
[repo](https://github.com/dailenson/DiffBrush)) — a diffusion model that generates *handwritten text lines* (one-shot:
one style reference + the requested text) — so that it can generate **German** handwriting. The motivation: German
form-field extraction needs training data, and this paper's Table 8 shows that recognizer training data built from
DiffBrush lines improves a down-stream recognizer by **+20.07 % relative CER** — the strongest evidence in the
literature that line-level *synthetic* handwriting is worth manufacturing. The upstream DiffBrush release ships
**inference only** (no training code), so this repo reconstructs the missing trainer from the paper text.

Status markers: ✅ = executed (evidence recorded under `reports/`), ⏳ = to be executed in this repo.

## Hardware & device policy (CUDA + MPS)

- One `pyproject.toml` / uv lock covers both backends: PyPI `torch` wheels ship the CUDA build on Linux and the MPS
  build on macOS — no per-machine dependency split.
- Single-process execution, no DDP/nccl (Phase-0 rewrite of `generate.py`; the trainer is single-device by
  construction).
- Host-machine-specific configuration lives in `.env` (git-ignored; template `.env.example`):
  `DEVICE` (`cuda|mps|cpu`), `PRETRAINED_MODEL`, `STABLE_DIF_PATH`, `SAVE_DIR`. Precedence:
  CLI flag > process environment > `.env` > built-in default. `generate.py` follows this
  already; the `german/` tools are brought in line when staged (see Staging notes).
- **Benchmarks are per-device facts, not promises.** Every stage re-benchmarks on the active device before scale-up (4a
  sizes 4b; the 4c `x0_mode` gate picks the arm).
- MPS measurements (this machine: M2 Max, 64 GB) form the reference ladder:
    - sampling ≈ 1.2 s/line (budget was ≤ 2 min/line — passed);
    - 4b trainer 2.3–9 s/step, ≤ 1.4 GB memory at batch 16;
    - 4c `chain5` ≈ 100 s/step vs `step1t` ≈ 7 s/step.
- CUDA expectation (to be confirmed by 4a/4c benchmarks): `chain5` — the paper recipe — becomes comfortably cheap, so
  the low-band `step1t` fallback should not be needed; batch sizes may be raised above the MPS ladder.

Phases in one glance:

| phase | purpose                                                                                          | key artifact                                                               |
|-------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| 0     | prove the released checkpoint + patched sampling run locally; record the "before" baseline       | single-process `generate.py`, `probe_texts.txt`, checkpoint anatomy report, probe images |
| 1     | German line corpus staged in DiffBrush's input contract                                          | `german/scads_lines/`, charset/quality reports                             |
| 2     | German charset end-to-end via glyph bitmaps (no embedding surgery)                               | `german/german_dataset.py` letters table + coverage audit                  |
| 3     | writer classes + warm-start recipe (proxy tables force-re-init 496→308)                          | `writers_{split}.json`                                                     |
| 4     | staged training: 4a sanity + benchmark → 4b German ε-MSE fine-tune → 4c D_line readability stage | `german/train_diffbrush_german.py`, `german/discriminators.py`, run ledger |
| 5     | generation + acceptance gate for recognizer-training-data usability                              | `german/check_ckpt_quality.py`, HTR gate protocol                          |

## Phase 0 — smoke test: can we run this at all, and does the checkpoint mean what we think?

### **What the phase does.** 
Before any German work, three questions that everything later depends on:

- (1) does the released checkpoint *strictly* load into the model as built from the upstream config (proving our reading
  of `EMB_DIM`/writer count is right, not guessed)?
- (2) does sampling run on this machine at tolerable speed? Upstream demands `torchrun` + 4×CUDA with `nccl` — rewritten
  here as a single-process sampler (`generate.py` Phase-0 rewrite) that also keeps the upstream two-loop FID/HWD
  protocol behind `--full_eval`.
- (3) what does the *untouched* IAM checkpoint produce for German probe text? That output becomes the **"before"
  baseline** every later quality gate compares against.

**Why it is needed.** The trainer was unavailable upstream, so every assumption about the checkpoint (class count,
embedding width, tensor nesting) had to be verified against the file itself before warm-starting; and the operating
discipline applies verbatim — stage everything device-agnostically (MPS here, CUDA as
contingency), benchmark honestly, escalate only on evidence.

**How validated.** `load_state_dict(strict)` with `missing=0 unexpected=0`; ≥3 legible lines within the time budget;
recorded probe images under `reports/phase0_probe/`.

### Required artifacts

**Re-run in this repo** (checkpoint anatomy + MPS smoke ✅, recorded under
`reports/phase0_report.md`; ⏳ only for a fresh clone — artifacts
`model_zoo/checkpoint/diffbrush_iam.pt` + `model_zoo/stable-diffusion-v1-5`
staged via `scripts/download_model_zoo.py`; `model_zoo/` is git-ignored):

```
cp .env.example .env            # adjust per machine (DEVICE, paths)
uv run --with gdown scripts/download_model_zoo.py   # fresh clone only
uv run generate.py
```

Host-machine-specific values live in `.env` (git-ignored; template `.env.example`):
`DEVICE` (`cuda|mps|cpu`), `PRETRAINED_MODEL`, `STABLE_DIF_PATH`, `SAVE_DIR`.
Precedence: CLI flag > process environment > `.env` > built-in default (built-in
defaults already point at `model_zoo/`).

**Probe texts** live in `probe_texts.txt` (repo root) — phase-independent by design:
the same three lines are probed in every phase (Phase-0 "before" baseline, all training
checkpoints via `generate.py --pretrained_model`, Phase-5 comparisons) so results are
comparable image-by-image. Locked set: English control; **"Schöne Grüße aus Leipzig"**
(ö/ü/ß — the 4b content-lock string); **"Straße 40, ähnlich zu Ölberg"** (ß/ä/Ö).
One sentence per line, `#` comments allowed; repeatable `--text` overrides the file.

A reference "before" baseline is already recorded under `reports/phase0_probe/`; a fresh
run of the locked Leipzig line should reproduce its findings: diacritics dropped (ö→o),
ß → "SS"-like pseudo-glyph, line-tail duplication hallucination — readable-but-rough,
the predicted glyph-CNN-conditioning behavior.

### What the two staged artifacts are, and where they are needed
#### The released DiffBrush checkpoint on IAM
**`model_zoo/checkpoint/diffbrush_iam.pt` (~1.1 GB)** — is the released DiffBrush checkpoint, trained
800 epochs on IAM English handwriting. Contents: the slim latent UNet (163 M params,
width 512), the `Mix_TR` style module it owns, and the 496-writer proxy anchor tables
`(496, 512)×2`. It is the **warm start for the whole transfer**:

- Phase 0: strict-load verification (proves our model reconstruction) + the untouched
  "before" baseline probes;
- Phase 3 / 4a: warm-start source — every tensor loads strict except the two proxy
  tables, which are force-re-initialized (496 → 308, shape mismatch);
- 4b / 4c: the IAM-drift probe every 5 epochs measures regression against this English
  prior;
- Phase 5: reference model for the English-aux / regression generation mode sets.

#### Stable Diffusion 1.5 VAE
**`model_zoo/stable-diffusion-v1-5/` (~335 MB, fp32 VAE weights only)** — the frozen
Stable Diffusion 1.5 VAE. **This is an upstream component of the original DiffBrush
implementation, not a reconstruction**: the released `generate.py` loads it via
`AutoencoderKL.from_pretrained(stable_dif_path, subfolder="vae")` (upstream default:
`runwayml/stable-diffusion-v1-5`) and freezes it (`requires_grad_(False)`), and the
upstream README's Model-Zoo section tells users to download it into `model_zoo/`. We
reuse it verbatim, unchanged.
The reason it is needed at all: DiffBrush does not generate pixels directly — it denoises
VAE **latents** `(4, 8, 128)` for 64×1024 lines (latents scaled ×0.18215). The VAE is
therefore only the codec between image space and latent space, needed at every point
where lines are encoded or decoded:

- Phase 0: decoding the sampled latents into the probe images;
- 4a / 4b: encoding real training lines into `z₀` for the ε-MSE objective, and decoding
  the per-epoch probes;
- 4c: the D_line discriminator's "real" branch is the **VAE round-trip of the real line**
  (both D branches share the codec, so the judge cannot cream off decode artifacts);
- Phase 5: decoding the generated lines that go to the acceptance-gate metrics.

# DRAFT for next phases

## Phase 1 — German line data preparation

**What the phase does.** The ScaDS.AI German handwriting corpus
([Zenodo 18301532](https://zenodo.org/records/18301532), CC-BY-4.0 — 5,843 line PNGs +
`line_annotations.csv` with per-line boxes and text) is turned into DiffBrush's *input
contract*:
`{split}/{wid}/{img}.png` folders on disk, label lines `"<wid>,<img> <transcription>"`, height-64 RGB images
(aspect-preserving resize; squash to 64×1024 only when wider), plus four audits: duplicates/missing/tiny/empty triage,
full charset census, Unifont glyph coverage, and per-page style-reference sufficiency (pages need at least one ≥512-px
line to serve as style refs; pages without one are dropped from train — logged). Executor:
`german/prepare_scads_lines.py`.

**Key decisions (fixed).**

- *Split:* fresh seed-42 80/10/10 over the line pages.
- *No `concat_short_img` augmentation:* it corrupts transcripts, which the Phase-5 gate refuses to train recognizers on.
- *Class ids as folder names:* upstream `BaseDataset` casts `int(wid)`, so zero-padded page ids would collide; train
  folders are numbered by class id.
- *Label hygiene:* line labels are consumed verbatim from the dataset's CSV — no silent relabeling (the corpus' known ~
  11 % label-normalization caveat is documented; flagged rows go to a CSV). The acceptance gate evaluates against
  dataset GT, so a mislabeled training line can hurt quality but never hides from the gate.

## Phase 2 — vocabulary & glyphs (deliberately small)

Unlike embedding-table transfers, DiffBrush conditions text as **glyph bitmaps through a CNN** — there is no embedding
table to surgically edit. Appending German characters is therefore weights-neutral by construction: the extended
`letters` constant in
`german/german_dataset.py` (IAM letters + the 19 audited extras, appended so existing indices do not move), 100 %
Unifont coverage (audited in Phase 1), and per-character 16×16 bitmaps (diacritics are part of the glyph — no special
casing). Residual risk (learned sufficiency of the content CNN for rare German glyphs) is checked at the Phase-5 gate,
not here.

## Phase 3 — writer classes & warm-start recipe

DiffBrush's style anchors need a class id per writer. ScaDS exposes no writer metadata, but its pages are single-page
documents, and the page-as-writer proxy was validated on the corpus itself (measured
same-page style-cosine cohesion 0.861 vs 0.812 cross-page — weak but real; and *within-page consistency* is precisely
what form extraction
needs). So class ids = staged train page ids, exported as
`writers_{split}.json`.

**The forced decision.** The pretrained anchor tables are `(496, 512)`; German needs 308 → strict loading of the warm
start is impossible for those two tensors. The shape mismatch **forces** fresh Kaiming re-init of both proxy tables; no
semantics are lost because proxies drive only the auxiliary style anchors, never the blender path. Loss-scale arithmetic
(measured): `L_diff` ≈ 0.06–0.12 but the anchor pair floats ≈ 16–23 — that imbalance becomes the named `style_weight`
dial in Phase 4.

## Phase 4 — the training itself (staged)

The upstream release ships **no training code**. The reconstructed trainer (`german/train_diffbrush_german.py`)
implements the paper's recipe (Appendix B) with every reconstruction decision annotated at its definition site in code.
Each stage keeps artifacts under `german/runs/<stage>/` and a ledger note.

### 4a — sanity: warm start + honest benchmark

Warm-start from the IAM checkpoint (proxies re-initialized, everything else strict-loaded), 100–300 steps at the ½–⅕-LR
convention with the ε-MSE diffusion objective (at a uniformly random `t ∈ [0,1000)`: `x_t = √ᾱ·z₀ + √(1−ᾱ)·ε`, loss =
`MSE(ε̂, ε)`), then benchmark s/step at batch 8/16 **on the active device** to size 4b honestly. Fallback ladder:
batch↓, then cloud. A second objective variant is only considered if the ε-MSE arm misbehaves in the samples.

### 4b — German-only fine-tune, with the measured LR dial

Continue 4a's winner on the German line split, amending as evidence arrives: per-epoch checkpoints, per-epoch dual
probes (plain-unguided primary — the Phase-0-comparable recipe — plus guided), a fixed-seed **recon-meter** (one-step
x̂₀ MSE at t=200/400/750 on a frozen training batch) as the content-assertiveness trend, and an IAM-drift probe every 5
epochs (regression guard for style generality).

**The dial the stage is really about.** The released checkpoint always holds *conditional content control for English*
(its IAM pretraining), and the first adaptation attempts at LR 2e-5 destroyed that binding (probes: styled ink,
arbitrary content). The fix is the learning rate, not data volume: **5e-6 gives the first genuine content lock** (
"Schöne Grüße aus Leipzig" — `probe_texts.txt` line 2 — rendered through most of the line prefix, stable across seeds/checkpoints), while 1e-5 gives
nothing and 2e-5 actively regresses. The
`style_weight 1.0 → 0.1` rebalance was tested and found neutral — kept at 0.1 since it costs nothing and spares the
anchors.

**Boundaries of the claim.** 4b outputs are *not* yet acceptable training data — prefix only. The gate lives in Phase 5;
4b's deliverable is the base checkpoint family with the dial history.

### 4c — the readability stage: D_line (multi-scale content discriminator)

**Why it exists.** The paper's own ablation (Fig. 7) is blunt: the generator trained with
`L_diff + L_style` only (which is exactly what 4b is) sits at **D_CER ≈ 55** (garbage readability) while the full model
with the content discriminator reaches **D_CER 8.59**. The released checkpoint reads crisply *because its 50-epoch
discriminator phase is baked in*; the plain loss cannot rebuild it on its own (recon-meter flat through 25+ epochs). So
4c adds the paper's mechanism: adversarial content supervision, λ = 0.05, on top of everything 4b already does.

**What runs.** Per step: a coarse generated image `x̂₀` is produced (default `--x0_mode
chain5` — the 5-step DDIM chain from pure noise, the paper recipe; the low-band `step1t`
at t∈[20,250] is the documented wall-time fallback), the D's real branch is the **VAE round-trip of the real line**
(both discriminator branches share the codec, so the judge cannot cream off decode artifacts), the glyph guidance
`I_line` is concatenated channel-wise, sliced into 32 left-to-right character-slices (paper Fig. 5a), and a small
3-layer 3-D CNN (`german/discriminators.py`) votes. Generators and discriminators update alternating (G:
`L_diff + w_s·L_style + 0.05·L_content` non-saturating BCE; D: plain BCE on the same pairs), with the CFG-dropped rows
(paper `p = 0.1` drop, implemented by a wrapper around `unet.mix_net` so the anchors never learn from blank references)
excluded from `L_content`.

**Stage plan and gates.** Before any 4b-style scale-up: a 4c benchmark branch (`chain5`
vs `step1t`) decides the first arm's `x0_mode` **on the active device** (MPS measured:
≈100 s vs ≈7 s/step); 5-epoch arms with pinned probes. Pass bands: plain probes + recon-meter not worse than the 4b
placebo (t200 0.126 / t400 0.578 / t750 1.078), D accuracy hovering 0.6–0.8 (pinned at 1.0 means the D won → λ or d_lr
dimmed), IAM drift unchanged.

### Assets & bookkeeping

Per-epoch `models/ckpt.pt` (+ `d_ckpt.pt` once 4c runs), `history.json` flushed every 25 steps (crash-safe), probe pngs
per epoch, `config.json` per run — any result re-derivable from the checkpoint + a config alone.

## Phase 5 — generation + the acceptance gate

Once a checkpoint family shows content lock in the probes, it still has to prove that lines produced by it are *usable
as recognizer training data* — the whole motivation.
`german/check_ckpt_quality.py` + generation mode sets (German styles × German text = the acceptance-critical one; German
text under IAM styles = the English-aux regression; IAM-drift check). The gate is evaluation, not vibes:

| metric                                                                                                      | why it is in the gate                                                                           | pass band                               |
|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------|
| German-HTR CER delta (recognizer trained on real lines vs on synthetic lines; evaluated on real test lines) | the motivation's own metric: synthetic data must teach a recognizer nearly as well as real data | synthetic ≤ 1.5× real-trained CER       |
| per-character CER: ä/ö/ü/ß vs base letters                                                                  | the umlaut story must not be a silent regression                                                | ≤ 2× base-letter CER                    |
| inter-word spacing / vertical-alignment distribution check                                                  | the capability the whole line-level detour was bought for                                       | ratio ≤ 2× vs real, or qualitative pass |
| page-style consistency                                                                                      | conditioning should bind to the requested writer                                                | reported (no hard gate v1)              |
| short-word audit (≤ 3 chars)                                                                                | short tokens historically explode in transferred generators                                     | flag-only                               |
| human check                                                                                                 | blind folding against real sheets                                                               | ≥ 80 % unflagged                        |

## Decision gates & risks (living)

- **umlaut/ß pass** → scale to paragraph-level generation next, same device discipline.
- **umlaut/ß fail** → ordered escalations: (a) umlaut-dense oversampling epochs at the proven LR; (b-mix) IAM+German
  mixed-line arm with a shared proxy table; (c) fall back to assembling lines from
  word-level generation (cost: known CER gap vs direct line synthesis); (d) small real-capture top-up.
- **only the mix passes** → the mixed recipe becomes the shipping configuration.

| risk                                                   | mitigation                                                                                                |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| upstream torch-1.13 idioms on torch 2.14               | Phase-0 smoke exercised every forward/backward path; each incompatibility fixed and marked `German patch` |
| MPS OOM at 1024-px latents + 512-token style attention | measured ≤ 1.4 GB at batch 16 — ladder never needed; batch↓ + grad accumulation remain the knobs          |
| from-scratch-scale compute (paper: 800 ep × 8 GPUs)    | warm-start + staged schedule; the LR dial (not data volume) is the expected unlock                        |
| proxy anchors destabilize                              | `style_weight` dial; the 4a control run exonerates them — kept on at 0.1×                                 |
| discriminator stage destabilizes the generator         | D accuracy 0.6–0.8 band + λ = 0.05 fixed by paper; D-LR dimmed if pinned                                  |
| ScaDS↔IAM geometry/ink shift                           | canonical h=64 resize at prep; lighter German ink is *style-true*, not damage                             |
| label caveat (~11 % normalization)                     | flags-only triage; the gate evaluates against dataset GT so noise cannot hide                             |

## Staging notes (before Phase 1)

Artifacts to be staged in this repo when training work starts:

- the `german/` package: `train_diffbrush_german.py` (reconstructed trainer),
  `german_dataset.py` (letters table + loaders), `discriminators.py`, `probe_release.py`,
  `check_ckpt_quality.py`, `prepare_scads_lines.py`;
- the staged corpus `german/scads_lines/` (113 MB: train/val/test, labels,
  `writers_{split}.json`) — Phase 1's output;
- the reference run ledgers (`reports/phase4_stage4a.md`, scads_lines audit reports).

Portability patches to apply when staging (CUDA + MPS):

1. follow the `.env` convention (template `.env.example`): the `german/` tools currently
   hard-wire `mps→cpu` detection or default `--device mps` — replace with `load_dotenv()`
   + env-backed defaults (`DEVICE`), same as `generate.py`;
2. VAE path defaults pointing at a foreign staging dir →
   `model_zoo/stable-diffusion-v1-5` — done for `generate.py`; the `german/` tools still
   need it when they are staged (env var `STABLE_DIF_PATH`).
