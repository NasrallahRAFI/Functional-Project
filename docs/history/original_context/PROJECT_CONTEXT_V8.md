# PROJECT_CONTEXT.md — AI-based Drone Detection (ML course project) — V8

> Single source of truth for this project. Humans and AI coding assistants (ChatGPT, Antigravity) must read this file fully before writing or changing any code.
> Tags used below: **[GIVEN]** stated by the project owner · **[ASSUMPTION]** chosen by us, not confirmed · **[VERIFIED]** checked against a source · **[TO CHECK]** not yet verified.
> **V6 (2026-11-22):** the data facts in sections 3, 6 and 6b.3 and in D003, D023, D024, D028 and D039 are corrected against Luna's exploration run, documented and audited in `DATA_PREPARATION_v5.md`. Where the two files disagree on a data fact, `DATA_PREPARATION_v5.md` wins. Decisions are unchanged unless marked **[V6]**; D040 and D041 and open questions 4 to 6 are new.
> **V7 (2026-11-22):** the model ladder is extended to the owner's full list of course methods (SVM, bagging and AdaBoost added; random forest kept; perceptron and MLP are one notebook). Sections 1, 5, 6, 7, the status list and open question 3 are updated; D042 is new and D038 is revisited. Nothing else changed; data facts and the front end are as in V6.
> **V8 (2026-11-22):** all open questions that blocked the split are answered (the owner accepted the assistant's recommendations): `L` = 0.5 s (D023), D039 approved, `extra_membo_D2` pinned to train and not evaluated and `B_S2_D1` grouped in 13 blocks (D040), near-silent rule approved (D041). D043 is new. Sections 3 (items 2, 7, 8), 6, 11 and the status list are updated; the V6 dry-run group counts are withdrawn.

---

## 1. Purpose and scope

- **[GIVEN]** Course project in a PhD ML course. The methods to try, as listed by the owner [V7]: linear regression (simple and multiple; least squares and normal equations; gradient descent), logistic regression, k-nearest neighbours, support vector machines (linear and RBF kernel), decision trees, bagging, boosted decision trees (AdaBoost with decision stumps), perceptron and MLP (one notebook, D042), CNN. Random forest, which the earlier V6 list had, is kept as well (D042).
- **[GIVEN]** We must try **all** of these algorithms on one problem and **judge each one**. The **final algorithm is a CNN**.
- **[GIVEN]** Topic: AI-based drone detection.
- **[GIVEN]** This is **not** an ML thesis and **not** a drone-detection thesis. **No hardware prototype.** Quality work is expected, but the most important deliverable is the **justification of every choice, big or small**.
- **[GIVEN]** Dataset: one dataset, not dirty (no major cleaning), under 1 GB.
- **[GIVEN]** Environment: Google Colab, several notebooks (a single notebook would be too big). Code is produced with ChatGPT and Antigravity.
- **[GIVEN]** **Library-first rule:** everything that already exists in an established Python library, and is better than what we would write, is taken from the library (audio I/O, resampling, STFT/mel/MFCC, scikit-learn models, metrics, plotting). Nothing is hand-written "from scratch" for its own sake.
- **[GIVEN]** **The CNN's architecture is our own design**, because it is the end goal of the project: no pretrained weights, no ready-made architecture imported from the internet. It is nevertheless built from standard library layers (convolution, normalisation, pooling, dense) and trained with library tools (optimizers, schedulers, losses). Ready libraries are used even when training our own models.
- **[GIVEN]** Final demo: run the trained model on the owner's laptop; a drone sound is played and the program outputs "drone" or "no drone". Software only (laptop microphone or audio file), see section 9b.

### Out of scope
Production or embedded deployment, hardware, microphone arrays, localisation/tracking, drone model identification (multiclass), GAN augmentation, pretrained/transfer-learning models (unless the owner explicitly adds them later).

---

## 2. Problem definition

- **[GIVEN]** Modality = **audio (acoustic detection)**, confirmed by the owner (D001). The original request did not name a modality; audio was chosen for these reasons: small clean public audio datasets exist; the same audio can be represented as a fixed-length feature vector (for classical models) and as a spectrogram image (for the CNN), which gives a fair comparison ladder; no hardware needed.
- **Task:** binary classification of an audio clip: `drone` (1) vs `no drone` (0).
- **Why binary:** it is the detection question; multiclass identification adds scope without adding to the course objective.
- **Why metrics matter more than accuracy:** the two error types are not equal in a detection setting (missed drone vs false alarm). See section 7.

---

## 3. Dataset

### Chosen candidate (pending the audit notebook)
**Al-Emadi DroneAudioDataset** (GitHub: `saraalemadi/DroneAudioDataset`).

- **[VERIFIED]** Per its README: drone propeller noise recorded **indoors** by the author and **artificially augmented with random noise clips**; the "Unknown" (non-drone) class uses clips from **ESC-50** and **white noise from Speech Commands**, plus a **silence clip created by the authors** to balance the dataset. It has **binary** and **multiclass** folders. We use the **binary** folder.
- **[VERIFIED]** It was released with the paper *Audio Based Drone Detection and Identification using Deep Learning* (IWCMC 2019); cite the paper and the repo.
- **[MEASURED, Luna's exploration of the downloaded zip, recomputed from `inventory.csv` and audited in `DATA_PREPARATION_v5.md`]** *Contents of the binary folder:* 11,704 clips. `yes_drone` holds **1,332** clips: 666 `bebop` and 666 `membo`, in **312 groups** (255 recordings of 5 one-second segments, 57 single clips: `B_S2_D1_133-bebop` and the 56 `extra_membo_D2_####`). `unknown` holds **10,372** clips: **10,000** ESC-50-style names (in **1,528 groups of 5 to 40 clips**, not 2,000), **301** `silence###.wav`, and **71** other noise (stems `doing_the_dishes`, `exercise_bike`, `running_tap`, `pink_noise`, `white_noise`, plus `1-7973-A-7_1002.wav`). Class balance: **88.6 % unknown, 7.79 : 1**.
- **[MEASURED, all 23,408 WAV files]** *Audio format:* 16 kHz, mono, 16-bit, 0 truncated. Lengths: drone 16,384 samples for 1,038 clips (shortest 10,399); ESC clips 16,347 (6,000), 15,604 (2,000) and 15,356 (2,000); 67 of the 71 other-noise clips about 5 s; `silence###` 24 to 17,833 samples. Size of the binary folder about **384 MB** (decimal), under the 1 GB requirement. One ESC clip (`2-102581-A-291.wav`) made Luna's descriptor script fail; it decodes with `soundfile` and is 99.88 % zeros (see below).
- **[MEASURED]** *Digital silence is a large share of the unknown class:* **1,089 exact-zero clips (10.5 % of the unknown class, 9.3 % of the folder)** in **10 distinct contents**: 7 groups of byte-identical files (330, 308, 191, 150, 57, 35 and 15 copies = 1,086 clips) plus 3 single clips (`silence000`, `silence299`, `silence300`, the last only 24 samples). 301 are named `silence###`; 788 carry ESC-style names and look like silence padding at the end of short source clips (the share of exact-zero clips rises from 0.4 % at segment 0 to 16.5 % at segment 4; own inference from names). A further 244 non-zero `unknown` clips have more than half of their samples exactly zero and 57 are below −80 dBFS (including `2-102581-A-291.wav`). No identical audio occurs under both labels.
- **[VERIFIED]** *Licence, read from the repo's own `LICENSE.md`* (replaces the third-party report in D031): see D031 for the clauses and what they mean for this project. The owner accepts the security-use clause for this course project.
- **[ASSUMPTION]** The owner's requirement "not dirty, no major cleaning" holds with small, logged actions only: 2 clips shorter than half a window are rejected (D024) and the exact-zero duplicates are collapsed (D039); everything else is flagged, not deleted.
- **[TO CHECK]** The licences of the third-party material inside the dataset (ESC-50 and Speech Commands) follow their own terms (LICENSE section 6); ESC-50 is reported as CC-BY-NC 3.0 by a third party (D031).

### Known limitations (to be stated in the final report, not hidden)
Full measured detail: `DATA_PREPARATION_v5.md`, sections 2 and 3.
1. **Indoor recording and only two drone models** (`bebop` and `membo`, 312 groups) → results will not transfer automatically to other drones, or to outdoor, distant, windy conditions.
2. **Segment leakage is structural.** Drone recordings are cut into up to 5 consecutive one-second segments; ESC groups hold 5 to 40 clips. A random clip-level split would inflate every model's score; group-aware splitting is mandatory (section 6). Neighbouring `B_S2_D1` recordings are also similar to each other (correlation about 0.55 between recording *n* and *n*+1), so one group per recording is a minimum for that family [V6]; blocks of consecutive recordings are used instead (D040, V8).
3. **Non-drone class is partly artificial and narrow:** 10.5 % is digital silence (mostly ESC padding), 70 files come from five noise stems, the rest is ESC-50 found sounds. Whether it contains speech or music is not established **[TO CHECK]**, so the demo's speech and music negative checks may be outside the training distribution.
4. **Class and source are confounded, but not by overall loudness** [V6, replaces the earlier four-clip level claim, withdrawn]: median levels of non-zero clips are equal (−24.8 against −24.6 dBFS). What differs is the set of recording artefacts: clipping (206 clips), DC offset (116) and exact-zero padding (1,089 clips plus 244 with more than half zeros) occur only in `unknown`, and the drone class is steadier (frame level std 2.3 against 4.5 dB). A model can learn artefacts instead of the drone; test scores will be optimistic for a laptop microphone.
5. **Imbalance of 7.79 : 1** (6.98 : 1 after the cleaning of D024 and D039).
6. **Clip length differs between classes** (drone mostly 16,384 samples, ESC 16,347 / 15,604 / 15,356, plus about 5 s noise clips). The fixed crop of `L` = 0.5 s removes length as a cue for all but 2 clips; length is never a feature.
7. **Small, narrow evaluation sets for the drone class:** about 180 to 190 drone clips in each of validation and test, but far fewer independent groups (the V6 dry-run counts of 28 and 38 to 39 no longer apply; notebook 01 reports the real ones). Bootstrap over groups.
8. **`extra_membo_D2` (56 clips) is mostly below 50 Hz** (87.5 % of its clips have rolloff at or below 100 Hz; the front end starts at 50 Hz) and, by owner decision (D040, V8), is pinned to train and not evaluated. Two drone-labelled clips (`mixed_membo_17-membo_003_`, `_004_`) are effectively silent.

### Rejected alternative
**DADS** (Drone Audio Detection Samples, Hugging Face): [VERIFIED] 16,729 no-drone files (33.81 h) + 163,591 drone files (27.10 h), 16 kHz mono 16-bit WAV. By our own arithmetic (~219,000 s × 16,000 samples/s × 2 bytes) that is **about 7 GB uncompressed**, above the 1 GB limit, and it merges several heterogeneous sources, so it would need more cleaning. Rejected for the size and cleaning criteria.

---

## 4. Workflow method: "freeze, then ladder, then judge once"

1. **Freeze the protocol before any model exists.** Task, dataset, split, metrics and feature sets are fixed in this file and in `config.yaml`. *Why:* if the split or metric changes after seeing results, the comparison between algorithms is no longer fair.
2. **Build the data once, and store it.** Audit → clean → split → features, each as its own notebook writing files to Drive. Every model notebook reads the *same* files. *Why:* identical inputs are what make "judge each algorithm" meaningful; Colab sessions reset, so only what is saved on Drive counts.
3. **One notebook per algorithm**, all with the same template (section 8). *Why:* a failed or slow run never blocks the others, and each notebook is short enough to review.
4. **Tune on validation only; score the test set once.** Each model notebook ends by *freezing* its chosen model and validation results. Only `90_comparison` scores the frozen models on the test set. *Why:* repeated test peeking silently turns the test set into a second validation set.
5. **Every choice goes in the decision log** (section 10) at the moment it is made, with alternatives and evidence. *Why:* this is the main graded quality criterion, and it is cheap to write now and hard to reconstruct later.
6. **Logic in `src/`, narration in notebooks.** Notebooks call functions and display results; reusable code lives in `src/dronedet/`. *Why:* AI-generated code is easier to review, test and reuse as plain `.py` files, and notebooks stay readable.

---

## 5. Folder architecture

```
drone-ml-project/                 (code: Git repo; data/results: on Google Drive)
├── PROJECT_CONTEXT.md            # this file
├── config.yaml                   # seed, paths, sample rate, split ratios, feature parameters
├── requirements.txt
├── notebooks/
│   ├── 00_setup_and_data_audit.ipynb
│   ├── 01_cleaning_and_split.ipynb
│   ├── 02_feature_extraction.ipynb       # also holds the front-end checks (6b.7) and writes data/processed/golden/
│   ├── 10_linear_regression.ipynb
│   ├── 11_ml_regression.ipynb            # linear regression trained by gradient descent (D038, D042)
│   ├── 12_logistic_regression.ipynb
│   ├── 13_knn.ipynb
│   ├── 14_decision_tree.ipynb            # single decision tree (D038)
│   ├── 15_random_forest.ipynb
│   ├── 16_bagging.ipynb                  # bagged trees, all features at every split (D042)
│   ├── 17_adaboost.ipynb                 # boosting with decision stumps (D042)
│   ├── 18_svm.ipynb                      # linear and RBF-kernel SVM (D042)
│   ├── 20_neural_network_mlp.ipynb       # perceptron and MLP in one notebook (D042); on the SAME tabular features
│   ├── 30_cnn_spectrogram.ipynb          # final algorithm: part A baseline; part B front-end selection and sweep (6b.4, 6b.10); then final tuning
│   ├── 90_comparison_and_final_test.ipynb
│   └── 91_export_model_bundle.ipynb      # packs the final CNN for the laptop demo
├── demo/
│   ├── run_demo.py               # laptop program: file mode and live-microphone mode
│   └── requirements_demo.txt
├── src/dronedet/
│   ├── config.py                 # loads config.yaml, resolves paths
│   ├── data.py                   # loading, audit helpers
│   ├── splits.py                 # split creation and loading
│   ├── features.py               # tabular features + log-mel spectrograms
│   ├── metrics.py                # all metrics, one place
│   ├── training.py               # plain-PyTorch training loop shared by MLP and CNN (no wrapper library)
│   ├── plotting.py
│   └── models/                   # model builders (sklearn / CNN)
├── data/                         # on Drive, NOT in Git
│   ├── raw/                      # downloaded dataset, never modified
│   ├── interim/                  # cleaned audio (if any cleaning happens)
│   └── processed/
│       ├── splits/               # train/val/test manifests (CSV: file, label, split, group)
│       ├── features_tabular/     # .npz per split
│       ├── spectrograms/         # mel-power .npy + index CSV per split (6b.1)
│       └── golden/               # synthetic WAVs + expected log-mels + report filterbank matrix (6b.7)
└── results/                      # on Drive
    ├── cards/                    # one JSON "verdict card" per model
    ├── models/                   # frozen models
    ├── figures/
    └── tables/
```

**Why Git for code and Drive for data:** Git gives history and lets the AI IDE edit files; Drive survives Colab resets and holds files too big for Git. **Why numbered notebooks:** the number tells the run order and groups notebooks by stage (0x data, 1x classical, 2x NN, 3x CNN, 9x comparison).

---

## 6. Data pipeline rules

- `raw/` is read-only. Any cleaning writes to `interim/`, and each cleaning action is logged with counts (how many files affected and why).
- **Audit (notebook 00):** count files per class, total size, sample rate, channels, bit depth, duration distribution, corrupted/empty files, exact duplicates (hash) with group sizes per class, class balance. Added after the file-list inspection: the per-class **length histogram against `win_samples` for every `L` candidate**; a **per-clip level histogram** (RMS and band power in dB) to find near-silent clips that are not exactly zero; parsing of the group keys below with a count of clips per group; a check that no group spans the two classes. Output is a table plus a short written conclusion.
- **Standardise:** one sample rate, mono, fixed window length `L` (rules in 6b.3). The chosen values go in `config.yaml` with a justification.
- **Split [V6]:** train / validation / test, stratified by class, **fixed seed**, saved as manifest CSVs. Procedure: a two-stage `StratifiedGroupKFold` (7 folds, fold 0 = test; then 6 folds on the rest, fold 0 = validation), about 71 / 14.3 / 14.3, with the exact-zero representatives and their groups pinned to train (D039). This replaces the earlier 70/15/15 proposal; the recipe is in `DATA_PREPARATION_v5.md`, step 5. **[V8]** `extra_membo_D2` is pinned to train as well (D040). With the owner decisions made, the split is **ready to freeze**: notebook 01 builds it, checks the gates and hashes the manifests.
- **Leakage control (group-aware split is mandatory; the file names allow it):** one group per source recording, so that all segments of a recording land in one split. Group keys: drone clips, the file name up to the last `_###_` (e.g. `mixed_9-bebop`), except that the 67 `B_S2_D1` recordings are merged into 13 blocks of consecutive recordings (D040, V8; 203 drone groups in all); the 56 `extra_membo_D2_####` clips, one group, pinned to train (D040); ESC-50 clips, the first two fields of the name (`fold-freesoundID`, e.g. `1-100032`; **1,528 groups**, not 2,000); `silence###`, one group per file; other noise, one group per stem plus one for `1-7973-A-7_1002`. Use a library group-aware splitter (`StratifiedGroupKFold`). The audit found no group under both labels. A split by recording session cannot be derived from the files (the meaning of `S`, `D`, `mixed`, `Membo_k` is unknown), and neighbouring `B_S2_D1` recordings correlate (handled by D040). Exact duplicates are handled by D039.
- **Normalisation statistics (mean/std) are computed on train only**, then applied to validation and test.
- **Imbalance:** measured at 7.79 : 1 (6.98 : 1 after D024 and D039); handled by class weights and stratified grouping, decision logged in notebook 01.

### Two representations of the same audio
| Representation | Used by | Why |
|---|---|---|
| Tabular features per clip (84 level-free features: MFCC c1–c19 statistics and mel-domain spectral descriptors, see 6b.5) | linear/ML regression, logistic regression, kNN, SVM, tree, bagging, random forest, AdaBoost, MLP | these models need a fixed-length vector |
| Log-mel spectrogram (2D, time × frequency) | CNN | a CNN exploits local time-frequency structure |

The MLP runs on the **tabular** features on purpose: if the CNN beats the classical models, comparing it with an MLP on the same features helps separate "the CNN algorithm is better" from "the spectrogram input is richer". **[ASSUMPTION]**

---

## 6b. Front-end specification (adapted from the owner's existing front-end report)

**Source and status.** Parameter values and design decisions come from the owner's earlier from-scratch front-end report (16 kHz, 1024-point FFT, hop 256, 64 HTK mel bins) and are reused as the **`reference` preset**. The implementation uses established libraries (library-first rule), **not** the report's code. Every decision below has a log entry (D012–D016 and D021–D036, section 10). All of them are **[ASSUMPTION]** until the owner confirms. The report's values are a starting preset, not a target: each parameter is a config key, chosen on merit and open to the bounded sweep in 6b.10. The items that still need a measurement or a check are listed in 6b.8.

### 6b.1 The pipeline (one definition for notebooks, training and demo)

Code lives in `src/dronedet/features.py` (config dataclass in `config.py`). The demo imports it and never re-implements it (rule 7).

1. **Load.** `soundfile`, float32 in [−1, 1]. Multichannel input: use channel 0 only, as an explicit config key `channel`.
2. **Resample** to `sr` with `soxr.resample`, quality `HQ`, only when the native rate differs from `sr`. Live mode uses `soxr.ResampleStream` with the same quality, so file mode and live mode share one filter design.
3. **Length handling** at waveform level (6b.3).
4. **Mel power.** `librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, win_length=1024, window="hann", center=False, power=2.0, n_mels=64, fmin=50, fmax=F_MAX, htk=True, norm="slaney")`, float32.
5. **Cache the mel power** of each full standardised clip (power, not dB), frames concatenated along time in one `.npy` per split plus an index CSV (`file, start_frame, n_frames`) in `processed/spectrograms/`. *Why power:* gain augmentation and PCEN both need it (6b.4).
6. **Window.** Take `n_win` consecutive frames from the cached clip (derived values below). With `center=False`, each frame depends only on its own 1024 samples, so a frame-level crop at a hop multiple equals a waveform crop followed by the STFT (own reasoning; the golden test in 6b.7 checks it). Training: random offset, redrawn every epoch. Validation, test and tabular features: one fixed centred window per clip, so the metric unit is the clip.
7. **Transform and normalise** according to the mode (6b.4) to a float32 tensor of shape `(1, 64, n_win)`.

**Derived values (never typed):** `n_win = 1 + floor((L·sr − n_fft) / hop)` and `win_samples = n_fft + (n_win − 1)·hop`. The demo buffer is exactly `win_samples` long. *Own arithmetic, run:* at 16 kHz, L = 0.5 / 1.0 / 1.5 / 2.0 s gives 28 / 59 / 90 / 122 frames; one frame is 64 ms long with a 16 ms hop and the FFT bin spacing is 15.625 Hz. The report's 128 frames need 33 536 samples (2.096 s). *Nominal vs usable length (own arithmetic, run):* `win_samples` is slightly below `L·sr`: 7 936 / 15 872 / 23 808 / 32 000 samples for L = 0.5 / 1.0 / 1.5 / 2.0 s, i.e. 64 / 128 / 192 / 0 samples fewer (0.496 / 0.992 / 1.488 / 2.000 s). **Every length rule below uses `win_samples`, never the nominal `L·sr`.**

### 6b.2 Parameter table (final values)

| Parameter | Value | Why |
|---|---|---|
| `sr` | 16 000 Hz, or the dataset's native rate if that is lower (audit decides) | the report's rate; drone rotor harmonics sit far below 8 kHz; one rate for training and demo |
| `channel` | mono, channel 0 | dataset is mono; explicit selection is deterministic (averaging array microphones can cancel) |
| `n_fft` / `win_length` / `hop` | 1024 / 1024 / 256 | report values; 15.6 Hz resolution resolves rotor harmonics, 16 ms hop gives about 60 frames per second of audio |
| window | Hann (librosa default, periodic), `center=False` | report convention (no padded edge frames) |
| `n_mels`, `fmin` | 64, 50 Hz | report values; below 50 Hz laptop microphones and speakers have almost no response; passes the starvation rule with 1.75× margin at fmax 7000 (1.84× at 8000; 6b.6) |
| `fmax` (`F_MAX`) | interim **7000 Hz**, final = largest multiple of 250 Hz below the measured −1 dB point of the resampler (never above 8000) | the top mel band must not overlap the resampler transition band; see below |
| mel scale / norm | `htk=True`, `norm="slaney"` (constant-area triangles) | **[GIVEN]** the report's config has `mel_normalization: constant_area`, each triangle scaled by `2 / (high − low)`, on the HTK scale; `librosa`'s `norm="slaney"` divides each triangle by its band width in the same way, and the newer D8 feature catalog also uses area-based Slaney normalisation on the HTK mel scale (owner-confirmed), so all three agree. The equivalence is checked by the filterbank test in 6b.7. Area normalisation also holds on merit, not only on precedent: each band estimates power per Hz, so flat noise gives a flat spectrogram (−0.2 dB band to band) while peak-1 triangles give a +9.6 dB upward slope for the same noise (own arithmetic, run). HTK and Slaney *scales* are close at 64 bands (22 vs 21 filters below 1 kHz); HTK has the finer lowest band (27 vs 44 Hz half-width), useful for rotor fundamentals. `mel_scale` {htk, slaney} and `mel_norm` {area, peak} are config keys; `mel_scale` enters the sweep in 6b.10, `mel_norm` does not while the mode is `global_stats`, where it is a provable no-op for the CNN (D035) |
| dB conversion | `power_to_db(S, ref=1.0, amin=1e-12, top_db=None)` | fixed absolute floor (−120 dB). With area normalisation, 16-bit quantisation noise gives a band power of about −87 dB (simulated: −87.6 to −86.8 dB over the 64 bands), only 13 dB above a 1e-10 floor; 1e-12 leaves 33 dB, so real quiet recordings never touch the floor. `top_db` would clip relative to each call's own peak, which makes the output depend on window level and on the other content of the window |
| window length `L` | audit rule, 6b.3 | |
| resampler quality | `HQ` | `soxr` default; the cost difference to `VHQ` is irrelevant at these sizes, and one setting everywhere gives parity |
| PCEN parameters (`pcen` condition only) | `sr` 16 000, `hop_length` 256, `time_constant` 0.4, `gain` 0.98, `bias` 2, `power` 0.5, `eps` 1e-6, input `sqrt(power) × 2^31`, `init` = window mean | explicit config keys, included in the config hash so a library default change cannot alter the front end unnoticed (6b.4, D036) |

**Why `fmax` falls from 7500 to 7000 Hz (changes D015's interim value).** Training audio is not resampled when the dataset is already at 16 kHz, but the demo audio always passes through `soxr` (microphones run at 44.1 or 48 kHz). The two paths then differ in the resampler's transition band. If the passband ends near 0.91 of Nyquist, about 7.3 kHz, which is my recollection of libsoxr's default and **[TO CHECK]**, 7500 Hz would sit inside it. Choosing `fmax` below the edge makes the two paths identical in band. The cost is the 7–8 kHz band; propeller noise is dominated by lower frequencies **[ASSUMPTION]**. Measurement: resample a log-swept tone from 48 kHz to 16 kHz with the chosen quality, take the first frequency where the level falls 1 dB below the level at 1 kHz, apply the rule in the table. One run, result into `config.yaml`.

### 6b.3 Window length and clip handling

- **Rule for `L`:** candidates {0.5, 1.0, 1.5, 2.0} s. `L` is the largest candidate for which at least 95 % of the clips **of each class** have at least `win_samples(L)` samples (the usable window of 6b.1, not the nominal `L·sr`: 15 872 rather than 16 000 samples at L = 1 s). *Why:* with the nominal length a clip of, say, 15 990 samples would count as too short for L = 1 s although it holds a full window, which can push `L` down by a whole grid step. Why a grid: a deterministic, reportable choice. Why 2 s cap: the report's 2.1 s was the reference ceiling and latency grows with `L` (6b.7). **Stop condition:** if even 0.5 s fails the 95 % test in either class, stop and ask the owner; do not improvise.
- **Clips with at least `win_samples` samples:** cropped as in 6b.1 step 6; no padding.
- **Clips with between `win_samples/2` and `win_samples` samples:** padded to `win_samples` at waveform level with `np.pad(mode="wrap")`. *Why not zeros:* zero padding creates digital silence, and the dataset's no-drone class contains 1,089 digital-silence clips (10 distinct contents; only 301 are the authors' `silence###` files), so zero-padded drone clips would acquire a "silence" cue. Wrapping keeps the clip's own spectral statistics.
- **Clips with fewer than `win_samples/2` samples:** rejected in notebook 01 and logged with counts per class (a cleaning action under section 6).
- Counts of padded and rejected clips per class go into the audit table and the decision log.
- **Measured outcome of the `L` rule [V6, from the exploration; notebook 00 must reproduce it]:** the ESC segments are 16,347 (60 %), 15,604 (20 %) and 15,356 (20 %) samples, so at `L` = 1.0 s only 60.5 % of the `unknown` clips (6,276) and 81.7 % of the drone clips (1,088) hold a full window of 15,872 samples. The 95 % rule therefore returns **`L` = 0.5 s** (28 frames, `win_samples` 7,936): at 0.5 s 100 % of both classes have full windows, no clip is padded, and exactly 2 clips are rejected (`silence300.wav`, 24 samples, and `doing_the_dishes019.wav`, 1,266 samples). Accepted by the owner: 0.5 s (V8, D023); the alternative of relaxing the rule to keep 1.0 s would wrap-pad 39.5 % of `unknown` and 18.3 % of drone clips, unevenly.

### 6b.4 Representation, normalisation and level handling

Four conditions share the pipeline up to step 6. `S` is the mel power window.

| Mode | Definition | Role |
|---|---|---|
| `global_stats` (**default**) | `x = (dB(S) − μ_k) / σ_k` per mel bin `k`; `μ, σ` computed on the **train split only**, over all frames of the full cached clips, at 0 dB gain, **excluding clips flagged `silent` (D028)** so that frames clamped at the −120 dB floor do not enter the statistics; the excluded count is logged | the report's choice; the exclusion keeps the statistics free of floor-clamped cells (D035) |
| `per_window_mean` | `x = (dB(S) − m) / σ_g`, where `m` is the **single scalar** mean of the whole window and `σ_g` a scalar std taken on train windows after removing `m` | exactly invariant to gain. *Own arithmetic, run:* a +6 dB gain shifted every cell of the dB spectrogram by exactly 6.0 dB. A per-bin mean would be wrong here: it removes the stationary spectral shape, which is the drone's signature |
| `pcen` | `librosa.pcen(S_in, sr=16000, hop_length=256, time_constant=0.4, gain=0.98, bias=2, power=0.5, eps=1e-6)` with **every value an explicit config key** (not librosa defaults; the defaults `sr=22050`, `hop_length=512` would shorten the 0.4 s smoother to 0.28 s at this frame rate), `S_in = sqrt(cached mel power) × 2^31` (**[TO CHECK]** the 2^31 convention against librosa's documentation), smoother initialised from the window mean per band, then one scalar standardisation (train statistics) | designed to be gain-robust. Always computed per window **after** cropping, so training matches the demo, where windows are independent. In raw cache units 92.5 % of cells lie below `eps` and PCEN is as gain-sensitive as dB (0.80 std at +12 dB); with the scaled input it falls to 0.15 std (synthetic data, D036). **Acceptance test:** the output change at ±12 dB on train windows must be ≤ 0.2 std, otherwise the condition is not run |
| `logstft` | linear 513-bin dB spectrogram, `global_stats` | ablation only; its 8× larger frequency axis changes the CNN input size, so a win is informative but not a like-for-like comparison |

**Gain augmentation (all conditions, training only):** multiply `S` by `10^(g/10)` with `g` drawn uniformly from [−12, +12] dB, applied to the cached mel power before the transform. *Why in the power domain:* it equals scaling the waveform by `10^(g/20)` (above the −120 dB floor) without recomputing the STFT, and PCEN, which is not additive in dB, can still use it.

**Selection protocol (pre-registered, runs in notebook 30 part B, validation only).**
1. Fixed reference CNN: the default-hyperparameter baseline (notebook 30 part A). 4 conditions × 3 seeds.
2. Evaluate validation F1 at gains {−12, −6, 0, +6, +12} dB; also report ±18 dB as extrapolation, outside the rule.
3. `score(c) = mean over seeds of the minimum F1 across the five gains`. Choose the highest score. Guard: the chosen condition's F1 at 0 dB must not be more than 0.02 below the best condition's F1 at 0 dB (so "invariant but weak" cannot win).
4. If several conditions are within one seed std of the best score (floor 0.01), take the simplest in the order `global_stats` < `per_window_mean` < `pcen` < `logstft`.
5. One extra control, `global_stats` **without** gain augmentation, 3 seeds, to quantify what the augmentation buys.
6. The winner is frozen into `config.yaml` before the final CNN tuning. Tabular features are derived from the dB mel and do not depend on the mode, so notebooks 10–20 can run before notebook 30 part B.

### 6b.5 Tabular features (classical models and MLP): final set

One front end feeds both representations (D005). Features come from the **same centred window** as the validation and test spectrograms. All 84 are level-free, i.e. unchanged by a gain.

- **MFCC c1–c19, each summarised by mean, std, min, max over time = 76.** `librosa.feature.mfcc(S=dB_mel, n_mfcc=20, dct_type=2, norm="ortho")`, then drop c0. A gain adds a constant to the dB spectrogram, which changes only c0.
- **Mel-domain spectral centroid, bandwidth, rolloff, flatness (4 features), each mean and std = 8,** computed from the mel power with the mel centre frequencies (`librosa.mel_frequencies`).

| Group | Features | Count |
|---|---|---|
| MFCC c1–c19 | mean, std, min, max over time | 76 |
| Mel-domain descriptors | centroid, bandwidth, rolloff, flatness: mean, std | 8 |
| **Total** | | **84** |

- **Excluded on purpose:** c0 and RMS (both carry the recording level, so a model could learn "loud means drone" from the dataset's mixing, which would not hold on the laptop microphone) and zero-crossing rate (waveform-only, would need a second cache). They can be added as a documented ablation. Exact `librosa` keyword arguments **[TO CHECK when coded]**.
- Note for the report: the CNN trains on random windows with gain augmentation; the classical models and the MLP train on the one fixed window per clip. The augmentation is part of the CNN's training recipe, not of the front end.

### 6b.6 Validation rules and edge cases

A validation function runs at config load and refuses configs that break these rules:
1. `n_samples ≥ n_fft` (otherwise zero frames).
2. `fmax ≤ min(Nyquist, measured passband limit)`.
3. **No starved mel filters.** Checked on the **built filterbank matrix** (`librosa.filters.mel` with the config values), not by closed-form arithmetic: (a) no empty filter (librosa's empty-filter warning is raised as an error); (b) every filter has at least 2 nonzero FFT bins; (c) the half-width of the lowest filter is at least the FFT bin spacing. The minimum nonzero-bin count and the margin go in the run log. The threshold is an **[ASSUMPTION]**: 2 bins is the least that lets a triangle interpolate between bins; a stricter 3-bin requirement is reported but not enforced. *Own calculation (HTK scale, fmin 50 Hz, fmax 8000), reproduced with a NumPy re-implementation of the librosa algorithm (`mel_check.py`; to be re-run with real librosa in notebook 02):*

| `n_fft` | `n_mels` | lowest-filter half-width | bin spacing | starved filters |
|---|---|---|---|---|
| 1024 | 64 | 28.8 Hz | 15.6 Hz | 0 |
| 1024 | 96 | 19.2 Hz | 15.6 Hz | 0 |
| 1024 | 128 | 14.4 Hz | 15.6 Hz | 5 |
| 512 | 40 | 46.2 Hz | 31.3 Hz | 0 |
| 512 | 64 | 28.8 Hz | 31.3 Hz | 3 |

   At fmax 8000 the chosen 64 mels at `n_fft` 1024 leave a margin of 1.84 bins; at the project's fmax 7000 the margin is **1.75** (half-width 27.4 Hz). Lowering `fmax` from 8000 to 7000 Hz changes the half-width by 1.4 Hz. Largest `n_mels` passing the rule (scan of 8–256): `n_fft` 512 → 59 at fmax 8000 and 56 at 7000 (the latter sits exactly at the limit, margin 1.00); 1024 → 117 / 111; 2048 → 235 / 223. With the stricter 3-nonzero-bin requirement: 47 / 44, 84 / 80, 170 / 161.
4. The config hash and the normalisation statistics are checked at bundle load (6b.7).
5. **Edge-case behaviour (checked in notebook 02):** all-zero input gives finite output (−120 dB floor) and sets a `silent` flag; NaN or inf raises in training and skips the window with a log line in the demo; more than 1 % of samples at |x| ≥ 0.999 sets a `clipped` flag and is not rejected; fewer than `n_fft` samples is rejected.

### 6b.7 Deployment parity, live capture, smoothing and latency

- **Bundle hash:** SHA-256 over canonical JSON (sorted keys) of every front-end parameter including derived values (`n_win`, `win_samples`, `F_MAX`, mode), plus the bytes of the float32 normalisation statistics. The demo refuses a bundle whose hash differs from its own front end. Package versions are recorded in the bundle but **not** hashed, so a harmless patch update does not block the demo; the golden test covers numerical drift instead.
- **Golden checks (cells in notebook 02; files in `data/processed/golden/`, copied into the model bundle by notebook 91 so the demo can repeat them at load):** two deterministic synthetic WAVs (seeded noise plus a chirp), one at 16 kHz and one at 48 kHz so the resampling path is covered, each with its expected log-mel saved. Pass when the maximum absolute difference is at most 1e-3 dB (tolerance to be reviewed after the first run). A third test checks that a frame crop equals the waveform crop (6b.1 step 6). A fourth check compares `librosa.filters.mel(htk=True, norm="slaney")` with the filterbank produced by the report's own code (`mel_filterbank` in the owner's `src/drone_audio/dsp.py`). **Done (librosa 1.0.1, NumPy 2.5.4, SciPy 1.18.1 in the audit sandbox):** at the `reference` settings (16 kHz, `n_fft` 1024, 64 mels, fmin 50, fmax 8000, HTK, constant area) the maximum absolute difference is 1.9e-9 and the maximum relative difference on non-zero weights is 1.2e-7; the first and last non-zero bin are identical in all 64 filters; row sums agree to 1.2e-7 (float32 rounding; fmax 7000 gives the same result). Rule for the notebook check: identical non-zero support and relative tolerance 1e-6 on the weights (**[ASSUMPTION]**, about 10× above the observed rounding). The report's `hann_periodic` also matches SciPy's periodic Hann (3e-8; a symmetric Hann would differ by 2.4e-3) and its radix-2 FFT matches `numpy.fft.rfft` (6e-16 relative). The reference matrix is saved in `data/processed/golden/`.
- **Live capture:** `sounddevice` input stream, one channel, at the device's default sample rate, float32, into a queue, then a ring buffer of `win_samples` after `soxr.ResampleStream`. Opening the device at 16 kHz is avoided because the operating system's own resampler is uncontrolled. On an overflow flag from the audio API: log it, empty the buffer, **create a new `ResampleStream`**, and suppress decisions until the buffer has refilled with `L` seconds of audio ("warming up"). There is no front-end state to reset because windows are computed independently. **[TO CHECK when coded]** the exact `sounddevice` flag name.
- **Decision cadence and smoothing:** a decision every 0.5 s on the latest window, majority vote over the last 3 decisions, threshold from validation. Worst-case time to confirm a change is about `L + 2 × 0.5 s` = 2.0 s at `L` = 1 s, plus inference time (own derivation: wait for the first window dominated by the new sound, then one more decision). Reported, not claimed. `N = 3` is fixed in advance and not tuned on test data.
- **Real-time factor, defined precisely:** because windows overlap by 50 %, each second of audio is processed twice. RTF = (front end + CNN time per decision) ÷ 0.5 s decision interval. Requirement: **RTF < 0.5** on the owner's laptop CPU (50 % headroom for jitter); measured once the CNN exists.

### 6b.8 What still needs measuring or checking (everything else in 6b is decided)

| # | Item | Where | Feeds |
|---|---|---|---|
| 1 | Native sample rate (**16 kHz in 11 sampled clips**), per-class duration distribution (**not measured**), file-name groups (**parsed from the file list, see section 6**) | notebook 00 | `sr`, `L` (6b.3), split leakage control |
| 2 | Resampler −1 dB point | one script, notebook 00 | final `F_MAX` (D015) |
| 3 | `librosa.pcen` signature defaults (`inspect.signature`), input-scaling convention, `zi` array shape; NumPy re-implementation vs real `librosa.pcen` on one random array; the ±12 dB acceptance test | when coded | PCEN condition, D036 |
| 4 | Normalisation-mode selection | notebook 30 part B, after the part A baseline | D014 / D026 |
| 5 | RTF and golden tolerance | demo / first test run | D029, D030 |
| 6 | Bounded sweep and stress evaluation (6b.10) | notebook 30 part B, stage 2 | D033, D034 |
| 7 | Filterbank equivalence with the report's own matrix: **done** (6b.7). Still to do: re-run the starvation scan (`mel_check.py`) with real librosa | notebook 02 | D021, D035 |

### 6b.9 What changed from the report and why

| Item | Report | This project | Why |
|---|---|---|---|
| Input length | fixed 128 frames (2.1 s) | `L` from the audit rule | avoids padding most of a short clip |
| `fmax` | 8000 Hz (Nyquist) | 7000 Hz interim, measured later | resampler transition band (6b.2) |
| Padding | dB floor, then normalised | waveform-level wrap, short clips rejected | padding at the floor becomes an extreme input value after normalisation and mimics the dataset's digital-silence clips |
| Normalisation | per-bin train statistics | four conditions, protocol in 6b.4, default unchanged | microphone gain differs between dataset and laptop |
| Mel normalisation | `constant_area`, `2 / (high − low)` | unchanged, as `norm="slaney"` with `htk=True` | same filters, library implementation |
| Implementation | handwritten FFT, mel filters, resampler, PCEN | `librosa`, `soundfile`, `soxr`, NumPy | library-first rule |
| Streaming | frame-level stateful `push` | independent windows with identical library calls | a detector needs window-level decisions; no custom streaming transform to maintain |
| Live resampling | offline only | `soxr.ResampleStream` | **[VERIFIED]** `soxr` provides `resample` and `ResampleStream` for real-time use (float32 and int16 among its dtypes) |
| dB floor | not specified here | `amin=1e-12` (−120 dB) | quantisation-noise margin under area normalisation (6b.2) |


### 6b.10 Flexibility and the bounded sweep

**Principle.** The report and the D8 feature catalog are starting points, not targets. Nothing in the front end is hard-coded: every value in 6b.2 is a config key, validated by 6b.6, resolved into derived values and hashed (D029). The CNN builder reads its input shape `(1, n_mels, n_win)` from the resolved config and must not hard-code 64 × 59. Two named presets: `reference` (the report's values: 16 kHz, 1024/256, 64 HTK area mels, fmax 8000, 128 frames, per-bin train statistics) for continuity checks, and `project` (the defaults in 6b.2). A run names its preset plus any overrides.

**Bounded sweep (notebook 30 part B, stage 2, after the mode is chosen in 6b.4).** One factor at a time around the default, same reference CNN, 3 seeds, validation only.

| Factor | Variants (default first) |
|---|---|
| `n_mels` | 64, 40, 96 |
| `n_fft` / `hop` (75 % overlap kept) | 1024/256, 512/128 (with 40 mels, a round value that passes rule 3 with a 1.40 margin at fmax 7000; 64 mels fails at `n_fft` 512 and the largest passing count is 56, at the limit; so this variant changes two things and is labelled so), 2048/512 |
| `mel_scale` | htk, slaney |

That is the default plus 5 variants (`mel_norm` is excluded while the selected mode is `global_stats`: area and peak give the same CNN input after per-bin standardisation, and the variant would also need a second full mel cache plus 3 CNN runs; if notebook 30 part B selects `per_window_mean` or `pcen`, `mel_norm` {area, peak} is added back as a variant because the equivalence does not hold there; D035). **Rule:** a variant replaces the default only if its score exceeds the default's by more than the default's seed std and by at least 0.01; otherwise the default stays. *Why:* it stops the sweep from chasing seed noise and keeps every choice explainable. **[ASSUMPTION]** The dataset is probably easy enough that most variants tie; a tie keeps the default and is itself a reportable result (these choices are not critical).

**Stress conditions (extend the score of 6b.4).** Validation audio is also rendered through a speaker–microphone proxy before the front end: 2nd-order Butterworth high-pass at 150 Hz and at 300 Hz, and low-pass at 5 kHz (`scipy.signal`, waveform level, validation only). *Why:* laptop speakers typically reproduce little at low frequencies **[ASSUMPTION, not measured on the owner's laptop]**, which could remove drone fundamentals in the demo (D011). The score becomes the mean over seeds of the minimum F1 over all eight conditions (five gains, three filters); each condition is also reported separately. One extra control: random smooth per-band gain curves added in the dB domain during training (mel-domain EQ augmentation, same hook as gain augmentation) versus none, 3 seeds, on the default only.

---

## 7. Models, evaluation and judging

### Model ladder (order of notebooks)
| # | Algorithm | Role in the project / how it is used |
|---|---|---|
| 10 | Linear regression | Regress the 0/1 label, threshold at 0.5. Fitted by least squares / normal equations (`sklearn.linear_model.LinearRegression`, same solution), with a one-feature simple regression as illustration next to the multiple regression on the 84 features **[ASSUMPTION]**. A deliberately naive baseline; shows why a classification-specific model is needed. |
| 11 | "ML regression" | Owner: the machine-learning version of classical linear regression. Same linear model as notebook 10, but fitted iteratively by gradient descent on the squared error (`sklearn.linear_model.SGDRegressor`, `loss="squared_error"`) on scaled features, with the regularisation strength and learning rate tuned on validation only; regress the 0/1 label, threshold at 0.5. **[ASSUMPTION]** this reading of the owner's description (D038). |
| 12 | Logistic regression | Proper linear classifier, interpretable coefficients. |
| 13 | k-NN | Distance-based; needs scaled features; sensitive to k and to dimensionality. k tuned on validation; any cross-validation is group-aware (see Tuning). |
| 14 | Decision tree | Owner-confirmed: a single decision tree (`sklearn.tree.DecisionTreeClassifier`); depth and leaf size tuned on validation only; interpretable splits, high variance, the single-tree base for the ensembles in 15 to 17. |
| 15 | Random forest | Bagged trees with random feature subsets; strong tabular baseline. |
| 16 | Bagging **[V7]** | Trees trained on bootstrap samples, every feature available at each split (`BaggingClassifier` with a decision-tree base); number of trees and sample fraction tuned on validation. The step between the single tree (14) and the random forest (15): it isolates what bootstrap averaging buys before random feature subsets are added. |
| 17 | AdaBoost **[V7]** | Boosting with decision stumps as weak learners (`AdaBoostClassifier`, depth-1 trees); number of rounds and learning rate tuned on validation; balanced sample weights. Contrasts sequential boosting (bias reduction) with parallel bagging (variance reduction). |
| 18 | SVM **[V7]** | Linear and RBF-kernel `SVC` on scaled features; C (and gamma for RBF) tuned on validation; ROC-AUC from `decision_function`; number of support vectors reported next to the metrics. |
| 20 | Perceptron and MLP | Fully connected network on tabular features; the course's perceptron and MLP are one notebook (D042). |
| 30 | CNN | On log-mel spectrograms. Final algorithm. **Our own architecture**, built from basic layers (convolution, normalisation, pooling, dense); no pretrained weights, no imported ready-made architecture. |

### Metrics (same for every model, defined once in `metrics.py`)
Accuracy, precision, recall, F1 (drone as positive class), ROC-AUC, confusion matrix, and **false-alarm rate** and **miss rate** reported explicitly. *Why:* accuracy alone hides the trade-off between missed drones and false alarms; ROC-AUC is threshold-independent.

### Tuning and variance
- Hyperparameters tuned on the **validation set** only, with a small, explicitly listed grid per model (the grid and the reason for it go in the notebook).
- Cross-validation, if a notebook uses it (the course exercises tune k and C that way), is group-aware (`StratifiedGroupKFold` on the frozen group IDs) and runs inside the training set; plain k-fold would put segments of one recording on both sides (section 3, limitation 2). Out-of-bag scores of bagging and the random forest are not used for tuning for the same reason: bootstrap samples draw clips, not groups. *Why:* segment leakage inflates any score computed across clips of one group. **[ASSUMPTION]**
- Class imbalance (section 6) is handled per model: `class_weight="balanced"` where the estimator has it (logistic regression, SVM, trees, random forest, the base tree of bagging), balanced sample weights where it does not (linear regression, SGD regression, AdaBoost), a positive-class weight in the loss for the MLP and CNN; k-NN has no weights, so the threshold-independent ROC-AUC is reported next to its metrics (`DATA_PREPARATION_v5.md`, step 7). The decision itself stays in notebook 01. **[ASSUMPTION]**
- Stochastic models (bagging, random forest, MLP, CNN) are run with **3 seeds** **[ASSUMPTION]**; report mean ± std so differences are not just seed noise.
- Report **training time and model size** next to accuracy; part of "judging" an algorithm is its cost.

### Verdict card (one JSON per model in `results/cards/`)
`model`, `representation`, `best_hyperparameters`, `val_metrics`, `seeds`, `train_time_s`, `strengths`, `weaknesses`, `failure_cases_noted`, `decision_log_ids`.
`90_comparison_and_final_test` loads all frozen models, scores the **test set once**, builds the comparison table and figures, and writes the final judgement per algorithm.

---

## 8. Notebook template (every model notebook, same order)

1. **Setup cell** — mount Drive, install requirements, load `config.yaml`, set seeds. *(identical in all notebooks)*
2. **Goal and hypothesis** — what do we expect from this algorithm on this data, and why?
3. **Load frozen data** — splits and features from `data/processed/`. Never re-split.
4. **Baseline run** — default hyperparameters.
5. **Tuning on validation** — listed grid, justified.
6. **Diagnostics** — learning/validation curve or equivalent, confusion matrix, a few misclassified examples.
7. **Freeze** — save model + validation results + verdict card draft. **No test-set scoring here.**
8. **Decision log entries** — appended to section 10 of this file.

---

## 9. Environment and conventions

- Google Colab; **[ASSUMPTION]** start on CPU runtime (all classical models) and switch to GPU only for notebook 30.
- Copy the dataset from Drive to Colab's local disk at the start of notebooks that read many small files, or read cached `.npz` files. *(Reading many small files directly from Drive is typically slow; unverified for this dataset, so measure once.)*
- Seed everywhere from `config.yaml`. Paths only from `config.py`; no hard-coded paths.
- Libraries **[ASSUMPTION]** (pin versions in `requirements.txt`):
  - data and plots: NumPy, pandas, matplotlib
  - audio: `soundfile` (WAV I/O), `soxr` (resampling), `librosa` (STFT, mel, MFCC, PCEN); live capture: `sounddevice` **[TO CHECK when coded]**
  - classical models, splits, metrics, tuning: scikit-learn
  - MLP and CNN: **PyTorch** (confirmed by the owner): standard `torch.nn` layers plus `torch.optim` optimizers, learning-rate schedulers and losses
  - training: **plain PyTorch, no wrapper library at all** (owner's rule; no Lightning, skorch, fastai or Ignite). Training uses `torch.utils.data.DataLoader`, `torch.optim`, schedulers and losses inside one shared training function (`src/dronedet/training.py`) used by both the MLP and the CNN **[ASSUMPTION: one shared function, see D020]**
  - `torchaudio` is **not** used: **[VERIFIED]** it moved into a maintenance phase, with its decoding/encoding APIs deprecated in 2.8 and removed in 2.9. Its transforms (e.g. MelSpectrogram, MFCC, Resample) remain, but `librosa` avoids the dependency.

### Rules for AI coding assistants
1. Read this file first. Do not change the task, split, metrics or folder structure without the owner's approval.
2. **Never read, score or tune on the test set** outside `90_comparison_and_final_test`.
3. Do not invent facts about the dataset; if something is not in section 3 or in the audit output, say it is unknown.
4. Every non-trivial choice (library, parameter, threshold, architecture size) must come with a one-line justification the owner can paste into section 10.
5. Put reusable code in `src/dronedet/`; keep notebooks thin.
6. If an instruction is ambiguous (e.g., which method "ML regression" refers to), ask instead of guessing.
7. The demo must import preprocessing from `src/dronedet/features.py`; never re-implement it in `demo/`.
8. **Library-first:** do not hand-write anything an established library already provides (FFT, mel filters, resampling, scalers, classical models, metrics). **CNN architecture:** design our own, built from standard framework layers and trained with library optimizers, schedulers and losses (no hand-coded convolution or optimizer; training is a plain PyTorch loop in the one shared training function, with no wrapper library such as Lightning, skorch, fastai or Ignite; test data is only touched in notebook 90); never download pretrained weights or import a ready-made architecture (ResNet, EfficientNet, VGG, audio-specific pretrained networks, AutoML/NAS). Every architectural choice (kernel sizes, channels, depth, pooling, dropout, normalisation) comes with a one-line justification for section 10.

---

## 9b. Laptop demo ("play a drone sound, get drone / no drone")

**What it is:** a separate inference program (`demo/run_demo.py`), not part of training. It does not run at the same time as the models; it runs *after* training, loads the frozen CNN, and reuses the exact preprocessing code from `src/dronedet/features.py`.

**Pipeline (same for both modes):** audio → cut into windows → same preprocessing as training (`features.py`) → CNN → score → threshold → "drone" / "no drone".

- **File mode:** reads a WAV file. Deterministic, good for testing and for the report.
- **Live mode:** reads the laptop microphone in short chunks. **[ASSUMPTION]** a Python audio-capture library such as `sounddevice`; to be confirmed when the demo is coded.

**Model bundle (exported by notebook 91 to `results/models/final/`):** CNN weights; preprocessing parameters (sample rate, clip length, mel settings); **normalisation statistics computed on the training set**; the **decision threshold chosen on the validation set**. *Why:* if the demo preprocesses even slightly differently from training, the model sees inputs it was never trained on and its output is meaningless.

**Windowing and smoothing:** window length = training clip length. A decision is shown after smoothing over the last N windows (for example a majority vote) **[ASSUMPTION]**; N is chosen by testing, not fixed in advance. *Why:* a single window can flicker between the two labels.

**Honest limits (state them in the report):**
1. The demo is an **illustration, not evidence**. The evidence is the test-set results from notebook 90. No demo number goes into the comparison table.
2. Playing a drone sound through laptop speakers into the laptop microphone adds speaker, room and microphone effects that the training data does not contain (the dataset is indoor and partly synthetic). The demo may behave worse than the test metrics suggest.
3. Demo audio must **not** be a training clip; use test-split clips or sounds from outside the dataset. Playing a training clip proves nothing.
4. Include negative checks in the demo: silence, speech, music, keyboard noise. A detector that only says "drone" is not a detector. **Silence must be real captured room noise, never digital zero** (a laptop microphone in a quiet room is never zero, while the training silence clips are exactly zero). As far as we know the training negatives contain no speech or music, so false alarms on those checks are possible and are reported as they are.

---

## 10. Decision log

Format: `D### — decision` · *why* · *alternatives considered* · *evidence/source* · *status (proposed / confirmed / revisited)*.

- **D001 — Modality: audio.** *Why:* small clean public datasets; one signal gives both tabular features and spectrograms; no hardware. *Alternatives:* images/video (larger datasets, more cleaning), RF (specialised data). *Evidence:* none yet beyond availability. *Status:* **confirmed by owner.**
- **D002 — Binary task (drone / no drone).** *Why:* it is the detection question and keeps scope within a course project. *Alternatives:* multiclass drone identification. *Status:* proposed.
- **D003 — Dataset: Al-Emadi DroneAudioDataset, binary folder.** *Why:* public, small (about 0.37 GB estimated), pre-organised by class, uniform 16 kHz mono 16-bit WAV (measured on 11 sampled clips; the README itself says nothing about the audio format), and file names that reveal the source recordings. *Alternatives:* DADS (about 7 GB by our arithmetic, mixed sources). *Evidence:* repo README; the repo's file list and 11 sampled clips; repo LICENSE; DADS dataset card. *Known costs:* two drone models only, 7.79 : 1 imbalance, 1,089 digital-silence clips, 5-segment recordings that force group splitting (section 3). *Status:* proposed; licence read from the repo; format, size, lengths and corruption measured by the exploration (`DATA_PREPARATION_v5.md`), to be reproduced by notebook 00.
- **D004 — Freeze split before modelling; test set scored once in notebook 90.** *Why:* fair, leak-free comparison. *Status:* proposed.
- **D005 — Two representations (tabular features for classical models and MLP; log-mel for CNN).** *Why:* section 6. *Status:* proposed.
- **D006 — MLP on tabular features before the CNN.** *Why:* separates the effect of the algorithm from the effect of the richer input. *Status:* proposed.
- **D007 — Metrics set including false-alarm and miss rates.** *Why:* unequal cost of the two error types. *Status:* proposed.
- **D008 — Code in `src/`, notebooks thin; Git for code, Drive for data.** *Why:* section 4 and 5. *Status:* proposed.
- **D009 — Laptop demo is a separate inference program that reuses `features.py`.** *Why:* identical preprocessing between training and demo; keeps training notebooks free of demo code. *Alternatives:* demo code written inside the CNN notebook (risk of drifting preprocessing). *Status:* proposed.
- **D010 — Export a model bundle (weights + preprocessing parameters + train-set normalisation statistics + validation-chosen threshold).** *Why:* section 9b. *Status:* proposed.
- **D011 — Demo results are illustrative only; evidence = test-set results from notebook 90.** *Why:* speaker-to-microphone domain shift and uncontrolled conditions make demo behaviour non-comparable to the test protocol. *Status:* proposed.
- **D012 — Reuse the existing front end's parameter values and design decisions as the `reference` preset, implemented with established libraries, with config-derived frame counts and runtime config validation.** *Why:* the parameters are already reasoned; fixed values (128 frames, fmax = Nyquist) do not fit unknown clip lengths; library code is maintained and tested. *Alternatives:* reuse the owner's from-scratch code (rejected: library-first rule); `torchaudio` (rejected: D017). *Revised:* the report's values are a starting preset, not a target (D033). *Status:* proposed.
- **D013 — Log-mel primary; PCEN and linear log-STFT as ablations; selection by the pre-registered rule in 6b.4 (D026) with a simplicity tie-break.** *Status:* proposed; details in D025/D026.
- **D014 — Level handling by selectable normalisation plus gain augmentation, chosen with a gain-robustness test over 0, ±6, ±12 dB (protocol in D026).** *Why:* laptop microphone gain differs from the dataset. *Status:* proposed.
- **D015 — fmax set from the measured resampler passband edge: largest multiple of 250 Hz below the −1 dB point, never above 8000; interim value lowered from 7500 to 7000 Hz.** *Why:* training audio is not resampled but demo audio is, so the band must stay below the transition band for both paths to match (6b.2). *Alternatives:* 8000 Hz (report), 7500 Hz. *Status:* proposed, pending measurement.
- **D016 — Live capture at the device's native rate, resampled with `soxr.ResampleStream`; an audio overflow triggers a log entry, an empty buffer, a new `ResampleStream` and a warm-up period of `L` seconds without decisions (6b.7).** *Status:* proposed, pending the demo.
- **D017 — Library-first for everything except the CNN; `librosa` + `soundfile` + `soxr` for audio, scikit-learn for classical models; `torchaudio` not used.** *Why:* owner's rule; **[VERIFIED]** torchaudio is in maintenance, with I/O APIs removed in 2.9. *Alternatives:* hand-written DSP (rejected), `torchaudio` transforms. *Status:* confirmed by owner for the rule, proposed for the specific libraries.
- **D018 — The CNN is our own architecture built from basic framework layers; no pretrained weights or imported architectures.** *Why:* it is the end goal of the project. *Status:* confirmed by owner, including that standard library layers and training tools are used.
- **D019 — PyTorch for the MLP and CNN.** *Why:* widely used and well supported by the coding assistants. *Alternatives:* Keras/TensorFlow. *Status:* confirmed by owner.
- **D020 — Plain PyTorch for training; no wrapper library at all.** *Why:* owner's rule. The MLP and the CNN are trained by one shared function in `src/dronedet/training.py`, so both get the same loop, seeding, early stopping on validation loss and checkpointing, which keeps their comparison fair. Early stopping, checkpointing (`torch.save`) and seeding (`torch.manual_seed`) are a few lines of our own because no wrapper provides them. The test set is not touched here (notebook 90 only). *Alternatives:* PyTorch Lightning, skorch, fastai, Ignite (all rejected by the owner). *Status:* wrapper decision confirmed by owner; the single shared function is an **[ASSUMPTION]**.

- **D021 — STFT and mel parameters: `n_fft` 1024, `hop` 256, Hann, `center=False`, 64 mel bins, `fmin` 50 Hz, `htk=True`, `norm="slaney"` (constant area).** *Why:* report values, including its `constant_area` mel normalisation (owner-confirmed); 15.6 Hz resolution resolves rotor harmonics; the 64-bin filterbank passes the starvation rule with a 1.75-bin margin at fmax 7000 (1.84 at 8000; reproduced with a NumPy re-implementation of the librosa algorithm, 6b.6, D035). *Alternatives:* 96 mels (passes, but the margin shrinks to 1.2 bins and the CNN input grows), 128 mels (5 starved filters, rejected by rule 3), `n_fft` 512 (64 mels fails), Slaney mel scale (close to HTK at 64 bands, tested in the sweep), peak-1 triangles `norm=None` (my first guess, wrong: contradicts the report and the D8 catalog, and adds a +9.6 dB slope for flat noise). Area normalisation is kept because it is better on merit as well as consistent with both earlier setups: flat noise gives a flat spectrogram (−0.2 dB band to band) and the discrete filter area stays within 4 % of 1 at 64 bands (up to 12 % at 96). For the CNN under `global_stats` the choice has no effect at all, provided floor-clamped frames are excluded from the statistics (area and peak weights differ by a fixed per-band constant that the per-bin mean removes; D035), so the merit argument applies to the tabular descriptors. *Evidence:* own arithmetic, run. *Status:* proposed; HTK scale confirmed by the owner as the scale of the report, of D8 and of the project; filterbank equivalence test in 6b.7; every key is configurable and enters the sweep (D033).
- **D022 — dB conversion with a fixed floor: `ref=1.0`, `amin=1e-12`, `top_db=None`.** *Why:* `top_db` clips relative to each call's own peak, so the output would depend on window level and content, breaking gain handling and demo parity. With area normalisation, 16-bit quantisation noise sits at about −87 dB per band (simulated), so the floor must be well below it: 1e-12 (−120 dB) leaves 33 dB, 1e-10 only 13 dB. *Alternatives:* `top_db=80` (librosa default), `amin=1e-10` (first draft), natural log. *Status:* proposed.
- **D023 — Window length `L` from the audit: largest of {0.5, 1, 1.5, 2} s such that at least 95 % of the clips of each class have at least `win_samples(L)` samples (the usable window, not `L·sr`); stop and ask if 0.5 s fails.** *Why:* deterministic and reportable; 95 % limits padding; counting against `win_samples` avoids a spurious cliff for clips that are a few samples shorter than the nominal length; 2 s cap from the report's 2.1 s and from latency. *Alternatives:* fixed 128 frames (report), median clip length. *Measured outcome [V6]:* L = 0.5 s (6b.3). *Status:* **confirmed by owner (V8): `L` = 0.5 s**; the rule's result is measured by the exploration and awaits reproduction in notebook 00.
- **D024 — Clip handling: crop (random in training, centred in validation/test and tabular), wrap-pad clips with between `win_samples/2` and `win_samples` samples, reject shorter ones; cache full-clip mel power and crop at frame level.** *Why:* zero padding would imitate the dataset's digital-silence clips (1,089 exact-zero clips in 10 distinct contents, not one); wrap keeps the clip's own statistics; frame-level cropping is exact with `center=False` and avoids recomputing the STFT every epoch. *Alternatives:* zero padding, padding-mask channel (extra channel and a second code path), padding at the dB floor (report). *Status:* proposed.
- **D025 — Normalisation conditions: `global_stats` (default, per-bin train statistics), `per_window_mean` (scalar window mean), `pcen`, `logstft` (ablation).** *Why:* the scalar mean keeps the stationary spectral shape that identifies a drone; a per-bin mean would remove it. Exact gain invariance of the scalar mean checked numerically. PCEN is fully parametrised, not run on library defaults (D036). *Alternatives:* per-bin per-window mean (rejected, see why), no normalisation. *Status:* proposed.
- **D026 — Gain augmentation of ±12 dB uniform, applied to the cached mel power in training for all conditions; selection by pre-registered rule (mean over seeds of the minimum validation F1 over gains {−12, −6, 0, +6, +12}, 0 dB guard of 0.02, simplicity tie-break), run in notebook 30 part B with the notebook-30 baseline CNN; ±18 dB reported as extrapolation; a no-augmentation control for `global_stats`.** *Why:* laptop microphone gain differs from the dataset; selecting on the worst case rewards robustness rather than the best-case score; the rule is fixed before any result. *Alternatives:* choose by 0 dB F1 only, waveform-domain augmentation (recomputes the STFT, no benefit). *Status:* proposed; execution pending the CNN baseline.
- **D027 — Tabular feature set of 84 level-free features: MFCC c1–c19 (mean, std, min, max) and mel-domain centroid, bandwidth, rolloff, flatness (mean, std).** *Why:* every feature is unchanged by a gain, so the classical models cannot learn "loud means drone" from the dataset's mixing; all come from the one cached mel window. *Alternatives:* include c0 and RMS (level shortcut), zero-crossing rate (needs the waveform, a second cache), linear-STFT descriptors. *Status:* proposed; excluded features available as a documented ablation.
- **D028 — Edge cases: all-zero input is finite and flagged `silent`; NaN/inf raises in training and skips with a log line in the demo; more than 1 % of samples at |x| ≥ 0.999 flags `clipped` without rejecting; fewer than `n_fft` samples is rejected.** *Why:* deterministic behaviour that tests can assert. *Measured [V6]:* all 1,089 exact-zero clips (10 distinct contents) are caught by this flag. Near-silent clips that are not exactly zero are not: the exploration found 57 `unknown` clips and 2 drone-labelled clips below −80 dBFS (among them `2-102581-A-291.wav`, 99.88 % zeros); they are flagged `near_silent` and excluded from the statistics, D041 (for the project's front end the 16-bit rounding-noise floor is about −87 dB per mel band, measured with librosa 1.0.1; the threshold must be re-derived if `n_fft` or the mel normalisation changes). *Status:* proposed.
- **D029 — Parity: SHA-256 over canonical JSON of all front-end parameters (including derived values and every PCEN parameter) plus the bytes of the normalisation statistics; library versions recorded but not hashed; golden WAVs at 16 kHz and 48 kHz with saved expected log-mels; tolerance 1e-3 dB; a frame-crop equivalence test.** *Why:* a patch update must not block the demo, while numerical drift must fail a test. *Alternatives:* hashing package versions (too brittle). *Status:* proposed; tolerance to be reviewed after the first run.
- **D030 — Decision every 0.5 s, majority over the last 3 decisions, `N = 3` fixed in advance; RTF defined as (front end + CNN time per decision) ÷ 0.5 s and required to be below 0.5.** *Why:* overlapping windows mean each second of audio is processed twice, so "processing time ÷ audio time" would understate the load; 50 % headroom absorbs jitter. Worst-case confirmation time about `L + 1.0 s` (2.0 s at `L` = 1 s) before inference time. *Status:* proposed; measured later.
- **D031 — Dataset licence record (read from the repo's own `LICENSE.md`, 2026-11-21).** *Academic Research and Educational Use Licence* (copyright 2019 to 2026, Sara Al Emadi). §1: use is allowed solely for educational, teaching and non-commercial academic research, including benchmarking, publishing papers and processing the data as needed. §2: cite the IWCMC 2019 paper and acknowledge third-party material. §3: the dataset must not be used for military, defence, intelligence, security or weapons applications, including military or defence drone detection, identification, tracking or surveillance systems, and law-enforcement or security uses that are part of a weapons, military or defence capability. §4: no commercial use without written permission. §5: no redistribution of the dataset beyond what the permitted research needs, with the licence and attribution kept. §6: third-party material inside it (ESC-50, Speech Commands) keeps its own licences; ESC-50 is reported as CC-BY-NC 3.0 by a third-party model card (not re-checked). *Consequence adopted:* this is a course project and fits the educational grant; the report and the demo describe it as an academic exercise and **not** as a security or defence system; the data and every cache derived from it stay on private Drive and are not redistributed; cite the paper and the repo. *Status:* verified against the repo; the owner accepts the security-use clause for this course project (confirmed).
- **D032 — Structure additions for the front end: `tests/` (`test_frontend.py`, `golden/`) and `notebooks/31_frontend_ablation.ipynb`.** *Why proposed:* the parity and edge-case tests needed a home, and the normalisation selection needed a notebook that runs after the notebook-30 baseline. *Status:* **rejected by owner**; replaced by D037.

- **D033 — Flexibility: every front-end value is a config key, validated and hashed; the CNN reads its input shape from the config; presets `reference` (the report) and `project` (defaults); bounded one-factor-at-a-time sweep in notebook 30 part B (n_mels, n_fft/hop, mel scale; mel norm excluded while the mode is `global_stats`, D035) with the replace-only-if-better-than-seed-noise rule.** *Why:* the old setups are not assumed to be optimal; a sweep gives evidence without chasing noise, and the default stays when variants tie. *Alternatives:* freeze the report's values without testing, full grid search (cost, multiple-comparison risk). *Status:* proposed; execution pending the CNN baseline.
- **D034 — Stress evaluation and optional EQ augmentation: validation audio also high-passed at 150 and 300 Hz and low-passed at 5 kHz; score is the worst F1 over eight conditions; a control with mel-domain EQ augmentation.** *Why:* the demo plays sound through laptop speakers and microphone, which probably removes low frequencies (D011); testing it before the demo makes the front-end choice evidence-based. *Alternatives:* gain only (original D014), recording a real speaker–microphone sweep on the owner's laptop. *Status:* proposed, **[ASSUMPTION]** on the speaker response.

- **D035 — Mel filterbank validated on the built matrix; `mel_norm` removed from the sweep under `global_stats`.** *Why (filterbank):* rule 3 was checked by closed-form arithmetic, which is fragile at the boundary (at `n_fft` 512 and fmax 7000, 56 mels sits at margin 1.00) and had produced a wrong claim ("40 is the largest passing count"; the true maximum is 59 at fmax 8000 and 56 at 7000). Config validation now builds `librosa.filters.mel` and checks: no empty filter, at least 2 nonzero FFT bins per filter, lowest-filter half-width ≥ bin spacing; the minimum nonzero-bin count and the margin are logged. *Why (`mel_norm`):* `W_area = diag(e)·W_peak` with `e_k = 2/(high_k − low_k)` (zero deviation on the matrices), so `S_area = e ⊙ S_peak` for any signal and the dB spectrograms differ by a signal-independent offset `d_k = 10·log10(e_k)` per band (−24.3 to −14.4 dB at 64 mels, fmax 7000). `global_stats` subtracts `μ_k`, which absorbs `d_k`, and `σ_k` is unchanged, so the CNN input is identical; gain augmentation adds the same `g` to both and does not break this. **Conditions for exactness:** each configuration computes its own statistics from its own cache, and no cell reaches the −120 dB floor. *Synthetic check (400 clips, 5 % digital silence, NumPy re-implementations, not librosa):* without floor-clamped cells the maximum difference is 4e-8 std (float32); with silence clips in the statistics it is up to 0.47 std even on non-silent frames, which is why `global_stats` now excludes `silent`-flagged clips (6b.4). **Where the equivalence does not hold:** `per_window_mean` leaves a fixed per-band tilt of ±0.6 std plus a 6 % rescale (differences up to 0.78 std; exact once that offset and rescale are removed), so it is a different input; `pcen` differs by about `c_k^(1−α)` ≈ 1.05 at α = 0.98 (correlation 0.999 at a sensible input scale, 0.59 when the input is far below `eps`); tabular MFCC statistics are identical after scaling, but spectral centroid and bandwidth are not (correlation 0.97–0.99; rolloff and flatness untested). Hence: `mel_norm` is excluded from the sweep only while the selected mode is `global_stats`; area normalisation stays the default for the tabular descriptors. *Alternatives:* keep the closed-form check; require 3 nonzero bins (stricter: at most 47 mels at `n_fft` 512, and the 1024/64 default has exactly 3); keep `mel_norm` in the sweep unconditionally (a guaranteed tie under `global_stats`, plus a second cache and 3 CNN runs). *Evidence:* `mel_check.py` and `norm_equiv.py` (NumPy re-implementations of librosa's filterbank and `pcen`; librosa was not installed in the audit sandbox); to be re-run with real librosa in notebook 02. *Status:* proposed. *Update:* the filterbank of the report's own code (`dsp.py`) was compared with real librosa at the `reference` settings and matches to float32 rounding (6b.7). *Correction:* the synthetic check mixed in 5 % digital silence; the real folder has 9.3 % (10.5 % of the unknown class), so the 0.47 std shift is probably an underestimate for the real data and should be recomputed on the real cache.

- **D036 — PCEN configuration: explicit parameters, scaled input, window-mean initialisation, acceptance test.** *Why:* `librosa.pcen` converts `time_constant` to frames using `sr` and `hop_length`; its defaults (22 050 Hz, 512), as far as I remember the signature **[TO CHECK]**, give 17.2 frames = 276 ms at this project's 16 ms hop instead of the intended 400 ms (31 % shorter). That effect on gain robustness is small (under 0.01 std). The larger issues, measured on synthetic data with a NumPy re-implementation of `pcen`: (a) in raw cache units 92.5 % of mel cells are below `eps` = 1e-6, so PCEN is not adaptive; the output changes by 0.31 / 0.80 std at −12 / +12 dB for power input, against 0.15 / 0.15 std for `sqrt(power) × 2^31` (and for power × 2^62), while dB + `global_stats` changes by about 0.84 std at ±12 dB and `per_window_mean` by 0; (b) initialising the smoother from frame 0 of each window differs from a warmed-up PCEN by 0.27 std and a +20 dB click in frame 0 changes the whole window by about 4 std, whereas initialising from the window mean gives 0.05 std and about 0.9 std (parity is kept because the demo sees the same window); (c) PCEN parameters were not in the config hash (D029). *Decision:* every PCEN value is an explicit config key and is hashed; input is `sqrt(cached power) × 2^31`; the smoother starts from the window mean per band; the condition runs only if the ±12 dB output change on train windows is ≤ 0.2 std (threshold chosen above the 0.15 measured, an **[ASSUMPTION]**). *Alternatives:* librosa defaults; raw-unit input with a re-tuned `eps`; first-frame initialisation (librosa default); no PCEN. *Evidence:* `pcen_check.py` (synthetic spectra; to be re-run with real librosa and real audio). *Status:* proposed.
- **D037 — No new folders: no `tests/` and no notebook 31 (owner declined D032).** Front-end checks (edge cases, golden WAVs, frame-crop equivalence, filterbank equivalence) run as cells in notebook 02; golden files live in `data/processed/golden/` and are copied into the model bundle by notebook 91 so the demo repeats the parity check at load. Front-end selection, gain-augmentation selection, the bounded sweep and the stress evaluation (D014, D026, D033, D034) become **part B of notebook 30**, after the part A baseline and before the final CNN tuning. *Why:* rule 1 forbids structure changes without owner approval, and the owner declined them. *Alternatives:* dropping the experiments, a separate notebook (declined). *Status:* proposed. **[ASSUMPTION]** the owner declined the extra files, not the experiments; if the experiments should also go, D014, D026, D033 and D034 are cut back to their defaults.
- **D038 — Course methods fixed by the owner: notebook 14 is a single decision tree; notebook 11 ("ML regression") is linear regression trained the machine-learning way.** *Why:* owner answers to the former open question 1. *Reading for notebook 11 (**[ASSUMPTION]**):* the same linear model as notebook 10, but solved by iterative gradient descent on the squared error instead of the closed-form least-squares fit, so that optimisation, feature scaling and regularisation strength are the new elements; the two will probably score almost the same, which is itself a reportable result. *Alternatives:* ridge/lasso/elastic net with a closed-form or coordinate-descent solver (use these if the course notes define "ML regression" as regularised regression; the notebook changes by one line). Boosted trees for notebook 14 (not chosen). *Status:* tree confirmed by owner; notebook 11 reading proposed. **Revisited by D042 [V7]:** the owner's course list confirms that notebook 11 is the gradient-descent version of linear regression; boosted trees are added as notebook 17 and no longer sit among the alternatives.
- **D039 — Group-aware split and exact-duplicate silence.** (a) Split by group (keys in section 6); (b) the 1,089 exact-zero clips (10 distinct contents) are reduced to one clip per distinct content: `silence300` (24 samples) is already rejected by D024, so **9 representatives** remain (first file name in sorted order, preferring `silence###`), in train only, each with its group pinned to train (**8 pinned groups**, because the representatives of the 330-copy and 308-copy contents share ESC group `1-100032`) [V6]; (c) consequently the validation and test sets contain no digital silence, and silence behaviour is checked separately by the demo's real-room-noise negative check (illustrative, D011); (d) the `silent` flag (D028) still excludes zero clips from the `global_stats` statistics. *Why:* identical copies in several splits would be counted as correct test samples and inflate scores; copies add no information but teach "digital zero means no drone", which the dataset cannot be corrected for. *Alternatives:* keep all copies in one split (the other splits then have no silence, and train is dominated by identical clips); keep all copies spread over splits (identity leakage, rejected); drop silence entirely (removes the only digital-silence examples the CNN sees). *Status:* **confirmed by owner (V8)**; counts measured (`DATA_PREPARATION_v5.md`, step 3.3).
- **D040 — Grouping of `extra_membo_D2` and `B_S2_D1` [V6; decided V8].** *Why:* measured in `DATA_PREPARATION_v5.md` 2.9, 2.10 and step 5. `extra_membo_D2` (56 clips, one group, median RMS −42.5 dBFS, 87.5 % of clips with rolloff at or below 100 Hz) lands in validation in 60 of 60 seeds of the two-stage split, making up 29 % of validation's drone clips and none of test's; the 50 Hz `fmin` of the front end removes most of its power. Neighbouring `B_S2_D1` recordings correlate at about 0.55 (shuffled baseline 0.00 ± 0.13), so one group per recording may leak conditions. *Alternatives:* for `extra_membo_D2`: pin to train and report as excluded from evaluation (assistant's recommendation), keep as one group (validation), force into test, drop; for `B_S2_D1`: one group per recording, blocks of about 5 consecutive recordings (assistant's suggestion, fixed before any model result), drop from evaluation. *Decision [V8; the owner accepted the assistant's recommendations]:* `extra_membo_D2` is pinned to train and reported as not evaluated. The `B_S2_D1` recordings 67 to 133 are merged into 13 blocks of consecutive recordings (12 blocks of 5: 67 to 71, ..., 122 to 126; and 127 to 133 as the last block, 7 recordings, 31 clips), group keys `B_S2_D1_blk067` ... `_blk127`, fixed before any model result; the other families keep one group per recording. *Why pin:* every hyperparameter is tuned on validation and test never contained the family, so keeping it in validation would tune the models on a family that test does not have; forcing it into test would make one atypical group roughly a quarter to 30 % of test's drone clips; dropping it hides data. *Why blocks:* the measured neighbour correlation is 0.55 for `B_S2_D1` against at most 0.26 for the `mixed` families. *Costs:* recall for `extra_membo_D2` is never reported; fewer, larger groups widen the bootstrap intervals; the recording-order assumption is not established. *Expected effect (own estimate):* about 180 drone clips each in validation and test. *Status:* **confirmed by owner.**
- **D041 — `near_silent` clips are excluded from the normalisation statistics [V6].** *Why:* D028 excludes only exactly-zero clips, but `2-102581-A-291.wav` (99.88 % zeros, 1 LSB peak, RMS −119.7 dBFS) and 56 other `unknown` clips are below −80 dBFS without being exactly zero; their frames sit at or near the −120 dB floor and would distort the per-bin mean and std (the effect shown for zero clips in D035). *Alternatives:* exclude only exact zeros (D028 as written); exclude clips below a higher level threshold. *Status:* **confirmed by owner (V8)**, including that the two near-silent drone-labelled clips stay in the manifest, flagged, neither relabelled nor excluded (labels come only from the folder; step 9 of `DATA_PREPARATION_v5.md` reports results with and without flagged clips).

*(Append new entries below. Keep the numbering.)*



- **D042 — Model ladder extended to the owner's course list [V7].** Added: notebook 16 bagging, 17 AdaBoost, 18 SVM. Kept: random forest (15). Perceptron and MLP are one notebook (20), per the owner ("put MLP instead"). Notebook 10 states that it is the least-squares / normal-equation fit; notebook 11 is the gradient-descent fit. Existing notebook numbers do not change. *Why:* the owner's list is the project's list of algorithms to try, and the project rule is to try all of them and judge each; keeping the numbers keeps D006, D038 and 6b.4 valid. *Alternatives:* renumber in course order (tidier, but rewrites references); drop random forest (not named in the list; owner said to include all); a separate perceptron notebook (owner chose MLP). *Per-notebook choices (**[ASSUMPTION]**):* see the ladder rows in section 7. *Consequences:* group-aware cross-validation, no out-of-bag tuning, and per-model imbalance handling are written into section 7 (Tuning). *Status:* list confirmed by owner; reading of "include them all" as keeping the random forest is an **[ASSUMPTION]**; per-notebook choices proposed.
- **D043 — Reproducibility record written by notebook 00 [V8].** `data/interim/environment_log.json` holds the Python and library versions (`librosa`, `soundfile`, `soxr`, `scikit-learn`, `torch`, `numpy`), the SHA-256 of the downloaded zip, the download date and, if the owner knows it, the repository commit. *Why:* provenance and versions cannot be established from the files, and recording them costs one cell. Versions are recorded, not hashed (D029). *Alternatives:* ask the owner for each value (slower, error-prone). *Status:* proposed, **[ASSUMPTION]**.

- **D044 — Validation threshold and selection score.** Choose the threshold that maximises validation F1 with drone as positive; ties go to the larger threshold. Score configurations by validation F1 at their own optimal threshold and report ROC-AUC alongside it. Also report miss rate at 1% and 5% false-alarm rates. For robustness conditions, choose the threshold once at 0 dB and hold it. *Why:* one validation rule supports all models while fixed thresholds expose robustness failures. *Alternatives:* accuracy, Youden's J, F-beta or re-optimising each condition. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.1 and 5.4. *Status:* confirmed by owner 2026-11-22.
- **D045 — Tuning protocol for notebooks 10 to 20.** Use explicit, bounded grids; tune on validation only; do not use cross-validation or out-of-bag scores; apply the replace-only-if-better rule; run three seeds for stochastic fits and use sample standard deviation (`ddof=1`); name each default; use one-factor-at-a-time searches when grids exceed about 30 configurations; combine accepted changes only under the specified combination rule; use a 0.01 margin alone for deterministic models. *Why:* small validation group counts make broad searches prone to overfitting. *Alternatives:* full grid search or default settings only. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.2 and 5.4. The earlier D045 model values are reference targets, not results. *Status:* confirmed by owner 2026-11-22.
- **D046 — Perceptron and MLP implementations.** Use `sklearn.linear_model.Perceptron` for the perceptron and PyTorch through the shared plain training loop for the MLP. *Why:* use the established library implementation for the perceptron and the owner-confirmed framework for the MLP. *Alternatives:* implement the perceptron in PyTorch or use `MLPClassifier`. *Evidence:* owner-confirmed in `Implementation_plan_the_detection_algorithms_v3.md`, section 2.1. *Status:* confirmed by owner 2026-11-22.
- **D047 — Statistics eligibility for tabular scaling.** Fit the tabular scaler on train rows where the stored `stats_eligible` flag is true; this excludes the union of `silent`, `near_silent` and `crop_all_zero` (88 train clips). Keep excluded clips in training. *Why:* use the recorded preparation eligibility mask and avoid reconstructing a potentially different exclusion set. *Alternatives:* fit on all train rows or recompute exclusions from other flags. *Evidence:* `DATA_PREPARATION_v5.md`, implementation report v5, and owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.1 and 5.3. *Status:* confirmed by owner 2026-11-22.
- **D048 — CNN baseline architecture A0.** Use three 3×3 convolution, batch-normalisation and ReLU blocks with channel widths 16-32-64; 2×2 max-pooling after the first two blocks; global average pooling; dropout 0.3; and one linear logit. Read the input shape from the resolved configuration. *Why:* a compact architecture with a shape-independent head suited to CPU training. *Alternatives:* flatten head, other widths or depths, and imported architectures. *Evidence:* 23,585 parameters at input `(1, 64, 28)` by architecture arithmetic in the plan. *Status:* confirmed by owner 2026-11-22.
- **D049 — MLP and CNN training recipe.** Train on CPU with `BCEWithLogitsLoss(pos_weight)`, AdamW, learning rate 1e-3, weight decay 1e-2, batch sizes 128 for MLP and 64 for CNN, `ReduceLROnPlateau` (factor 0.5, patience 3), early stopping (patience 8, at most 100 epochs, restore best state), and three seeds. CNN training uses one fresh random window per clip per epoch and training-only gain augmentation. *Why:* a shared, explicit recipe makes the neural comparison reproducible. *Alternatives:* other optimisers, schedulers, samplers or numerical modes. *Evidence:* owner-confirmed CPU recipe in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.2 and 5.5. *Status:* confirmed by owner 2026-11-22.
- **D050 — CNN tuning and designated seed.** Tune one factor at a time around A0 using the stated grids and selection rule. Designate the validation-best seed before any test score; score every seed in notebook 90. *Why:* avoid selecting a seed from test results and limit validation overfitting. *Alternatives:* full grid search, random search or seed ensembling. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.1, 9.10 and 9.11. *Status:* confirmed by owner 2026-11-22.
- **D051 — Group bootstrap evaluation.** Use 1,000 group-bootstrap draws with percentile 95% intervals; share draws for paired model comparisons and report group counts. *Why:* clips from one recording are not independent, and shared draws measure paired differences. *Alternatives:* clip bootstrap or unpaired model intervals. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 5.4 and 10. *Status:* confirmed by owner 2026-11-22.
- **D052 — Demo decision-layer clarifications.** Not applied in this job; decision step, vote length and demo behaviour are out of scope. *Why:* keep the algorithm work within its approved scope. *Alternatives:* change cadence or smoothing settings. *Evidence:* owner decision in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.1 and 2.3. *Status:* not applied by owner decision 2026-11-22.
- **D053 — Frame-level centre of the fixed window.** Use frame offset `(n_frames - n_win) // 2` for fixed tabular, validation and test windows; draw training offsets uniformly over valid frames. *Why:* cached windows must align to the stored frame grid. *Alternatives:* sample-level centring requiring a second front-end path. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.1 and 5.3. *Status:* confirmed by owner 2026-11-22.
- **D054 — Notebook 30 part B bookkeeping.** Apply the eight-condition score from B1 onward with the threshold fixed at 0 dB; test a combination when at least two B2 variants are accepted; reuse the stage-C default's existing seeds. B1 runs `global_stats`, `per_window_mean`, and PCEN only if its acceptance check passes; `logstft` is recorded as not run. B1 control and B3 are reported but never adopted into the freeze. *Why:* select modes and variants under the registered conditions and avoid freezing an untested combination. *Alternatives:* filters on only the default or freezing accepted changes without testing the combination. *Evidence:* owner-confirmed protocol in `Implementation_plan_the_detection_algorithms_v3.md`, sections 2.2, 9.5 and 9.7. *Status:* confirmed by owner 2026-11-22.
- **D055 — Separate mel-cache content hash.** Add `cache_content_hash(config)` over only the settings read by the mel-power cache builder: sample rate, channel, FFT/window/hop, window function and centring, mel count/range/scale/norm, power, `L` and resampler quality. Exclude mode and PCEN parameters; retain the existing full `params_hash()` / `config_hash()` for front-end and model lineage identity. *Why:* mode and PCEN transform cached power after it is stored, so changing them must not invalidate the power cache. *Alternatives:* use the full lineage hash for caches, which would invalidate identical mel-power data on mode-only changes. *Evidence:* owner amendment 1 to `Implementation_plan_the_detection_algorithms_v3.md`, section 2.2; implementation check reports cache hash `6d435113ecb663cc08145c56b34da6def8fda2283acbd2855da02dacfe7360d0`. *Status:* confirmed by owner 2026-11-22.

- **D056 — Read-only Drive upload audit.** Add an optional preflight cell to notebook 10 that lists missing expected Drive bundle paths and checks manifest-listed raw clip paths, with test paths checked for existence only. *Why:* identify incomplete folder uploads before C0.5–C0.8 without opening audio or inspecting test labels. *Alternative:* manually inspect more than 11,000 Drive items. *Status:* implemented at owner request 2026-11-22.
- **D057 — Local model execution.** Run the detection algorithms and evaluation locally on the owner’s Windows machine using CPU execution, and leave notebooks unchanged. This supersedes Q-O’s Colab execution arrangement for this task; the frozen splits, validation-only selection, model protocols, test guard, and one-pass final test evaluation remain in force. *Why:* the owner changed the execution plan to run all models locally. *Alternative:* use the Colab notebook workflow. *Evidence:* owner instruction 2026-11-22. *Status:* confirmed by owner 2026-11-22.
- **D058 — CNN B1 front-end mode selection.** Retain `global_stats` as the CNN front-end mode. The three-seed mean validation F1 at 0 dB was 0.89512 for `global_stats`, 0.91010 for `per_window_mean`, and 0.82635 for PCEN. PCEN failed the registered 0 dB guard; `per_window_mean` passed that guard but its eight-condition mean worst-case F1 was 0.01769 versus 0.30518 for `global_stats`, so it did not tie within the registered margin. *Why:* apply D054 and the 9.7 `select_mode()` rule without changing criteria. *Evidence:* local B1 report `results/tables/run_report_30_local_b1.json`, generated by code commit `a1433de`; the three `global_stats` seeds are reused from Stage A. Test data was not read. *Status:* selected by the registered rule 2026-11-23.

---

## 11. Open questions for the owner

1. *(Resolved [V8]: `L` = 0.5 s, D023.)*
2. *(Resolved [V8]: D039 approved.)*
3. *(Resolved [V7]: see Resolved, D038 and D042.)*
4. *(Resolved [V8]: `extra_membo_D2` pinned to train, not evaluated, D040.)*
5. **Meaning of `mixed`, `S`, `D`, `Membo_k` (non-blocking, V8):** not established by the files, and a quick web search found nothing. If the repository README or the paper explains them, a session-level split may become possible; the `B_S2_D1` block rule (D040) stands in the meantime.
6. *(Resolved [V8]: near-silent rule approved and the two drone-labelled clips kept flagged, D041.)*

**Resolved**

- Modality: audio, confirmed by the owner (D001).
- Report format, deadline and grading rubric: dropped by the owner.
- Framework for the CNN: closed. Nothing is mandated or forbidden; PyTorch (D019), own architecture (D018) and a plain training loop (D020) are already owner-confirmed.
- Dataset licence: the owner accepts the security-use clause for this course project, and the terms were checked against the repo's own LICENSE file (D031).
- Folder additions `tests/` and notebook 31: declined (D032 rejected); the work is relocated (D037).
- Filterbank matrix from the report's code: provided as `dsp.py`, compared with librosa, matches (6b.7, D035).
- Mel scale of the D8 feature catalog: HTK, the same as the report and the project (6b.2, D021).
- Course methods [V7]: the owner's list is linear regression (normal equations and gradient descent), logistic regression, k-NN, SVM, decision trees, bagging, AdaBoost, perceptron/MLP, CNN; "ML regression" is the gradient-descent version of linear regression; random forest is kept; perceptron and MLP are one notebook (D038, D042).
- Exploration of the dataset: Luna's run, audited against the raw files and six owner WAV clips (`DATA_PREPARATION_v5.md`, 4.3) [V6].
- Split blockers [V8]: `L` = 0.5 s (D023); D039 approved; `extra_membo_D2` pinned to train and not evaluated and `B_S2_D1` in 13 blocks (D040); near-silent rule approved, the two drone-labelled near-silent clips kept flagged (D041); reproducibility log by notebook 00 (D043).

---

## 12. Deliverables (proposed **[ASSUMPTION]**)

- The notebooks above, runnable top to bottom on Colab.
- `results/tables/comparison.csv` and figures from notebook 90.
- A short written judgement per algorithm (from the verdict cards).
- This file with a completed decision log, as evidence of justified choices.

## 13. Status

- [x] Dataset exploration (Luna) done, audited and written into section 3 and `DATA_PREPARATION_v5.md` [V6]
- [x] 00 audit notebook reproduces the gates of `DATA_PREPARATION_v5.md` step 1
- [ ] Front-end design decided (6b, D021–D034), pending owner confirmation and the checks in 6b.8
- [x] Split frozen in notebook 01 (decisions made, V8; gates in `DATA_PREPARATION_v5.md` step 5)
- [x] Features cached
- [ ] Classical models frozen (10–18)
- [ ] MLP frozen (20)
- [ ] CNN frozen (30)
- [ ] Front-end normalisation selected (30, part B)
- [ ] Comparison and final test (90)
- [ ] Model bundle exported (91)
- [ ] Laptop demo working in file mode, then live mode
