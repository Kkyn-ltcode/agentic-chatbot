# Master Roadmap: From Zero to Top-Tier Paper

**Paper:** *"How Does the Quantum Circuit Solve It? Mechanistic Interpretability of Hybrid Quantum-Classical Models on Syntactic Tasks"*

**Timeline:** 28 weeks (~7 months) at 15-20 hrs/week  
**Start Date:** April 2026  
**Target Venue:** EMNLP 2027 (deadline ~Jun 2027) or ICLR 2028 (deadline ~Oct 2027)  
**Intermediate:** NeurIPS 2026 QML Workshop (~Sep 2026)

> [!IMPORTANT]
> This roadmap incorporates all fixes from the critique: structural fingerprints (not contracted tensors), expanded datasets, mechanistic framing (not quantum advantage), trainable projection (not PCA), and debiased CKA.

---

## Phase 0 — Environment & Foundations (Week 0)

**Goal:** Working dev environment. Mental model of every library you'll use.

### Tasks

- [ ] **0.1 — Create conda environment**
  ```bash
  conda create -n qnlp python=3.10 -y
  conda activate qnlp
  ```

- [ ] **0.2 — Install all dependencies**
  ```bash
  pip install pennylane pennylane-lightning lambeq discopy \
    sentence-transformers torch torchvision scikit-learn \
    cca-zoo matplotlib seaborn pandas numpy \
    wandb jupyter tqdm
  ```

- [ ] **0.3 — Verify installations**
  - Run `lambeq` Hello World: parse a sentence with `BobcatParser`, draw the diagram.
  - Run PennyLane quickstart: create a 4-qubit circuit, measure expectation values.
  - Run sentence-transformers: encode a sentence with `all-MiniLM-L6-v2`, check output shape is `(384,)`.

- [ ] **0.4 — Set up project structure**
  ```
  Quantum/
  ├── src/
  │   ├── data/           # Dataset loading, generation, preprocessing
  │   ├── models/          # PQC model, MLP baseline, hybrid pipeline
  │   ├── probing/         # RSA/CKA, ablation, transfer, saliency
  │   ├── syntax/          # Diagram structural fingerprint extraction
  │   └── utils/           # Logging, seeding, config
  ├── notebooks/           # Exploration and visualization
  ├── configs/             # Hyperparameter configs (YAML)
  ├── scripts/             # Entry points: train.py, evaluate.py, run_all.py
  ├── data/                # Raw and processed datasets
  ├── results/             # Experiment outputs, figures, tables
  ├── paper/               # LaTeX source
  ├── tests/               # Unit tests for critical functions
  └── requirements.txt
  ```

- [ ] **0.5 — Set up experiment tracking**
  - Initialize a Weights & Biases project (or MLflow if you prefer local).
  - Create a `run_config` dataclass that captures all hyperparameters.

- [ ] **0.6 — Study the math (parallel with coding)**
  - Read: Coecke, Sadrzadeh & Clark (2010) — *"Mathematical Foundations for a Compositional Distributional Model of Meaning"* — This is the DisCoCat paper. You need to understand types, cups, caps, and functors.
  - Read: Kornblith et al. (2019) — *"Similarity of Neural Network Representations Revisited"* — This is the CKA paper. Understand linear CKA vs. RBF CKA.
  - Read: Hewitt & Liang (2019) — *"Designing and Interpreting Probes with Control Tasks"* — This will sharpen your probing methodology. Critical for avoiding the "probing illusion."
  - Skim: Kübler et al. (2021) — *"The Inductive Bias of Quantum Kernels"* — Understand how quantum kernel methods relate to PQC expressivity.

**✅ Checkpoint:** You can parse a sentence, draw its diagram, encode it with SBERT, and run a trivial PQC — all in one notebook.

---

## Phase 1 — Data Infrastructure (Weeks 1–3)

**Goal:** All datasets ready, cleaned, split, and versioned. You should never touch data code again after this phase.

### Week 1: Load and Understand Existing Datasets

- [ ] **1.1 — Load MC dataset from lambeq**
  - Parse all ~130 sentences with `BobcatParser`.
  - Record which sentences fail to parse (there will be some). Document failure modes.
  - Save parsed diagrams as serialized objects.
  - Create train/val/test splits (60/20/20). Use stratified splits to preserve label balance.

- [ ] **1.2 — Load RelPron dataset from lambeq**
  - Same procedure as MC. ~105 sentences.
  - Understand the task structure: each example is a (noun, relative clause, label) triple.

- [ ] **1.3 — Explore both datasets thoroughly**
  - Plot label distributions.
  - Plot sentence length distributions.
  - Examine 20 diagrams manually. Understand the type structures.
  - Document: What syntactic phenomena does each dataset test?

### Week 2: Generate Synthetic Datasets

- [ ] **1.4 — Build a syntactic sentence generator**
  - Write a template-based CFG (Context-Free Grammar) that generates sentences with controlled syntactic structure.
  - Use the **same vocabulary** as RelPron/MC to control for lexical effects.
  - Target constructions:
    - **Wh-movement questions:** "Which planet did the device detect?" (~500 sentences)
    - **Center-embedding:** "The cat the dog chased ran." (~500 sentences)
    - **Simple transitive control:** "The dog chased the cat." (~500 sentences)
  - Each sentence must have a binary label (grammatical/ungrammatical or meaning match/mismatch — consistent with the task format).

- [ ] **1.5 — Select and adapt BLiMP subsets**
  - Download BLiMP from [the official repository](https://github.com/alexwarstadt/blimp).
  - Select 4-5 relevant sub-tasks:
    - `anaphor_agreement`
    - `filler_gap_dependency`
    - `irregular_past_participle_adjectives` (morpho-syntactic)
    - `sentential_complement_no_that` (long-distance dependency)
    - `wh_questions_object_gap` (directly relevant)
  - Convert each to a binary classification format compatible with your pipeline.
  - Parse all selected sentences with `BobcatParser`. Remove unparseable ones.

### Week 3: Preprocessing and Embedding Extraction

- [ ] **1.6 — Extract SBERT embeddings for all datasets**
  - Run `all-MiniLM-L6-v2` on every sentence.
  - Save: `{dataset_name}_sbert_384.npy` for each dataset.
  - Verify: Check embedding shapes, run a quick cosine similarity sanity check (similar sentences should have high similarity).

- [ ] **1.7 — Create a unified data pipeline**
  - Write a `SyntaxDataset` PyTorch class that returns `(sbert_embedding, label, sentence_id, dataset_name)`.
  - Write a `DataModule` that handles all datasets, splits, and batching.
  - All downstream code should use this single interface.

- [ ] **1.8 — Version and freeze all data**
  - Save all processed data with checksums.
  - Document dataset statistics in a `data/README.md`.
  - **Never modify data files after this point.**

**✅ Checkpoint:** You have ~2500+ sentences across 6+ dataset splits, all with SBERT embeddings and parsed diagrams. A single `DataModule` class serves everything.

---

## Phase 2 — Model Pipeline (Weeks 4–6)

**Goal:** Four working models, all trainable and evaluable through a single script.

### Week 4: Classical Baselines

- [ ] **2.1 — Logistic Regression baseline**
  - Train `sklearn.LogisticRegression` on raw 384-dim SBERT embeddings for each dataset.
  - Report accuracy with 5-fold cross-validation.
  - This is your **classical upper bound** (no compression, no quantum).

- [ ] **2.2 — MLP baseline (full dimensionality)**
  - 2-layer MLP, 64 hidden units, ReLU, on 384-dim SBERT.
  - Train with Adam, early stopping.
  - Report accuracy ± std over 5 seeds.

- [ ] **2.3 — MLP baseline (compressed)**
  - `nn.Linear(384, 16)` → `nn.ReLU` → `nn.Linear(16, 64)` → `nn.ReLU` → `nn.Linear(64, 1)`.
  - The `nn.Linear(384, 16)` is the trainable compression layer.
  - **Match the total parameter count** to what the hybrid PQC model will have. Document the count.

### Week 5: Hybrid PQC Model

- [ ] **2.4 — Build the trainable compression layer**
  - `nn.Linear(384, 16)` — this replaces PCA. It's part of the computational graph.
  - Initialize with PCA weights as a warm start (optional but useful).

- [ ] **2.5 — Build the 4-qubit PQC in PennyLane**
  - Architecture: 4 qubits, data encoding via `AngleEmbedding` (4 features per layer, 4 layers = 16 inputs), variational layers via `StronglyEntanglingLayers` (2 layers).
  - Output: expectation value of `PauliZ` on qubit 0 (for binary classification).
  - Wrap as a `torch.nn.Module` using `qml.qnn.TorchLayer`.

- [ ] **2.6 — Build the full hybrid pipeline**
  ```
  SBERT(frozen) → nn.Linear(384, 16) → PQC(4 qubits) → sigmoid → BCE loss
  ```
  - Train with Adam (lr=0.01 for PQC, lr=0.001 for linear layer — PQC needs higher lr).
  - Implement gradient clipping (PQC gradients can be unstable).

- [ ] **2.7 — Train hybrid model on MC and RelPron**
  - Report accuracy ± std over 5 seeds.
  - Log training curves to W&B.
  - Compare to classical baselines.

### Week 6: DisCoCat Baseline + PCA Variant

- [ ] **2.8 — Pure DisCoCat baseline**
  - Implement the standard `lambeq` pipeline: `BobcatParser` → `IQPAnsatz` → `PennyLaneModel`.
  - Train with SPSA optimizer on MC and RelPron.
  - Report accuracy. Expect ~65-72% on MC, ~55-60% on RelPron.

- [ ] **2.9 — PCA + PQC variant (for comparison)**
  - `sklearn.PCA(n_components=16)` on SBERT embeddings → feed to same PQC architecture.
  - This is the **non-differentiable** compression variant. You need it for the 2×2 factorial design.
  - Train with Adam (PQC params only; PCA is frozen).

- [ ] **2.10 — Results summary table**
  - Create Table 0 (internal): All baselines across all datasets.
  - This table is your sanity check. If Logistic Regression beats everything, your task is too easy or your models are broken.

**✅ Checkpoint:** Four models trained and evaluated. You have a results table with 5-seed error bars. Hybrid PQC should be competitive with (not necessarily better than) the MLP.

---

## Phase 3 — Syntax Skeleton Tooling (Weeks 7–8)

**Goal:** A well-defined, validated, purely structural representation of each sentence's grammar.

### Week 7: Build the Diagram Structural Fingerprint

- [ ] **3.1 — Write the `extract_structural_fingerprint(diagram)` function**
  - Input: a `lambeq` diagram (output of `BobcatParser`).
  - Extract the following features from the diagram's structure:

  | Feature | How to Extract |
  |:---|:---|
  | Number of boxes | `len(diagram.boxes)` |
  | Number of cups | Count `Cup` morphisms |
  | Number of caps | Count `Cap` morphisms |
  | Number of swaps | Count `Swap` morphisms |
  | Box type signatures | For each box: `(len(box.dom), len(box.cod))` → histogram |
  | Total wire count | `len(diagram.dom) + len(diagram.cod)` + internal |
  | Diagram depth | Longest sequential composition chain |
  | Type distribution | Histogram of atomic types (`n`, `s`, `n.r`, `s.l`, etc.) |
  | Cup nesting depth | Max recursion depth of nested cups |
  | Connectivity pattern | Adjacency of box inputs/outputs → flatten upper triangle |

  - Concatenate all features into a fixed-size vector (pad/truncate to 128 dims).
  - Normalize: z-score each feature across the corpus.

- [ ] **3.2 — Write unit tests**
  - Test that two sentences with identical syntactic structure (but different words) produce identical fingerprints.
  - Test that structurally different sentences produce different fingerprints.
  - Test edge cases: single-word sentences, very long sentences, failed parses.

- [ ] **3.3 — Extract fingerprints for all sentences**
  - Run on all datasets. Save: `{dataset_name}_syntax_fingerprints.npy`.
  - Handle parse failures gracefully (skip or impute).

### Week 8: Validate the Fingerprint

- [ ] **3.4 — Clustering validation**
  - Compute pairwise cosine similarity between all syntax fingerprints.
  - Apply MDS or t-SNE. Color by syntactic construction type.
  - **Expected result:** Relative clauses cluster together, Wh-questions cluster separately, simple transitives form a third cluster.
  - If this doesn't happen, the fingerprint is too coarse. Add more features or switch to the graph kernel approach.

- [ ] **3.5 — Discriminability test**
  - Train a simple classifier (Logistic Regression) on syntax fingerprints → predict syntactic construction type.
  - If accuracy is near chance, your fingerprint contains no syntactic information. Go back to 3.1.
  - Target: >80% accuracy at distinguishing construction types.

- [ ] **3.6 — CKA baseline: Syntax vs. SBERT**
  - Compute `debiased_linear_CKA(syntax_fingerprints, sbert_embeddings)` for each dataset.
  - This is your **baseline** number. If syntax and SBERT are already highly aligned, compression won't "destroy" syntax (weakening your story).
  - Expected: moderate CKA (0.1-0.4). SBERT should partially capture syntax, but not perfectly.

- [ ] **3.7 — Graph kernel alternative (if fingerprint fails validation)**
  - If step 3.4/3.5 fails, implement a Weisfeiler-Leman subtree kernel on the diagram's graph structure.
  - Use `grakel` library. Produces a kernel matrix directly usable in CKA (no fixed-size vector needed).

**✅ Checkpoint:** You have a validated syntax representation that clusters by construction type and is distinguishable from semantic (SBERT) representations. You have a baseline CKA number.

---

## Phase 4 — Probing Experiments (Weeks 9–15)

**Goal:** Execute the four core experiments. This is the heart of the paper.

### Experiment 1: RSA / CKA Across Compression (Weeks 9–10)

- [ ] **4.1 — Define the comparison matrix**
  - For each dataset, you need representations at 5 stages:
    1. Syntax fingerprint (128-dim)
    2. Raw SBERT (384-dim)
    3. Compressed SBERT (16-dim, from trained linear layer)
    4. PQC quantum state vector (16-dim, from `default.qubit` statevector)
    5. MLP hidden layer activations (from matched MLP)

- [ ] **4.2 — Extract quantum state vectors**
  - Modify the PQC forward pass to return the full statevector before measurement.
  - In PennyLane: use `qml.state()` as a measurement in a separate QNode.
  - Save: `{dataset_name}_pqc_states.npy` (shape: `N × 16`).

- [ ] **4.3 — Extract MLP hidden activations**
  - Register a forward hook on the MLP's hidden layer.
  - Save: `{dataset_name}_mlp_hidden.npy`.

- [ ] **4.4 — Compute the full CKA matrix**
  - For each pair of representation spaces (5 stages → 10 pairs):
    - Compute debiased linear CKA.
    - Compute permutation test p-value (1000 permutations).
    - Compute bootstrap 95% confidence interval (1000 resamples).
  - Present as a 5×5 heatmap.

- [ ] **4.5 — CKA across compression levels**
  - Vary compression dimension: d ∈ {4, 8, 16, 32, 64, 128}.
  - For each d, retrain the hybrid model (linear + PQC) and the matched MLP.
  - Plot: `CKA(representation, syntax_fingerprint)` vs. compression dimension.
  - Four lines on the plot: raw SBERT, compressed, PQC state, MLP hidden.
  - **This is Figure 2 of the paper.** The key question: does the PQC line stay above the compressed line?

- [ ] **4.6 — Statistical validation**
  - For the critical comparison (PQC vs. MLP alignment with syntax), run a paired permutation test.
  - Report: "The PQC's quantum state shows significantly higher alignment with syntactic structure than the MLP's hidden layer (ΔCKA = X.XX, p < 0.01)."
  - If p > 0.05, you still have a valid result — frame it as "no significant difference in structural alignment."

### Experiment 2: Mechanistic Comparison — 2×2 Factorial (Weeks 11–12)

- [ ] **4.7 — Define the four conditions**

  | Condition | Compression | Classifier | Description |
  |:---|:---|:---|:---|
  | A | PCA (frozen) | PQC | Fixed compression, quantum classification |
  | B | PCA (frozen) | MLP | Fixed compression, classical classification |
  | C | Learned Linear | PQC | Learned compression, quantum classification |
  | D | Learned Linear | MLP | Learned compression, classical classification |

- [ ] **4.8 — Train all four conditions**
  - For each condition × each dataset × 5 seeds = many runs. Use W&B sweeps or a simple loop.
  - Record: accuracy, loss curves, final model weights.

- [ ] **4.9 — Ablation interventions on the best model (Condition C)**
  - **Ablation A (Reset PQC):** Randomly reinitialize PQC parameters. Keep compression layer frozen. Retrain PQC only. Measure accuracy.
  - **Ablation B (Freeze PQC):** Freeze PQC. Add a trainable linear output layer. Train only the output layer. Measure accuracy.
  - **Ablation C (Remove PQC):** Replace PQC with identity. Feed compressed features directly to a linear classifier. Measure accuracy.

- [ ] **4.10 — Representation geometry analysis**
  - For each of the 4 conditions, extract internal representations.
  - Compute intrinsic dimensionality (using MLE estimator) of each representation space.
  - Compute effective rank of the representation matrix.
  - **Question:** Does the PQC produce higher-dimensional representations than the MLP? If so, it's expanding the feature space — a structural difference, even if accuracy is the same.

- [ ] **4.11 — Assemble Table 1**
  - Rows: all conditions + ablations.
  - Columns: accuracy (mean ± std), CKA with syntax, intrinsic dimensionality, effective rank.

### Experiment 3: Syntactic Transfer (Weeks 13–14)

- [ ] **4.12 — Define the transfer protocol**
  - Train the hybrid model (Condition C) on **RelPron** (relative clauses).
  - Test zero-shot on:
    - Wh-movement questions (different construction, same vocabulary)
    - Center-embedding sentences (different construction, same vocabulary)
    - Simple transitives (control — should be easy)
  - Also train and test the matched MLP (Condition D) under identical protocol.

- [ ] **4.13 — Run the transfer experiments**
  - For each model × each test set × 5 seeds:
    - Record accuracy.
    - Record confidence (mean output probability, calibration).

- [ ] **4.14 — Statistical analysis of transfer**
  - For each test set, compute:
    - Accuracy vs. chance baseline (50%). One-sided binomial test.
    - PQC accuracy vs. MLP accuracy. Paired t-test or Wilcoxon signed-rank.
  - Bootstrap 95% CIs on all accuracy numbers.

- [ ] **4.15 — Assemble Table 2**
  - Rows: train set (RelPron) × model (PQC, MLP).
  - Columns: test accuracy on each syntactic construction + chance baseline.

- [ ] **4.16 — Qualitative error analysis**
  - For PQC and MLP: which specific sentences does each get wrong on transfer?
  - Are the errors systematic? (e.g., does the MLP fail on all center-embeddings?)
  - Document 5-10 illustrative examples for the paper's appendix.

### Experiment 4: Gradient Saliency (Week 15)

- [ ] **4.17 — Implement Integrated Gradients for the hybrid pipeline**
  - Use Captum library or implement from scratch.
  - Baseline: zero vector (standard for IG).
  - Target: the compression layer input (384-dim SBERT space).
  - Output: a 384-dim attribution vector for each sentence.

- [ ] **4.18 — Map attributions to tokens**
  - For each sentence, get the SBERT token embeddings (not just [CLS]).
  - Project the 384-dim IG attribution onto each token's embedding using dot product.
  - Result: a saliency score per token.

- [ ] **4.19 — Correlate saliency with POS tags**
  - POS-tag all sentences using spaCy.
  - Group tokens by POS tag. Compute mean saliency per POS tag.
  - **Hypothesis:** Function words (DET, AUX, PRON, SCONJ) should have higher saliency than content words (NOUN, VERB, ADJ) if the model encodes syntax.
  - Run a Mann-Whitney U test: function word saliency vs. content word saliency.

- [ ] **4.20 — Compare PQC saliency vs. MLP saliency**
  - Run identical IG analysis on the matched MLP.
  - Compute: does the PQC concentrate saliency on function words more than the MLP?
  - This is a direct test of whether PQC and MLP "attend" differently.

- [ ] **4.21 — Create Figure 3: Saliency heatmaps**
  - Select 6-8 example sentences (2 per syntactic construction type).
  - Show token-level saliency as a heatmap for both PQC and MLP side by side.
  - Show the aggregate POS-tag saliency bar chart.

**✅ Checkpoint:** All four experiments complete. You have Figures 2-3, Tables 1-2, and statistical tests for every claim.

---

## Phase 5 — Robustness & Analysis (Weeks 16–18)

**Goal:** Make every result bulletproof. Anticipate reviewer objections.

### Week 16: Hyperparameter Sensitivity

- [ ] **5.1 — Sweep PQC depth**
  - Test: 1 layer, 2 layers (default), 3 layers, 4 layers.
  - For each depth, re-run Experiments 1 and 2 (accuracy + CKA).
  - **Question:** Does the syntactic alignment increase with depth? (If so, strong evidence for H1.)

- [ ] **5.2 — Sweep compression dimension**
  - Already partially done in 4.5. Ensure all conditions (PCA/Learned × PQC/MLP) are swept.
  - Summary plot: accuracy and CKA vs. compression dimension for all four conditions.

- [ ] **5.3 — Sweep number of qubits**
  - Test: 2, 4, 6, 8 qubits (with corresponding compression dimensions 4, 16, 64, 256).
  - This addresses the "would it work at larger scale?" question.

### Week 17: Noise Robustness (Bonus, but strengthens paper significantly)

- [ ] **5.4 — Add depolarizing noise to PQC**
  - Use PennyLane's `qml.DepolarizingChannel` with noise rates p ∈ {0.001, 0.01, 0.05, 0.1}.
  - Re-run Experiment 1 (CKA) under each noise level.
  - Plot: CKA(PQC_state, syntax) vs. noise rate.
  - **Purpose:** Shows results are robust to realistic NISQ noise. Strong EMNLP/ICLR selling point.

- [ ] **5.5 — Shot noise simulation**
  - Switch from statevector simulator to shot-based simulation (1024, 4096, 8192 shots).
  - Re-run accuracy experiments.
  - Show that accuracy degrades gracefully.

### Week 18: Buffer & Catch-Up

- [ ] **5.6 — Fix failed experiments**
  - Re-run any experiments that had bugs or unexpected results.
  - Investigate any anomalous findings.

- [ ] **5.7 — Finalize all figures and tables**
  - Generate publication-quality figures (matplotlib + seaborn, proper font sizes, colorblind-friendly palettes).
  - All figures at 300 DPI, saved as both PDF and PNG.

- [ ] **5.8 — Organize the results directory**
  ```
  results/
  ├── tables/
  │   ├── table1_ablation.csv
  │   ├── table2_transfer.csv
  │   └── table0_baselines.csv
  ├── figures/
  │   ├── fig2_cka_compression.pdf
  │   ├── fig3_saliency.pdf
  │   ├── fig_noise_robustness.pdf
  │   └── fig_syntax_clusters.pdf
  └── raw/
      └── {all .npy files and W&B run IDs}
  ```

- [ ] **5.9 — Write a `run_all_experiments.sh`**
  - Single entry point that regenerates all results from raw data.
  - Must be reproducible with a fixed random seed.

**✅ Checkpoint:** Every result is reproducible, statistically tested, and visualized. You have sweep plots showing that findings are not sensitive to hyperparameters.

---

## Phase 6 — Writing (Weeks 19–23)

**Goal:** Publication-ready manuscript.

### Week 19: Introduction + Related Work

- [ ] **6.1 — Write the Introduction (Section 1)**
  - Para 1: Hybrid QNLP models achieve competitive performance, but we don't understand *how* they solve linguistic tasks.
  - Para 2: In classical NLP, probing and interpretability tools (BERTology) have revealed deep insights. No equivalent exists for quantum NLP circuits.
  - Para 3: We introduce a mechanistic interpretability framework for hybrid QNLP — four experiments testing where and how syntactic information flows.
  - Para 4: Contributions list (3-4 bullets).

- [ ] **6.2 — Write Related Work (Section 2)**
  - 2.1: DisCoCat, compositional distributional semantics, lambeq.
  - 2.2: Hybrid quantum-classical NLP (SBERT + PQC pipelines).
  - 2.3: Probing and interpretability in classical NLP (Tenney, Belinkov, Hewitt).
  - 2.4: Interpretability of quantum circuits (very few papers — emphasize the gap).

### Week 20: Methodology

- [ ] **6.3 — Write Methodology (Section 3)**
  - 3.1: Hybrid model architecture (precise math: equations for the linear projection, PQC unitary, measurement).
  - 3.2: Diagram structural fingerprint (define each feature, explain why it captures syntax without semantics).
  - 3.3: Debiased CKA (cite Dávari et al., explain the U-statistic estimator).
  - 3.4: Mechanistic comparison protocol (2×2 factorial, ablation interventions).
  - 3.5: Syntactic transfer test design.
  - 3.6: Integrated Gradients adapted for quantum circuits.
  - **Every section must have equations.** Reviewers at top venues expect mathematical precision.

### Week 21: Results

- [ ] **6.4 — Write Results (Section 4)**
  - 4.1: Baseline performance (Table 0 — concise, not the main contribution).
  - 4.2: RSA (Figure 2 + statistical analysis).
  - 4.3: Mechanistic comparison (Table 1 — 2×2 factorial + ablations).
  - 4.4: Syntactic transfer (Table 2 + error analysis).
  - 4.5: Gradient saliency (Figure 3 + POS-tag analysis).
  - **Rule:** Results section reports facts. No interpretation. Save that for Discussion.

### Week 22: Discussion + Conclusion

- [ ] **6.5 — Write Discussion (Section 5)**
  - What do the results mean for QNLP architecture design?
  - How do PQC and MLP representational strategies differ (or not)?
  - Connection to quantum kernel theory and expressivity results.
  - Limitations: simulator-only, small dataset scale, limited syntactic phenomena.
  - Future work: larger-scale experiments, real hardware, additional languages.

- [ ] **6.6 — Write Conclusion (Section 6)**
  - 1 paragraph. Restate contributions. End with the big picture.

- [ ] **6.7 — Write Abstract**
  - Write this LAST. 250 words max. Must contain: problem, gap, method, key finding, implication.

### Week 23: Polish + arXiv

- [ ] **6.8 — Internal review**
  - Read the entire paper aloud. Fix all awkward phrasing.
  - Check: every claim is supported by a specific experiment and number.
  - Check: every figure/table is referenced in the text.
  - Check: notation is consistent throughout.

- [ ] **6.9 — Format for arXiv**
  - Use the NeurIPS or ICML LaTeX template (these are standard for arXiv preprints).
  - Ensure all figures render correctly.
  - Check bibliography completeness.

- [ ] **6.10 — Post to arXiv**
  - Submit to `cs.CL` (primary) + `quant-ph` (cross-list).
  - This establishes priority and becomes visible to PhD admissions committees.
  - Share the arXiv link on Twitter/X and relevant communities (Quantum Computing, NLP).

**✅ Checkpoint:** arXiv preprint posted. You have a public, citable paper.

---

## Phase 7 — Workshop Submission (Week 24)

- [ ] **7.1 — Submit to NeurIPS 2026 QML Workshop**
  - Adapt the paper to the workshop format (typically 4-6 pages).
  - Focus on the most compelling experiment (probably the RSA + CKA analysis).
  - Workshop acceptance is very likely for a well-executed empirical study in this space.

- [ ] **7.2 — Prepare a poster/presentation**
  - Even if the workshop is months away, drafting a poster now forces clarity.

**✅ Checkpoint:** Workshop paper submitted.

---

## Phase 8 — Revision & Top-Tier Submission (Weeks 25–28)

### Week 25: Incorporate Workshop Feedback

- [ ] **8.1 — Analyze reviewer comments**
  - List every criticism. Classify as: (a) must fix, (b) should address, (c) can push back.

- [ ] **8.2 — Run additional experiments if needed**
  - Reviewers may request: different datasets, different metrics, comparison to additional baselines.
  - This is why Phase 5 included sweeps — you may already have the data.

### Weeks 26–27: Format for Target Venue

- [ ] **8.3 — Reformat for EMNLP 2027 or ICLR 2028**
  - EMNLP: ACL template, 8 pages + unlimited appendix. Linguistic framing.
  - ICLR: OpenReview format, 9 pages. ML interpretability framing.
  - **Decision point:** Choose based on where your results are strongest. If saliency + POS results are compelling → EMNLP. If representation geometry + CKA results dominate → ICLR.

- [ ] **8.4 — Write the rebuttal-ready appendix**
  - Include: all hyperparameter sweep results, noise robustness, dataset generation details, full statistical tables.
  - This preempts reviewer requests and signals thoroughness.

### Week 28: Submit

- [ ] **8.5 — Final proofreading**
  - Have someone else read it. Fresh eyes catch what yours won't.

- [ ] **8.6 — Submit**
  - Double-check anonymization (if required).
  - Submit. Celebrate.

**✅ Checkpoint:** Top-tier paper submitted. arXiv preprint citable for PhD applications.

---

## Quick Reference: Key Deliverables by Week

| Week | Key Deliverable |
|:---|:---|
| 0 | Environment running, project structure created |
| 3 | All datasets loaded, preprocessed, versioned |
| 6 | Four models trained and baselined |
| 8 | Syntax fingerprint validated, CKA baseline computed |
| 10 | Experiment 1 (RSA) complete — Figure 2 |
| 12 | Experiment 2 (Ablation) complete — Table 1 |
| 14 | Experiment 3 (Transfer) complete — Table 2 |
| 15 | Experiment 4 (Saliency) complete — Figure 3 |
| 18 | All sweeps done, results reproducible |
| 23 | arXiv preprint posted |
| 24 | Workshop paper submitted |
| 28 | Top-tier venue submitted |

---

> [!NOTE]
> **If you fall behind:** Phases 1-4 are non-negotiable. Phase 5 (sweeps) can be compressed. Phase 6 (writing) cannot be compressed — bad writing kills good science. Always protect writing time.

> [!TIP]
> **Weekly rhythm:** Spend the first 2 hours of each week reviewing the roadmap and updating checkpoints. Spend the last hour documenting what you did. The time between those bookends is for actual work. This habit prevents drift.
