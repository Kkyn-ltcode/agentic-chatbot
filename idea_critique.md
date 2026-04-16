# Critique & Proposed Modifications to `idea.md`

> **Goal:** Transform this from a *good workshop paper* into a *credible top-tier submission*.
> I will be blunt. That's what you need before investing 7+ months.

---

## Overall Assessment

The idea document is well-structured and identifies a genuinely interesting research question. The four-experiment design (RSA → Ablation → Priming → Saliency) is methodologically mature. However, there are **five critical problems** that, if left unaddressed, will result in desk rejection at any A* venue. Each is fixable. Let's go through them.

---

## Problem 1: The "Syntax Skeleton Tensor" Is Conceptually Broken

> [!CAUTION]
> This is the single biggest threat to the paper. If the central reference representation is ill-defined, all four experiments collapse.

### What the idea says

> *"Extract the raw tensor of the DisCoCat diagram before any semantic parameters are assigned. Flatten to 1024 dims."*

### Why this doesn't work

A DisCoCat diagram **before parameterization** is a morphism in a compact closed category — it's a *type-level wiring diagram*, not a numerical object. To get any numbers out of it, you must apply a **functor** that maps:
- Grammar types (noun, sentence) → vector space dimensions
- Word boxes → concrete tensors

But once you assign concrete tensors to word boxes, you've introduced **semantic content**. The resulting vector is no longer "pure syntax." If you assign random tensors, the result is random noise. If you assign trained tensors, it's semantics. There is no middle ground that gives you "pure grammatical structure as a vector."

The pseudocode in the document:
```python
syntax_tensor = diagram.to_tn().contract().tensor.flatten()
```
...will not work. `diagram.to_tn()` requires the diagram to already be in a concrete tensor category (i.e., post-functor). Pre-functor diagrams don't have `.tensor` attributes.

### The Fix: Structural Fingerprint, Not Contracted Tensor

Instead of trying to extract a "tensor," extract the **topological/combinatorial structure** of the diagram as a feature vector. This is actually more defensible as "pure syntax" because it contains zero semantic information.

**Concrete proposal — A Diagram Structural Fingerprint:**

| Feature | Description | Example Value |
|:---|:---|:---|
| Number of boxes (words) | How many morphisms | 5 |
| Number of cups | Grammatical contractions | 3 |
| Number of caps | Grammatical expansions | 1 |
| Number of swaps | Crossing wires | 0 |
| Type signature histogram | Distribution of atomic types (n, s) on wires | [4, 2, 1] |
| Diagram depth | Longest path from input to output | 3 |
| Branching factor | Average fan-out at each layer | 1.6 |
| Type complexity | Total number of atomic types in domain/codomain | 7 |
| Cup-nesting depth | Maximum depth of nested cups | 2 |
| Wire entanglement graph | Adjacency matrix of which types connect | flattened vector |

Concatenate these into a fixed-size vector (pad to, say, 128 dims). This is your **syntax skeleton**: a purely structural descriptor with no semantic contamination.

**Why this is stronger for the paper:** You can now make a clean claim: *"We compare representations in the quantum state space to a purely structural descriptor of grammatical wiring, with zero semantic leakage."* No reviewer can argue that your syntax reference is contaminated by content.

**Alternative (more powerful but more work):** Use a **graph kernel** (Weisfeiler-Leman subtree kernel) on the diagram's underlying graph. This captures richer structural information and produces a kernel matrix directly usable for CKA. No fixed-size vector needed — CKA operates on kernel matrices.

---

## Problem 2: The Dataset Is Too Small for the Claims

> [!WARNING]
> 235 natural sentences + 200 synthetic is insufficient for statistically reliable CKA, ablation studies, and transfer experiments.

### The numbers

| Dataset | Size | Used For |
|:---|:---|:---|
| MC | ~130 sentences | Training + Eval |
| RelPron | ~105 sentences | Training + Eval |
| Synthetic Wh | ~200 sentences | Transfer test |
| **Total** | **~435** | **All experiments** |

### Why this is fatal

1. **CKA on small N is unreliable.** Recent work (NeurIPS 2024) proved that CKA exhibits substantial finite-sample bias when the number of samples is much smaller than the feature dimensionality. With 105 RelPron sentences and 384-dim SBERT vectors, you're in the regime where CKA trends toward 1.0 regardless of actual alignment. Reviewers will know this.

2. **Ablation variance will be catastrophic.** A 4-qubit PQC trained on 100 sentences has enormous variance across random seeds. Even with 5 seeds, confidence intervals will overlap across conditions, making it impossible to draw causal conclusions.

3. **Transfer testing on 200 synthetic sentences is unconvincing.** If the model gets 55% accuracy on Wh-questions (vs. 50% chance), is that signal or noise? With 200 binary-classification samples, the 95% confidence interval for 50% chance is roughly [43%, 57%]. You'd need >57% to claim above-chance performance.

### The Fix: A Three-Pronged Data Strategy

1. **Augment the syntactic benchmarks.** Use template-based generation (CFG or frame-based) to create controlled syntactic datasets, at minimum:
   - 500 relative clause sentences (extending RelPron)
   - 500 Wh-movement sentences
   - 500 garden-path / center-embedding sentences (new syntactic phenomenon)
   
   Use a fixed vocabulary drawn from RelPron/MC to control for lexical effects. You can generate these programmatically — no LLM needed.

2. **Add BLiMP subsets.** BLiMP (Benchmark of Linguistic Minimal Pairs) provides ~67 sub-tasks covering specific syntactic phenomena. Select 4-5 relevant sub-tasks (e.g., `anaphor_agreement`, `filler_gap_dependency`, `relative_clause`) and adapt them to binary classification. This adds ~1000+ sentences per sub-task and gives you coverage across syntactic phenomena, which strengthens the generalizability claim enormously.

3. **Use debiased CKA.** Instead of naive CKA, use the debiased U-statistic estimator (Dávari et al., 2023). Report permutation-test p-values alongside CKA scores. This is a one-line change in code but a massive credibility boost in the paper.

---

## Problem 3: The PQC vs. MLP Narrative Will Probably Fail

> [!IMPORTANT]
> At 4-qubit scale (~20-60 parameters), there is **no theoretical reason** to expect PQC to outperform a parameter-matched MLP. Recent benchmarking studies (2024-2025) confirm this empirically. If your central claim is "PQC beats MLP," you will lose.

### What the idea claims

> *"The PQC should outperform the classical MLP on syntactically ambiguous examples."*

### Why this is dangerous

The entire narrative hinges on H1 being true and H3 being false. But:

- A 4-qubit PQC operates in a 16-dimensional Hilbert space. A 2-layer MLP with 16 hidden units has access to the same representational capacity.
- Recent large-scale benchmarking shows that classical models are generally competitive with or outperform quantum classifiers at this scale. Entanglement alone does not confer advantage.
- If the MLP matches the PQC (highly likely), the paper's Story collapses and reviewers see a negative result with no insight.

### The Fix: Reframe from "Advantage" to "Mechanism"

**Do not frame the paper as a quantum advantage claim.** Instead, frame it as a **mechanistic interpretability** study. The question becomes:

> *"Do PQCs and classical non-linearities arrive at the correct answer via the same representational pathway, or do they employ qualitatively different strategies?"*

This is a much stronger framing because:
1. **It's always publishable.** Whether PQC and MLP differ or not, you've produced insight.
2. **It's novel.** Nobody has done mechanistic comparison of PQC vs. MLP representations for NLP tasks.
3. **It avoids the quantum advantage minefield.** Reviewers won't trigger on hype.

**Concrete modification to Experiment 2:**

| Original Design | Revised Design |
|:---|:---|
| PQC vs. MLP: which gets higher accuracy? | PQC vs. MLP: do their internal representations differ? |
| Claim: PQC is "necessary" | Claim: PQC representations are structurally different from MLP |
| Metric: accuracy gap | Metric: CKA(PQC_states, Syntax) vs. CKA(MLP_hidden, Syntax) + representation geometry analysis |

Even if both achieve 72% accuracy, you can show that the PQC's internal state has higher structural alignment with the syntactic skeleton than the MLP's hidden layer. That's a *discovery*, not a benchmark.

---

## Problem 4: Gradient Attribution Through PCA Is Non-Differentiable

> [!WARNING]
> The gradient saliency experiment (Experiment 4) has a technical blocker.

### The pipeline

```
Token → SBERT(384-dim) → PCA(16-dim) → PQC → output
```

### The problem

PCA truncation is a **fixed linear projection** fitted on training data. It is not part of the computational graph. You cannot backpropagate through `sklearn.PCA` — there are no gradients to compute through the truncated principal components.

Even if you implement PCA as a matrix multiplication (which is differentiable), the inverse mapping from 16→384 dims is **non-unique** (you lost 368 dimensions). Mapping saliency back to token-level is therefore ambiguous.

### The Fix: Replace PCA with a Trainable Linear Projection

Use a learnable `nn.Linear(384, 16)` instead of PCA. This:
1. Preserves gradient flow through the full pipeline.
2. Gives the model the ability to learn task-relevant compression (better than PCA anyway).
3. Makes Integrated Gradients mathematically well-defined from PQC output back to 384-dim input.

**Bonus:** You now have a richer Experiment 2. You can compare:
- **PCA + PQC** (fixed compression, quantum classification)
- **Learned Linear + PQC** (learned compression, quantum classification)
- **Learned Linear + MLP** (learned compression, classical classification)

This gives you a 2×2 factorial design (compression × classifier), which is methodologically much stronger.

For token-level attribution:
- Compute Integrated Gradients from PQC output to the 384-dim SBERT vector.
- Use the SBERT model's own token-embedding matrix to project the 384-dim saliency vector back to token saliency scores. This is the standard approach in BERTology.

---

## Problem 5: Venue Strategy Needs Adjustment

### ACL may not be the best first target

ACL reviewers are computational linguists. They will:
- Be skeptical of quantum computing claims (rightly).
- Expect deep linguistic analysis that goes beyond "function words get more attention."
- Want comparison to established probing methods on established models.

### Better venue sequence

| Venue | Deadline (projected) | Why |
|:---|:---|:---|
| **NeurIPS 2026 QML Workshop** | ~Sep 2026 | Test run. Get feedback. Almost certainly accepted. |
| **EMNLP 2027** | ~Jun 2027 | More methodologically open than ACL. Strong interpretability track. |
| **ICLR 2028** | ~Oct 2027 | Best for ML interpretability + quantum ML crossover. |
| **ACL 2027** | ~Dec 2026 | Keep as option, but only if linguistic analysis is very deep. |

---

## The Revised Framing

Here is the modified paper, with all five fixes incorporated:

### Revised Title
*"How Does the Quantum Circuit Solve It? Mechanistic Interpretability of Hybrid Quantum-Classical Models on Syntactic Tasks"*

### Revised Research Question
> In a hybrid SBERT → Learned Compression → PQC pipeline applied to syntactic tasks, **does the quantum circuit employ a qualitatively different representational strategy than a matched classical non-linearity, and does that strategy encode structural (syntactic) information?**

### Revised Hypotheses
- **H1 (Structural encoding):** The PQC's quantum state vectors exhibit significantly higher alignment with syntactic structure (as measured by CKA with diagram fingerprints) than the compressed input representations.
- **H2 (Pathway divergence):** The PQC and a parameter-matched MLP, even when achieving similar task accuracy, arrive at solutions via representationally distinct pathways (measured by CKA between their respective hidden states).
- **H3 (Syntactic generalization):** The PQC exhibits non-trivial transfer to unseen syntactic constructions, consistent with abstract rule learning rather than lexical memorization.

### Revised Experimental Design

| # | Experiment | Key Modification from Original |
|:---|:---|:---|
| 1 | **RSA/CKA Analysis** | Use **diagram structural fingerprint** (not contracted tensor). Use **debiased CKA** with permutation tests. Run on **MC + RelPron + BLiMP subsets** (~2000+ sentences). |
| 2 | **Mechanistic Comparison** | Frame as PQC vs. MLP **representation geometry**, not accuracy contest. Use 2×2 factorial design (PCA vs. Learned Linear) × (PQC vs. MLP). Report CKA with syntax skeleton for all four conditions. |
| 3 | **Syntactic Transfer** | Expand synthetic datasets to **500+ sentences per construction**. Test on 2-3 syntactic phenomena (RelPron → Wh-movement, center-embedding). Report with bootstrap confidence intervals. |
| 4 | **Gradient Saliency** | Replace PCA with **trainable linear projection**. Use Integrated Gradients end-to-end. Map to tokens via SBERT embedding matrix. Correlate saliency with POS tags quantitatively (not just visually). |

### Revised Paper Outline

```
1. Introduction
   - Hybrid QNLP models work, but why?
   - We introduce mechanistic interpretability tools for quantum NLP circuits.
   - Three hypotheses (structural encoding, pathway divergence, syntactic generalization).

2. Background
   - 2.1 DisCoCat and lambeq
   - 2.2 Hybrid SBERT + PQC pipelines
   - 2.3 Probing and mechanistic interpretability in classical NLP
   - 2.4 The gap: no interpretability work for quantum NLP circuits

3. Methodology
   - 3.1 Hybrid Model Architecture (SBERT → Linear → PQC)
   - 3.2 Diagram Structural Fingerprints (the syntax reference)
   - 3.3 Debiased CKA for Representational Similarity
   - 3.4 Mechanistic Comparison Protocol (2×2 factorial)
   - 3.5 Syntactic Transfer Test Design
   - 3.6 Integrated Gradients for Quantum Circuits

4. Experimental Setup
   - 4.1 Datasets (MC, RelPron, BLiMP subsets, Synthetic)
   - 4.2 Model configurations and hyperparameters
   - 4.3 Statistical methodology (5+ seeds, bootstrap CIs, permutation tests)

5. Results
   - 5.1 Task Performance (all models, all datasets)
   - 5.2 RSA: Quantum states align with syntactic structure
   - 5.3 PQC vs. MLP: Same accuracy, different representations
   - 5.4 Transfer: Generalization across syntactic constructions
   - 5.5 Saliency: Where the quantum circuit attends

6. Discussion
   - What this means for QNLP architecture design
   - Limitations (simulator-only, small scale)
   - Connection to quantum kernel theory

7. Conclusion
```

---

## Summary of Changes

| Aspect | Original | Revised |
|:---|:---|:---|
| **Syntax reference** | Contracted tensor (broken) | Diagram structural fingerprint (well-defined, zero semantic leakage) |
| **Dataset size** | ~435 sentences | ~2000+ sentences (MC + RelPron + BLiMP + synthetic) |
| **Central claim** | "PQC recovers syntax; MLP cannot" | "PQC and MLP use different representational strategies; PQC's is more syntax-aligned" |
| **Compression** | PCA (blocks gradients) | Trainable linear projection (preserves gradient flow) |
| **CKA method** | Naive CKA | Debiased CKA + permutation tests |
| **Narrative frame** | Quantum advantage | Mechanistic interpretability |
| **Primary venue** | ACL 2027 | EMNLP 2027 or ICLR 2028 |

> [!TIP]
> The revised framing is **strictly stronger** than the original. If the PQC turns out to encode syntax better than the MLP, you've shown something remarkable. If it doesn't, you've still produced the first mechanistic comparison — which is a contribution regardless of outcome. The original framing only works if H1 is true. The revised framing works no matter what the data shows.
