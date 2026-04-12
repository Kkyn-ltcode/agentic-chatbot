# MambaFlow-IDS: End-to-End Research Blueprint

> **Paper Title (Working):** *MambaFlow: Empirical Validation of Selective State Space Models for Ultra-Long Context APT Detection in Continuous Network Traffic*
>
> **Target Venue:** IEEE Transactions on Information Forensics and Security (TIFS) — Q1
>
> **Estimated Timeline:** 4–5 months

---

## Part 0: The Elevator Pitch (Memorize This)

Every Mamba-NIDS paper (MeeDet, Norns, NetMamba) cites Mamba's $\mathcal{O}(L)$ linear scaling as their core motivation — then paradoxically truncates the input to 5–128 packets. **Nobody has ever tested whether Mamba actually works on sequences longer than a few hundred tokens for network intrusion detection.**

Your paper will be the first to:
1. Feed **100,000+ chronologically ordered packets** into a Mamba backbone
2. Prove (or disprove) that Mamba's selective memory can link a reconnaissance ping at step $t=1$ to an exfiltration payload at step $t=100{,}000$
3. Establish the **"Low-and-Slow Stress Test"** — a new benchmark that the entire NIDS community currently lacks

If Mamba succeeds → you publish a landmark positive result.
If Mamba fails → you publish a high-impact negative result proving the community's theoretical assumptions are wrong.

**Either outcome is publishable at TIFS.**

---

## Part 1: Literature Survey (Week 1–2)

### 1.1 What You Must Read

You already have the [Mamba for Long-Context NIDS Novelty.txt](file:///Users/nguyen/Documents/Work/IEEE_Conference_Template_ver_2%20%281%29%2014.34.42/Mamba%20for%20Long-Context%20NIDS%20Novelty.txt) document. Below is the structured reading list organized by priority.

#### Tier 1: Must-Read (Your Direct Competitors)
| Paper | Why It Matters | What to Extract |
|-------|---------------|-----------------|
| **MeeDet** (MDPI Electronics 2026) | Closest competitor. 12-layer Mamba for IIoT APT detection | Their truncation flaw: "first $N$ packets" per flow. Your Table 1 comparison target |
| **Norns** (arXiv 2026) | Multi-modal Mamba for NIDS | Their flaw: uses summarized NetFlow, not raw packets. Their 99.98% PR-AUC baseline |
| **NetMamba** (arXiv 2024) | Pre-trained Mamba for encrypted traffic | Their flaw: 5–10 packets max, converted to 28×28 images. Destroys temporality |
| **Mamba-ECANet** (ResearchGate 2024) | Mamba + channel attention for NIDS | Their flaw: tested on legacy datasets (NSL-KDD), short sequences |
| **Manet** (ResearchGate 2025) | Hybrid Mamba-Attention for malicious traffic | Their flaw: "fixed-length interception" preprocessing |

#### Tier 2: Must-Read (The Mamba Architecture)
| Paper | Why It Matters |
|-------|---------------|
| **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023) | The original Mamba paper. You MUST understand the selective scan mechanism, HiPPO initialization, and hardware-aware algorithm |
| **Mamba-2** (Dao & Gu, 2024) | The improved version with structured state space duality. Check if it offers benefits for your use case |
| **S4: Efficiently Modeling Long Sequences with Structured State Spaces** (Gu et al., 2022) | The predecessor. Understand why S4's static parameters fail for content-aware filtering |

#### Tier 3: Must-Read (APT Datasets & Threat Models)
| Paper | Why It Matters |
|-------|---------------|
| **DAPT2020** (referenced in your research) | APT dataset with 45–75 minute attack intervals. Potential primary dataset |
| **CICAPT-IIoT** (arXiv 2024, ref 47 in your doc) | Provenance-based APT dataset for IIoT. Check if it provides raw PCAPs |
| **XAPT** (PMC 2025, ref 46 in your doc) | Explainable APT prediction. Check their evaluation methodology |
| **CIC-IDS2017/2018 PCAPs** | These datasets provide raw PCAP files. Your potential data source for packet-level sequences |

#### Tier 4: Should-Read (Alternatives & Context)
| Paper | Why It Matters |
|-------|---------------|
| **AutoGMM-RWKV** | RWKV for WSN anomaly detection. Your "adjacent architecture" baseline |
| **RetNet for CPS Anomaly** | RetNet with decay masks. Proves decay-based forgetting hurts APT detection |
| **Deep Learning for Contextualized NetFlow-Based NIDS** (arXiv 2026) | Comprehensive survey. Extract their temporal causality enforcement methodology |

### 1.2 What to Write Down While Reading

For every paper, fill out this extraction template:

```
Paper: [Title]
Year: [Year]
Venue: [Journal/Conference]
Architecture: [Model name]
Sequence Length Tested: [Exact number]
Data Modality: [Raw PCAP / Flow records / Images / Headers]
Attack Types Evaluated: [List]
Datasets Used: [List]
Key Limitation: [Why it doesn't solve ultra-long context]
Quotable Sentence: [A sentence you can cite to prove the gap]
```

### 1.3 The Related Work Section Structure

Your Related Work should have exactly 4 subsections:

1. **GNN and Transformer-Based NIDS** — cite your own GraphTPA, E-GraphSAGE, etc. Establish that spatial models exist but ignore temporal depth
2. **State Space Models for Sequence Modeling** — S4 → Mamba → Mamba-2 progression. Mathematical background
3. **Mamba in Network Security** — MeeDet, Norns, NetMamba, Manet. Systematically expose the "Truncation Fallacy"
4. **APT Detection and Long-Horizon Threat Models** — DAPT2020, CICAPT-IIoT, the "low-and-slow" behavioral model. Prove no benchmark tests $L > 1{,}000$

---

## Part 2: Dataset Engineering (Week 2–4)

> [!CAUTION]
> This is the hardest and most critical phase. Your dataset strategy will determine whether the paper succeeds or fails. Do not rush this.

### 2.1 The Dataset Dilemma

No existing NIDS dataset provides:
- Raw, chronologically ordered packet sequences at $L \geq 100{,}000$
- Labeled APT events with known temporal separation between stages
- Both benign background noise and multi-stage attack injections

**You must build your own evaluation pipeline.** This is not unusual for a TIFS paper — it's expected when you're defining a new benchmark.

### 2.2 Recommended Dataset Strategy (Two-Track)

#### Track A: Flow-Level Chronological Sequences (Primary — Lower Risk)

**Concept:** Take an existing large-scale flow dataset, sort ALL flows by timestamp, and feed the entire chronological stream as one ultra-long sequence.

**Steps:**
1. Download **NF-ToN-IoT** or **NF-BoT-IoT-v2** (you already have these from GraphTPA)
2. Sort all flow records strictly by `Timestamp` (not by source IP or flow ID)
3. Each flow record becomes one "token" in your sequence
4. The sequence length $L$ = total number of flows in chronological order
5. For NF-ToN-IoT: $L \approx 22.3\text{M}$ flows. Sub-sample or window as needed

**Token Features (per flow):**
```python
features = [
    # Network identifiers (embedded, not raw)
    hash(src_ip) % vocab_size,     # Learnable embedding
    hash(dst_ip) % vocab_size,     # Learnable embedding  
    src_port,                       # Normalized
    dst_port,                       # Normalized
    protocol,                       # One-hot or embedded
    
    # Temporal features (CRITICAL for long-context)
    inter_arrival_time,             # Time since previous flow
    log_duration,                   # Flow duration
    
    # Volume features
    log_total_fwd_packets,
    log_total_bwd_packets,
    log_total_bytes,
    
    # Behavioral features
    tcp_flags_encoded,              # Bitwise encoding
    flow_iat_mean,
    flow_iat_std,
    
    # ... (select ~20-30 most discriminative features)
]
# Project to d-dimensional embedding via linear layer
```

**The "Low-and-Slow" Injection Protocol:**
1. Take a long benign-only subsequence of length $L$ (e.g., $L = 100{,}000$)
2. At position $t_1$ (near the start), inject a synthetic "reconnaissance" flow with specific signature features (e.g., unusual port scan pattern)
3. At position $t_2 = t_1 + \Delta$, inject a synthetic "exfiltration" flow with correlated signature features
4. Vary $\Delta \in \{100, 1000, 10000, 50000, 100000\}$
5. The model must correctly flag BOTH flows as part of the same APT campaign, proving it can link events separated by $\Delta$ benign flows

> [!IMPORTANT]
> The injection must be subtle. Do NOT inject obvious volumetric anomalies. The reconnaissance flow should look nearly identical to benign traffic except for specific feature crosses (e.g., a specific [port, protocol, flag] combination that only appears in the recon+exfil pair). This tests Mamba's ability to selectively remember a subtle signal across massive temporal gaps.

#### Track B: Packet-Level Raw PCAP Sequences (Secondary — Higher Risk, Higher Reward)

**Concept:** Use raw PCAP files from CIC-IDS2017/2018, extract packet headers chronologically, and feed them as ultra-long sequences.

**Steps:**
1. Download CIC-IDS2017 PCAP files from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Parse each PCAP using `scapy` or `dpkt`
3. Extract per-packet feature vectors (see below)
4. Maintain strict chronological order across ALL packets (not per-flow)
5. Label packets using the provided ground-truth CSV

**Packet Token Features:**
```python
packet_features = [
    # Layer 3
    ip_src_hash,          # Hashed and embedded
    ip_dst_hash,          # Hashed and embedded
    ip_ttl,               # Normalized
    ip_len,               # Normalized
    ip_proto,             # Embedded
    
    # Layer 4 (TCP/UDP)
    src_port,             # Normalized
    dst_port,             # Normalized
    tcp_flags,            # Bitwise: SYN, ACK, FIN, RST, PSH, URG
    tcp_window_size,      # Normalized
    
    # Temporal
    inter_packet_time,    # Delta from previous packet (log-scaled)
    
    # Total: ~12-15 features per packet → embed to d dimensions
]
```

**Why This Track is Riskier:**
- Parsing millions of packets from PCAPs is slow and storage-intensive
- Labeling individual packets (not flows) requires careful alignment with ground-truth CSVs
- The CIC datasets don't contain multi-month APT campaigns, so you still need synthetic injection

**Recommendation:** Start with Track A. Add Track B only if Track A produces strong results and you have time.

### 2.3 The Synthetic APT Event Pair Generator

This is the core of your novel benchmark. You need to build a Python module that:

```python
class LowAndSlowInjector:
    """
    Injects correlated APT event pairs into a benign traffic sequence.
    
    The pair consists of:
    1. A "Reconnaissance" event (subtle port scan, DNS query, etc.)
    2. An "Action" event (data exfiltration, C2 beacon, lateral movement)
    
    Both events share a hidden correlation signature that the model 
    must learn to detect across massive temporal gaps.
    """
    
    def inject(self, benign_sequence, delta, attack_type):
        """
        Args:
            benign_sequence: Tensor of shape [L, d] — the clean traffic
            delta: int — temporal gap between recon and action events
            attack_type: str — one of ['exfiltration', 'c2_beacon', 'lateral_movement']
        
        Returns:
            injected_sequence: Tensor of shape [L, d] — with APT pair inserted
            labels: Tensor of shape [L] — 0=benign, 1=recon, 2=action
            pair_metadata: dict — {t1, t2, delta, attack_type}
        """
        t1 = random.randint(100, L // 4)        # Recon near the start
        t2 = t1 + delta                          # Action after delta steps
        
        recon_flow = self.generate_recon(attack_type)
        action_flow = self.generate_action(attack_type, recon_flow)
        
        injected_sequence[t1] = recon_flow
        injected_sequence[t2] = action_flow
        labels[t1] = 1  # Reconnaissance
        labels[t2] = 2  # Action/Exfiltration
        
        return injected_sequence, labels, {'t1': t1, 't2': t2, 'delta': delta}
```

### 2.4 Dataset Splits

| Split | Purpose | Sequence Lengths |
|-------|---------|-----------------|
| **Train** | Standard training | $L \in \{10\text{K}, 50\text{K}\}$ with $\Delta \in \{100, 1000, 5000\}$ |
| **Val** | Hyperparameter tuning | $L = 50\text{K}$ with $\Delta \in \{1000, 5000, 10000\}$ |
| **Test-Standard** | Baseline comparison | $L = 50\text{K}$ with $\Delta \in \{1000, 5000, 10000\}$ |
| **Test-Stress** | The novel benchmark | $L = 100\text{K}+$ with $\Delta \in \{10000, 50000, 100000\}$ |

---

## Part 3: Architecture Design (Week 3–5)

### 3.1 The MambaFlow-IDS Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MambaFlow-IDS                       │
│                                                      │
│  ┌──────────────┐                                    │
│  │ Raw Flows /   │  Each flow/packet → feature vector│
│  │ Packets       │  of ~20-30 raw features           │
│  └──────┬───────┘                                    │
│         │                                            │
│  ┌──────▼───────┐                                    │
│  │  Embedding    │  Linear(d_input, d_model)          │
│  │  Layer        │  + LayerNorm + GELU                │
│  │  + Temporal   │  + Relative Time Encoding          │
│  │  Encoding     │                                    │
│  └──────┬───────┘                                    │
│         │  Shape: [B, L, d_model]                    │
│  ┌──────▼───────┐                                    │
│  │  Mamba Block  │  × N layers (N = 8–16)            │
│  │  Stack        │  Each: SSM + Conv1d + SiLU gate    │
│  │               │  With residual + LayerNorm         │
│  └──────┬───────┘                                    │
│         │  Shape: [B, L, d_model]                    │
│  ┌──────▼───────┐                                    │
│  │  Detection    │  Per-token MLP classifier          │
│  │  Head         │  d_model → d_model/2 → C classes   │
│  └──────┬───────┘                                    │
│         │  Shape: [B, L, C]                          │
│  ┌──────▼───────┐                                    │
│  │  Output       │  Per-token softmax → class label   │
│  │  (Per-Token)  │  0=Benign, 1=Recon, 2=Attack, ... │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decisions

#### Decision 1: Per-Token vs. Per-Sequence Classification

- **Per-Token (Recommended):** Classify every flow/packet individually. This naturally handles the "label both the recon ping and the exfiltration payload" requirement
- **Per-Sequence:** Classify the entire sequence as "contains APT" or "clean." Simpler but doesn't localize the attack steps

> Use per-token classification. It's harder but more impressive and more useful operationally.

#### Decision 2: Temporal Encoding

Standard positional encoding (sinusoidal or learned) encodes *position index*, not *real time*. But in network traffic, the temporal gap between packets varies wildly (microseconds during a burst, minutes during idle periods).

**Use Relative Temporal Encoding:**
```python
class RelativeTimeEncoding(nn.Module):
    """Encodes the actual time delta between consecutive events."""
    def __init__(self, d_model):
        self.time_proj = nn.Linear(1, d_model)
    
    def forward(self, timestamps):
        # timestamps: [B, L] — absolute timestamps
        deltas = timestamps[:, 1:] - timestamps[:, :-1]  # Inter-arrival times
        deltas = torch.log1p(deltas)  # Log-scale to handle huge variance
        deltas = F.pad(deltas, (1, 0), value=0)  # Pad first position
        return self.time_proj(deltas.unsqueeze(-1))  # [B, L, d_model]
```

Add this to the embedding: `x = feature_embed(x) + time_encode(timestamps)`

#### Decision 3: Mamba Configuration

```python
mamba_config = {
    'd_model': 256,          # Hidden dimension
    'n_layers': 12,          # Number of Mamba blocks
    'd_state': 16,           # SSM state dimension (N in Mamba paper)
    'd_conv': 4,             # Local convolution width
    'expand': 2,             # Inner dimension expansion factor
    'dt_min': 0.001,         # Minimum step size
    'dt_max': 0.1,           # Maximum step size
}
# Total params ≈ 15-25M (much smaller than LLMs, appropriate for NIDS)
```

#### Decision 4: Handling Class Imbalance

Your sequence is 99.9% benign tokens. You MUST handle this:
- **Focal Loss** ($\gamma = 3.0$) — same as GraphTPA, proven effective
- **Class-frequency weighting** — weight rare classes proportionally
- **Hard negative mining** — oversample sequences containing attack tokens during training

### 3.3 Implementation Stack

```
Framework:        PyTorch 2.x
Mamba Library:    mamba-ssm (pip install mamba-ssm)  — Official Tri Dao implementation
                  OR causal-conv1d + selective-scan-cuda (manual)
Data Loading:     PyTorch DataLoader with custom SequenceDataset
Experiment Track: Weights & Biases (wandb)
Hardware:         1× NVIDIA A100 (40GB) or 1× RTX 4090 (24GB)
```

### 3.4 Code Structure

```
mambaflow-ids/
├── configs/
│   ├── model.yaml          # Architecture hyperparameters
│   ├── data.yaml            # Dataset paths, sequence lengths
│   └── train.yaml           # Training hyperparameters
├── data/
│   ├── preprocessor.py      # Raw data → chronological token sequences
│   ├── injector.py          # LowAndSlowInjector class
│   ├── dataset.py           # PyTorch Dataset/DataLoader
│   └── tokenizer.py         # Flow/Packet → feature vector
├── models/
│   ├── mambaflow.py         # Main MambaFlow-IDS architecture
│   ├── temporal_encoding.py # Relative time encoding
│   ├── detection_head.py    # Per-token classification MLP
│   └── baselines/
│       ├── transformer_ids.py  # Transformer baseline
│       ├── lstm_ids.py         # LSTM baseline
│       └── truncated_mamba.py  # MeeDet-style truncated Mamba
├── train.py                 # Training loop
├── evaluate.py              # Evaluation + stress test
├── stress_test.py           # The "Low-and-Slow Stress Test" runner
└── visualize.py             # Figures for the paper
```

---

## Part 4: Experiments (Week 5–10)

### 4.1 The Five Experiments You Must Run

#### Experiment 1: Standard NIDS Benchmark (Table 1)
**Purpose:** Prove MambaFlow-IDS is competitive on standard datasets at standard sequence lengths.

| Dataset | Sequence Length | Metric | Baselines |
|---------|----------------|--------|-----------|
| NF-ToN-IoT | $L = 10\text{K}$ | Macro F1, Accuracy | E-GraphSAGE, MeeDet, LSTM, Transformer |
| NF-BoT-IoT | $L = 10\text{K}$ | Macro F1, Accuracy | Same |
| CIC-IDS2018 | $L = 10\text{K}$ | Macro F1, Accuracy | Same |

> [!NOTE]
> This experiment alone is NOT your contribution. It's the **entry ticket** that proves your model works before you run the novel stress tests. If MambaFlow fails here, debug before proceeding.

#### Experiment 2: The Low-and-Slow Stress Test (Table 2 + Figure — THE KEY CONTRIBUTION)

**Purpose:** The paper's flagship experiment. Test whether each architecture can detect correlated APT events as the temporal gap $\Delta$ increases.

| $\Delta$ (gap) | MambaFlow | Transformer | LSTM | Truncated-Mamba (MeeDet-style) |
|------|-----------|-------------|------|------|
| 100 | ? | ? | ? | ? |
| 1,000 | ? | ? | ? | ? |
| 5,000 | ? | ? | ? | ? |
| 10,000 | ? | ? (OOM?) | ? | ? |
| 50,000 | ? | OOM | ? | ? |
| 100,000 | ? | OOM | ? | ? |

**Expected Results:**
- **Transformer:** Should OOM at $\Delta \geq 10\text{K}$ (quadratic memory)
- **LSTM:** Should work at all $\Delta$ but accuracy degrades to random at $\Delta \geq 5\text{K}$ (catastrophic forgetting)
- **Truncated-Mamba:** Should work at all $\Delta$ but accuracy drops at $\Delta > \text{truncation window}$ (information destroyed by preprocessing)
- **MambaFlow:** Should maintain high accuracy across ALL $\Delta$ values (the hypothesis)

**The Killer Figure:**
Plot a line chart: X-axis = $\Delta$ (log scale), Y-axis = Detection F1-Score.
- Transformer line crashes to 0 (OOM)
- LSTM line gradually decays to random
- Truncated-Mamba line drops after truncation boundary
- **MambaFlow line stays flat** ← This is the money shot

#### Experiment 3: Memory & Latency Profiling (Table 3 + Figure)

**Purpose:** Prove that MambaFlow achieves linear scaling in practice, not just theory.

```python
sequence_lengths = [1000, 5000, 10000, 50000, 100000, 500000]

for L in sequence_lengths:
    # Measure:
    peak_gpu_memory_mb = measure_gpu_memory(model, L)
    inference_time_ms = measure_inference_time(model, L)
    throughput_packets_per_sec = L / inference_time_ms * 1000
```

Plot two figures:
1. **GPU Memory vs. Sequence Length:** Mamba = linear line, Transformer = quadratic explosion
2. **Inference Latency vs. Sequence Length:** Same pattern

#### Experiment 4: Ablation Study (Table 4)

| Component | Macro F1 | Stress Test F1 ($\Delta$=50K) |
|-----------|----------|------|
| Full MambaFlow-IDS | **X** | **X** |
| − Relative Time Encoding (use positional) | ? | ? |
| − Selective Scan (use S4 static params) | ? | ? |
| − Deep stack (use 4 layers instead of 12) | ? | ? |
| Replace Mamba with Transformer (short $L$) | ? | — |
| Replace Mamba with LSTM | ? | ? |

**Expected insight:** Removing the selective scan mechanism should cause the most dramatic failure on the stress test, proving that Mamba's content-aware gating is essential for long-range APT correlation.

#### Experiment 5: State Probing — Visualizing What Mamba Remembers (Figure — High Impact)

**Purpose:** Prove that Mamba's hidden state actually retains information about the reconnaissance event across 100K benign tokens.

**Method:**
1. Feed a sequence with a reconnaissance event at $t_1 = 500$ and exfiltration at $t_2 = 100{,}500$
2. At every 1,000 steps, extract the Mamba hidden state $\mathbf{h}_t$
3. Train a simple linear probe: can $\mathbf{h}_t$ predict "a recon event has already occurred"?
4. Plot: X-axis = step $t$ (from $t_1$ to $t_2$), Y-axis = probe accuracy

**Expected result:** The probe accuracy should remain high (>80%) even at $t = 100{,}000$, proving Mamba has NOT forgotten the reconnaissance event. For LSTM, the probe accuracy should decay to 50% (random) rapidly.

This is a **stunning visualization** that will captivate reviewers.

### 4.2 Baseline Implementations

You need 4 baselines. Here's how to build each:

| Baseline | Implementation Strategy |
|----------|------------------------|
| **Transformer-IDS** | Standard multi-head self-attention encoder. Will OOM at $L > 10\text{K}$ on 40GB GPU. This is the point |
| **LSTM-IDS** | 4-layer bidirectional LSTM with same hidden dim. Will run at all $L$ but forget |
| **Truncated-Mamba** | Your own MambaFlow but with MeeDet-style preprocessing: truncate to first 128 tokens per flow, reset context between flows |
| **S4-IDS** | Replace Mamba blocks with S4 blocks (static parameters). Tests whether selective scan matters |

### 4.3 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Macro F1-Score** | Primary metric. Same as GraphTPA. Handles class imbalance |
| **Pair Detection Rate (PDR)** | **Your novel metric.** "Of all injected APT event pairs, what fraction had BOTH the recon AND action events correctly detected?" This is the metric that directly measures long-range correlation |
| **Memory (GB)** | Peak GPU memory during inference |
| **Latency (ms)** | End-to-end inference time for one sequence |
| **Throughput (packets/sec)** | Operational viability metric |

> [!IMPORTANT]
> **Pair Detection Rate is your paper's unique metric.** No other NIDS paper uses it because no other paper tests correlated event pairs separated by massive temporal gaps. Define it formally in Section IV.

---

## Part 5: Writing the Paper (Week 8–12)

### 5.1 Paper Structure (IEEE TIFS Format, ~13 pages)

```
I.    Introduction                              (~1.5 pages)
II.   Related Work                              (~2 pages)
III.  Problem Formulation                       (~1 page)
IV.   Proposed Architecture: MambaFlow-IDS      (~2.5 pages)
V.    The Low-and-Slow Stress Test Benchmark    (~1 page)
VI.   Experimental Setup                        (~1 page)
VII.  Results and Analysis                      (~3 pages)
VIII. Discussion                                (~0.5 page)
IX.   Conclusion                                (~0.5 page)
      References                                (~1 page)
```

### 5.2 Section-by-Section Writing Guide

#### I. Introduction

**Paragraph 1:** The APT problem. APTs are multi-stage, span months, "low-and-slow." Cite Microsoft/Okta threat reports.

**Paragraph 2:** Why current NIDS fails. LSTMs forget. Transformers OOM. Everyone truncates. The "Truncation Fallacy" — one sentence summary.

**Paragraph 3:** Mamba's promise. Linear $\mathcal{O}(L)$ complexity, selective scan, theoretically infinite context. But nobody has empirically tested this on sequences $L > 1{,}000$.

**Paragraph 4:** Your contribution. Three bullet points:
1. **MambaFlow-IDS:** First architecture to feed $L = 100\text{K}+$ chronologically ordered network events into a selective SSM for per-token intrusion classification
2. **The Low-and-Slow Stress Test:** First benchmark that measures detection accuracy as a function of temporal gap $\Delta$ between correlated APT stages
3. **Empirical validation:** Prove (with state probing) that Mamba's selective scan retains reconnaissance signatures across $100\text{K}$ benign tokens without catastrophic forgetting

#### III. Problem Formulation

Define formally:
- Chronological traffic stream $\mathcal{S} = (x_1, x_2, \ldots, x_L)$ where $x_t \in \mathbb{R}^d$ and $L \gg 10{,}000$
- APT event pair $(e_{\text{recon}}, e_{\text{action}})$ at positions $(t_1, t_2)$ where $\Delta = t_2 - t_1 \gg 1{,}000$
- Goal: $f_\theta(x_t | x_1, \ldots, x_{t-1}) \rightarrow y_t \in \{0, 1, \ldots, C\}$ for all $t$
- **Pair Detection Rate:** $\text{PDR} = \frac{|\{(t_1, t_2) : \hat{y}_{t_1} \neq 0 \wedge \hat{y}_{t_2} \neq 0\}|}{|\text{All injected pairs}|}$

#### V. The Low-and-Slow Stress Test Benchmark (CRITICAL — Your Novel Contribution)

This section defines a **reusable benchmark** that other researchers can adopt. Describe:
1. The injection protocol (how APT pairs are generated)
2. The $\Delta$-schedule (100 → 100K)
3. The evaluation protocol (PDR at each $\Delta$)
4. Why existing datasets cannot test this
5. How to reproduce the benchmark (release the code)

> [!TIP]
> By framing the stress test as a **reusable, open-source benchmark**, you increase citation potential dramatically. Other papers will cite your benchmark even if they use different models.

### 5.3 Figures You Need to Produce

| Figure | Content | Tool |
|--------|---------|------|
| Fig. 1 | Architecture diagram of MambaFlow-IDS | draw.io or TikZ |
| Fig. 2 | The "Truncation Fallacy" illustration — showing how MeeDet/NetMamba truncate vs. your full-context approach | TikZ |
| Fig. 3 | **The Stress Test Plot** — Detection F1 vs. $\Delta$ for all architectures | matplotlib |
| Fig. 4 | GPU Memory scaling — linear (Mamba) vs. quadratic (Transformer) | matplotlib |
| Fig. 5 | **State Probe Visualization** — Mamba retains recon signal, LSTM forgets | matplotlib |
| Fig. 6 | t-SNE of hidden states at key timesteps | matplotlib + sklearn |

### 5.4 Tables You Need to Produce

| Table | Content |
|-------|---------|
| Table 1 | Standard NIDS benchmark results (Macro F1 across 3 datasets) |
| Table 2 | **The Stress Test Results** — PDR at each $\Delta$ for all architectures |
| Table 3 | Memory and latency profiling at various $L$ |
| Table 4 | Ablation study |
| Table 5 | Comparison with existing Mamba-NIDS approaches (sequence length, attacks, modality) — extracted from your literature review |

---

## Part 6: Risk Mitigation

### Risk 1: Mamba Fails the Stress Test (Negative Result)

**Probability:** Medium (30%)

**If this happens:** Your paper becomes "Challenging the Infinite Context Hypothesis: An Empirical Study of Selective State Space Models for Long-Horizon Network Anomaly Detection." Negative results are publishable at TIFS if the experimental methodology is rigorous. You prove that the community's theoretical assumptions about Mamba's selective memory are incorrect under realistic conditions. This is still a high-impact finding.

**Mitigation:** Before testing at $\Delta = 100\text{K}$, first verify at $\Delta = 1{,}000$ and $\Delta = 5{,}000$. If Mamba fails at $\Delta = 5{,}000$, you still have a story: "Mamba outperforms LSTM and Transformers but has a practical context limit of ~X thousand tokens."

### Risk 2: Synthetic APT Injection is Unrealistic

**Probability:** Medium (30%)

**Reviewer attack:** "Your injected APT events are artificial and don't represent real APT campaigns."

**Mitigation:**
- Use the DAPT2020 dataset which has realistic APT timing (45–75 min intervals)
- Ground your injection protocol in documented APT frameworks (MITRE ATT&CK stages)
- Cite real-world APT reports (SolarWinds, NotPetya) showing attackers use exactly these temporal gaps

### Risk 3: The Model Detects APT Events by Local Features, Not Long-Range Context

**Probability:** Medium (25%)

**Reviewer attack:** "Your model detects the recon and exfil events independently by their local features, not by linking them across the temporal gap."

**Mitigation:** This is exactly why Experiment 5 (State Probing) exists. If the hidden state at $t = 50{,}000$ can predict "a recon event occurred at $t = 500$," then the model IS maintaining long-range memory. Additionally, design your injected events so that the exfiltration event looks benign in isolation — it can only be classified as malicious if the model remembers the preceding reconnaissance.

### Risk 4: Computational Infeasibility

**Probability:** Low (15%)

**If a single A100 (40GB) cannot handle $L = 100\text{K}$:** Use gradient checkpointing, mixed precision (bfloat16), and sequence-parallel training. Mamba's SRAM-optimized kernel should handle this, but test early.

---

## Part 7: Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1–2 | Literature Survey | Related work notes, extraction templates filled, reference list finalized |
| 2–4 | **Dataset Engineering** | Preprocessor pipeline, LowAndSlowInjector, train/val/test splits ready |
| 3–5 | **Architecture Implementation** | MambaFlow-IDS code, all 4 baselines implemented, sanity checks pass |
| 5–7 | **Experiment 1 & 2** | Standard benchmarks (Table 1) + Stress Test (Table 2, Fig. 3) |
| 7–9 | **Experiments 3, 4, 5** | Memory profiling, ablations, state probing visualization |
| 8–12 | **Paper Writing** | Full manuscript draft, all figures and tables |
| 12–14 | **Revision & Polish** | Professor review, address feedback, finalize |
| 14+ | **Submission** | Submit to IEEE TIFS |

---

## Part 8: Checklist Before Submission

- [ ] All 5 experiments completed with 10 random seeds and confidence intervals
- [ ] The Stress Test figure clearly shows MambaFlow maintaining accuracy at $\Delta = 100\text{K}$
- [ ] State probe visualization proves Mamba retains long-range memory
- [ ] Ablation proves selective scan (not just Mamba architecture) is the critical component
- [ ] Code and benchmark publicly released on GitHub (TIFS reviewers value reproducibility)
- [ ] Paper is within 13-page TIFS limit
- [ ] Professor has reviewed and approved
- [ ] All claims are mathematically precise (learned from GraphTPA review)
- [ ] No "fluffy" language — every claim is backed by an equation or an experiment
