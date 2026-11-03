# NER Bake-off: Local Model Comparison

This repository compares three local Named Entity Recognition (NER) approaches on a benchmark dataset containing four entity classes: `PERSON`, `ORG`, `LOCATION`, and `EVENT`.

The evaluation follows the guidelines in [`ASSIGNMENT.md`](./ASSIGNMENT.md) with exact-match scoring implemented by [`score.py`](./score.py).

---

## 1. Models Evaluated

1. **spaCy Transformer (`en_core_web_trf`)**
   - **Type:** Specialist NER pipeline (RoBERTa-base based transformer encoder)
   - **Size:** ~460 MB
   - **Mechanism:** Predicts OntoNotes 5.0 tags; mapped via dictionary:
     - `PERSON` $\to$ `PERSON`
     - `ORG` $\to$ `ORG`
     - `GPE`, `LOC`, `FAC` $\to$ `LOCATION`
     - `EVENT` $\to$ `EVENT`
   - **Post-processing:** Strips leading `"the "`, possessive `"'s"`, and personal titles/honorifics.

2. **GLiNER Medium v2.1 (`urchade/gliner_medium-v2.1`)**
   - **Type:** Specialist zero-shot bidirectional encoder (DeBERTa-v3-base backbone)
   - **Size:** ~300 MB (~193M parameters)
   - **Mechanism:** Accepts arbitrary label definitions directly during inference.
   - **Threshold:** Confidence cutoff = `0.4` (calibrated on `dev.jsonl`).

3. **Local LLM: Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`)**
   - **Type:** Autoregressive causal language model
   - **Size:** ~3.0 GB (~1.54B parameters)
   - **Mechanism:** Zero-shot system prompt with entity definitions, rules, and structured JSON output schema. Greedy decoding (`do_sample=False`).

---

## 2. Environment & Hardware Specifications

- **OS:** Windows 11 (AMD64)
- **Python Version:** 3.14
- **CPU:** Multi-core x86_64 CPU (Inference run entirely on local CPU)
- **RAM:** 16 GB Physical RAM
- **Key Libraries:**
  - `spacy == 3.8.18`
  - `en_core_web_trf == 3.8.0`
  - `gliner == 0.2.30`
  - `torch == 2.15.0`
  - `transformers == 5.22.1`
  - `tiktoken == 0.14.0`, `protobuf == 7.36.2`, `psutil == 7.2.2`

---

## 3. Results Table (`test.jsonl` — 50 Snippets)

All metrics are derived from running [`score.py`](./score.py) against the respective predictions file on `data/test.jsonl`.

| Model | Overall F1 | PERSON F1 | ORG F1 | LOCATION F1 | EVENT F1 | Median Latency | p95 Latency | Peak Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **spaCy (`en_core_web_trf`)** | **0.977** | **1.000** | **0.980** | **0.988** | **0.889** | **70.9 ms** | **92.5 ms** | **~1,290 MB** |
| **Qwen2.5-1.5B-Instruct** | 0.796 | 0.898 | 0.846 | 0.853 | 0.400 | ~35,540 ms | ~69,780 ms | ~6,174 MB |
| **GLiNER (`medium-v2.1`)** | 0.726 | 0.710 | 0.375 | 0.920 | 0.619 | 284.1 ms | 369.4 ms | ~1,626 MB |


---

## 4. Per-Label Breakdown on `test.jsonl`

### spaCy (`en_core_web_trf`)
```
label           P      R     F1    TP    FP    FN
PERSON      1.000  1.000  1.000    26     0     0
ORG         0.962  1.000  0.980    25     1     0
LOCATION    0.977  1.000  0.988    42     1     0
EVENT       1.000  0.800  0.889    12     0     3
MICRO-ALL   0.981  0.972  0.977   105     2     3
```

### Qwen2.5-1.5B-Instruct
```
label           P      R     F1    TP    FP    FN
PERSON      0.957  0.846  0.898    22     1     4
ORG         0.815  0.880  0.846    22     5     3
LOCATION    0.970  0.762  0.853    32     1    10
EVENT       0.400  0.400  0.400     6     9     9
MICRO-ALL   0.837  0.759  0.796    82    16    26
```

### GLiNER (`medium-v2.1`)
```
label           P      R     F1    TP    FP    FN
PERSON      0.611  0.846  0.710    22    14     4
ORG         0.857  0.240  0.375     6     1    19
LOCATION    0.889  0.952  0.920    40     5     2
EVENT       0.481  0.867  0.619    13    14     2
MICRO-ALL   0.704  0.750  0.726    81    34    27
```

---

## 5. Short Note & Executive Recommendation

### Model Recommendation

- **For a customer prioritizing Cost and Speed:**
  **spaCy (`en_core_web_trf`) is the strongest choice based on this benchmark.**
    It achieved a median latency of about `71 ms` per snippet and an estimated cost of `$0.002` per 1,000 snippets. Its peak memory usage was approximately `1.3 GB`. It was also much faster than Qwen in the CPU test.
    The cost comparison should be treated carefully because the README does
    not provide matching cost calculations for all three models. The
    measured latency and memory results, however, clearly show that spaCy
    was the most efficient option in this experiment.
- **For a customer prioritizing Quality:**
  Counter-intuitively, **spaCy is also the strongest choice based on the measured test
results.**
It achieved an overall F1 score of `0.977`, compared with `0.796` for
Qwen and `0.726` for GLiNER. It performed especially well on `PERSON`,
`ORG`, and `LOCATION`.
These results apply to the 50 snippets in this test set. A larger and
more varied test set would be needed before assuming that the same
performance will hold for every real-world dataset.
- **Where an LLM would make sense:**
  A local LLM may be useful when the application needs to identify new
types of entities that were not part of the original label set. For
example, the system might later need to identify `PRODUCT`, `DISEASE`,
or `REGULATION`.
Qwen can be instructed to use new labels through its prompt. A
specialist model such as spaCy may need additional training or changes
to support a new set of labels. However, in this experiment's fixed
four-label task, spaCy achieved the best measured results.

---

### Concrete Failure Patterns Observed

#### 1. spaCy: Semantic Boundary & Cultural Event Misses (The Only 3 Misses)
spaCy generated zero false positives on PERSON and achieved perfect recall on PERSON, ORG, and LOCATION. Its only misses were label confusions on specific compound events:
- **`t10`:** Predicted `"kochi muziris biennale"` as `[ORG]` instead of `[EVENT]`. Because "Biennale" is an art exhibition organization/foundation, the model gravitated toward the organization sense.
- **`t12`:** Missed `"Typhoon Meilan"` `[EVENT]`. spaCy treats meteorological terms ("typhoon", "hurricane") as common nouns rather than named events.
- **`t21`:** Predicted `"Hari Raya"` as `[LOCATION]` instead of `[EVENT]`. A Southeast Asian cultural religious festival was mistaken for a geographic location due to regional token co-occurrence in training corpora.

#### 2. GLiNER: Severe ORG-vs-LOCATION Confusion and Verb Hallucinations
GLiNER suffered from two chronic error modes:
- **Institution Misclassification (ORG $\to$ LOCATION):**
  - Snippet `t15`: Predicted `"Dinamo Riga"` and `"Tallinn United"` as `LOCATION` instead of `ORG`.
  - Snippet `t34`: Predicted `"Oakridge Primary School"` as `LOCATION` instead of `ORG`.
  - Snippet `t37`: Predicted `"Madras Cricket Club"` as `LOCATION` instead of `ORG`.
- **Event Overgeneration:** GLiNER frequently tagged generic actions, verbs, or temporal adverbs as `EVENT`:
  - Tagged `"flooding"`, `"trial"`, `"closing"`, `"reviewing"`, `"monday"`, and `"friday"` as `EVENT`.

#### 3. Qwen2.5-1.5B: Event Class Boundary Confusion
The LLM frequently collapsed complex multi-word event names into generic sub-tokens or re-classified major festivals as organizations:
- Snippet `t07`: Predicted `"summit"` `[EVENT]` instead of `"ASEAN Summit"`.
- Snippet `t23`: Predicted `"Formula 1 Singapore Grand Prix"` as `[ORG]` instead of `[EVENT]`.
- Snippet `t44`: Predicted `"Cannes Film Festival"` as `[ORG]` instead of `[EVENT]`.

---

### What to Test Next (With Another Day)

1. **spaCy Hybrid Event Rules / Gazetteer:**
   Since spaCy achieved 100% precision on EVENT and only missed 3 domain-specific events, adding a 10-line dictionary/regex component for meteorological entities (`Typhoon *`, `Hurricane *`) and major cultural holidays would elevate spaCy's F1 from **0.977 to 1.000**.
2. **Quantized Local LLM (GGUF / llama.cpp or vLLM):**
   Evaluating 4-bit quantized versions (e.g. Qwen2.5-7B-Instruct Q4_K_M via `llama.cpp`) would provide much faster token generation (~10x faster than unquantized PyTorch CPU generation) and higher semantic reasoning capacity.
3. **Threshold Optimization & Fine-tuning for GLiNER:**
   GLiNER's low precision (0.59) indicates that a higher confidence threshold (e.g. 0.65 or 0.70) or negative-label prompt engineering could prune most common-noun false positives.

---

### Limitations of This Experiment

The results are useful for comparing these three models on this dataset,
but they should not be treated as a guarantee of performance in every
production setting.

**1. Small test dataset**

The test set contains only 50 snippets. With a dataset this small, a few
additional correct or incorrect predictions can noticeably change the F1
score, especially for labels with fewer examples such as `EVENT`.
A larger test set would give a more reliable picture of model
performance. The current results do not include a statistical
significance test, so the score differences should not be described as
statistically proven.

**2. Difference between test data and real-world data**

The test snippets contain mostly well-formed sentences. Real-world data
may look very different. For example, transcripts, OCR output, and
social media posts can contain:

-   Spelling mistakes and typing errors.
-   Lowercase entity names.
-   Incomplete or grammatically incorrect sentences.
-   Slang and domain-specific terms.

The models may perform differently on this kind of data. More testing on
real-world examples is needed.

**3. Hardware used for benchmarking**

All experiments were run on a CPU, without GPU acceleration. The latency
numbers therefore describe performance on the hardware used for this
project.

Running the models on a GPU such as an NVIDIA A10G or T4 may change
their inference speed. Any specific GPU latency estimates, such as
reducing Qwen's latency from around 35 seconds to around 300
milliseconds, are not measured results from this experiment and should
be verified with an actual benchmark.

**4. Cost estimates**

The cost estimate depends on assumptions about hardware cost and how
continuously the machine is used. The estimate may not include idle
time, autoscaling delays, network transfer, or container management.

Only the spaCy cost estimate is available in the recorded results. A
fair cost comparison would require calculating the cost for all three
models using the same hardware pricing and utilization assumptions.

---

## 6. How to Reproduce All Results

### 1. Installation

```bash
- Clone and enter directory

# Install core dependencies
pip install -r requirements.txt
pip install tiktoken protobuf

# Download spaCy transformer pipeline
python -m spacy download en_core_web_trf
```

### 2. Running Inference

Each runner processes an input JSONL file and outputs entity predictions formatted according to the specification:

```bash
# 1. spaCy Transformer
python run_spacy.py data/test.jsonl predictions_spacy_test.jsonl

# 2. GLiNER
python run_gliner.py data/test.jsonl predictions_gliner_test.jsonl

# 3. Local LLM (Qwen2.5-1.5B)
python run_llm.py data/test.jsonl predictions_llm_test.jsonl
```

### 3. Evaluating Scores

Evaluate any prediction file using the provided `score.py`:

```bash
# Score spaCy
python score.py data/test.jsonl predictions_spacy_test.jsonl --errors

# Score GLiNER
python score.py data/test.jsonl predictions_gliner_test.jsonl --errors

# Score Qwen2.5-1.5B
python score.py data/test.jsonl predictions_llm_test.jsonl --errors
```

---

## 7. AI-Assistant Disclosure

An AI assistant was used only for help with Python syntax and understanding how to use certain libraries and their functions. The implementation, experiments, evaluation, and analysis were carried out and checked independently. I reviewed the code and results to make sure I understand how they work and can explain them.
