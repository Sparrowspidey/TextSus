# `interfaces.md`

# SynthID-Text Research Project --- Member 2 ↔ Member 3 Interface Specification

**Project:** Scalable Watermarking for Identifying Large Language Model
Outputs\
**Primary reference:** Dathathri et al., *Scalable watermarking for
identifying large language model outputs*, Nature, 2024.\
**Supporting project document:** *AI WORK DIVISION --- Final Locked
Three-Member Research Work Division*

------------------------------------------------------------------------

## 1. Purpose

This document defines the technical contract between **Member 2 (Core
Technical Method)** and **Member 3 (Experimental Evaluation,
Scalability, Results & Limitations)**.

The objective is to ensure that Member 3 can build the complete
experimental and evaluation layer directly on Member 2's implementation
without changing the core watermark-generation logic.

The interface is based on the SynthID-Text architecture described in the
research paper:

1.  Random seed generation
2.  Watermark sampling
3.  Watermark scoring
4.  Threshold-based detection

The paper describes these as the three core generative-watermark
components---random seed generator, sampling algorithm, and scoring
function---with detection obtained by applying a threshold to the score.

> **Important:** The function signatures in this document are **project
> implementation contracts**. They are derived from the paper's
> algorithms and terminology; they are not claimed to be official APIs
> published by the authors.

------------------------------------------------------------------------

# 2. Responsibility Boundary

## 2.1 Member 2 --- Core Technical Method

Member 2 owns:

-   Autoregressive LLM next-token distribution
-   Token sampling
-   Temperature, top-k and top-p handling
-   Watermark key handling
-   Sliding-window/random seed generation
-   Pseudorandom watermark functions
-   g-value generation
-   Candidate-token selection
-   Tournament sampling
-   Single-layer and multilayer tournament logic
-   Watermark score calculation
-   Detection threshold application
-   Non-distortionary configuration
-   Distortionary configuration
-   Core generation/detection APIs

Member 2 **must not** implement experiment-specific conclusions or
evaluation plots as part of the core API.

## 2.2 Member 3 --- Experimental Evaluation

Member 3 owns:

-   Experimental configuration
-   Model/dataset execution
-   Baseline comparisons
-   Detection experiments
-   Quality evaluation
-   Entropy/perplexity analysis
-   TPR at fixed FPR
-   Distortionary vs non-distortionary evaluation
-   Latency and computational-overhead measurements
-   Speculative-sampling experiments
-   Result aggregation
-   Tables/plots
-   Interpretation of experimental results
-   Limitations and future-work analysis

Member 3 **must consume Member 2's public interfaces instead of
reimplementing watermarking algorithms.**

------------------------------------------------------------------------

# 3. High-Level Architecture

``` text
                    ┌──────────────────────┐
                    │      LLM / Model     │
                    └──────────┬───────────┘
                               │
                     next-token probabilities
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Member 2 Core      │
                    │   Watermark Engine   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       Seed Generator    G-value Engine    Tournament
              │                │             Sampling
              └────────────────┼────────────────┘
                               │
                               ▼
                       Generated Tokens
                               │
                               ▼
                       Watermark Scorer
                               │
                               ▼
                          Score / Evidence
                               │
                               ▼
                      Threshold Detection
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Member 3        │
                    │ Experimental Layer   │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       Detectability       Quality            Scalability
       / TPR@FPR          / PPL / Entropy     / Latency
```

------------------------------------------------------------------------

# 4. Shared Data Contracts

## 4.1 Watermark Configuration

``` python
from dataclasses import dataclass
from typing import Literal

@dataclass
class WatermarkConfig:
    watermark_key: int | str
    history_length: int = 4
    num_layers: int = 30
    competitors_per_match: int = 2
    distortion_mode: Literal["non_distortionary", "distortionary"] = "non_distortionary"
    repeated_context_masking: bool = True
    masking_window: int = 1
```

### Contract

-   `watermark_key` is the secret key used by the deterministic seed
    generator.
-   `history_length=4` matches the paper's experimental setting for the
    recent-token context used by the seed generator.
-   `num_layers=30` is the paper's general experimental configuration
    unless another value is explicitly required.
-   `competitors_per_match=2` corresponds to the standard
    non-distortionary Tournament configuration.
-   More than two competitors may be used for the distortionary
    configuration.
-   `repeated_context_masking` and `masking_window` control
    repeated-context masking.

------------------------------------------------------------------------

## 4.2 Token Distribution

Member 2 must expose the LLM's next-token distribution in a form
accepted by the sampling interface.

Recommended contract:

``` python
TokenDistribution = dict[int, float]
```

Example:

``` python
{
    101: 0.12,
    205: 0.08,
    310: 0.04,
    ...
}
```

The probabilities should represent:

\[ p\_{LM}(x_t `\mid `{=tex}x\_{\<t}) \]

after any agreed temperature/top-k/top-p processing.

------------------------------------------------------------------------

## 4.3 Generation Result

``` python
@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    watermarked: bool
    watermark_scores: list[float]
    final_score: float | None
    config: WatermarkConfig
```

`final_score` may remain `None` until detection is explicitly performed.

------------------------------------------------------------------------

## 4.4 Detection Result

``` python
@dataclass
class DetectionResult:
    score: float
    threshold: float
    is_watermarked: bool
    confidence: float | None = None
    num_tokens: int = 0
```

The paper's core decision is based on comparing the score with a
threshold.

------------------------------------------------------------------------

# 5. Member 2 Public API

## 5.1 Seed Generation

``` python
def generate_seed(
    context: list[int],
    watermark_key: int | str,
    history_length: int = 4,
) -> int:
    ...
```

### Purpose

Generate a deterministic pseudorandom seed from the recent token context
and watermark key.

### Contract

-   Only the most recent `history_length` tokens are used.
-   The same context + same key must always produce the same seed.
-   Changing the key should produce a different pseudorandom sequence.
-   The function must be deterministic.

The paper describes the seed as being generated using a deterministic
hash of recent token history and the watermark key.

------------------------------------------------------------------------

# 6. Pseudorandom g-Value Interface

## 6.1 Single g-Value

``` python
def g_value(
    token_id: int,
    seed: int,
    layer: int,
) -> float:
    ...
```

### Mathematical contract

The implementation represents:

\[ g\_`\ell`{=tex}(x,r) \]

where:

-   \(x\) = candidate token
-   \(r\) = random seed
-   (`\ell`{=tex}) = tournament layer

The output is a pseudorandom value sampled according to the selected
g-value distribution.

------------------------------------------------------------------------

## 6.2 Vectorized g-Values

For evaluation and scalability experiments, Member 2 should also expose:

``` python
def compute_g_values(
    token_ids: list[int],
    seed: int,
    layer: int,
) -> list[float]:
    ...
```

### Contract

``` text
len(output) == len(token_ids)
```

`output[i]` corresponds to `token_ids[i]`.

This vectorized interface is important because the paper emphasizes
vectorized implementation and negligible additional computational
overhead.

------------------------------------------------------------------------

# 7. Candidate Sampling

``` python
def sample_candidates(
    token_probabilities: dict[int, float],
    num_candidates: int,
    rng_seed: int | None = None,
) -> list[int]:
    ...
```

### Purpose

Draw candidate tokens from the LLM probability distribution before
tournament selection.

### Contract

-   Candidate selection must respect the supplied LLM distribution.
-   The returned candidates must be valid vocabulary token IDs.
-   `num_candidates` must be at least 2 for a tournament.
-   Sampling must be reproducible when `rng_seed` is supplied.

------------------------------------------------------------------------

# 8. Single-Layer Tournament Sampling

``` python
def tournament_sample_layer(
    token_probabilities: dict[int, float],
    seed: int,
    layer: int,
    competitors_per_match: int = 2,
) -> int:
    ...
```

### Purpose

Perform one tournament layer and return the winning candidate token.

### Standard configuration

``` text
competitors_per_match = 2
```

For each match, the candidates compete according to their g-values. The
candidate with the larger g-value wins.

The paper's single-layer algorithm samples candidates from the LLM
distribution and retains the candidate(s) with the maximum g-value, with
uniform sampling among ties.

------------------------------------------------------------------------

# 9. Multilayer Tournament Sampling

``` python
def tournament_sample(
    token_probabilities: dict[int, float],
    seed: int,
    num_layers: int = 30,
    competitors_per_match: int = 2,
) -> int:
    ...
```

### Purpose

Execute the complete Tournament Sampling procedure.

### Contract

1.  Generate/obtain the candidate set.
2.  Apply the tournament procedure.
3.  Repeat across `num_layers`.
4.  Return exactly one output token.

### Return

``` python
int
```

representing the selected token ID.

The paper generally uses `m = 30` layers in its experiments unless
otherwise stated.

------------------------------------------------------------------------

# 10. Watermarked Token Generation

Member 2 should expose one high-level function:

``` python
def generate_watermarked_token(
    token_probabilities: dict[int, float],
    context: list[int],
    config: WatermarkConfig,
) -> int:
    ...
```

### Internal flow

``` text
context + key
      ↓
generate_seed()
      ↓
g-value generation
      ↓
candidate sampling
      ↓
tournament_sample()
      ↓
selected token
```

Member 3 should use this function rather than directly manipulating
tournament internals for ordinary generation experiments.

------------------------------------------------------------------------

# 11. Watermark Scoring

## 11.1 Token-Level Score

``` python
def score_token(
    token_id: int,
    context: list[int],
    watermark_key: int | str,
    num_layers: int = 30,
) -> float:
    ...
```

### Purpose

Calculate the watermark evidence contributed by one generated token.

------------------------------------------------------------------------

## 11.2 Sequence Score

``` python
def calculate_score(
    token_sequence: list[int],
    watermark_key: int | str,
    history_length: int = 4,
    num_layers: int = 30,
    masking_window: int = 1,
) -> float:
    ...
```

### Purpose

Calculate the aggregate watermark score for a generated sequence.

### Contract

-   Reconstruct the same deterministic seeds used during generation.
-   Calculate the relevant g-values.
-   Aggregate the evidence across tokens/layers.
-   Return one scalar score.

The paper describes detection using the mean g-values across the
text/layers; longer text generally provides more evidence.

------------------------------------------------------------------------

# 12. Threshold-Based Detection

``` python
def detect_watermark(
    score: float,
    threshold: float,
) -> bool:
    ...
```

### Contract

``` text
score >= threshold  → True
score < threshold   → False
```

The exact threshold is an experimental parameter and must not be
hard-coded into the core watermark implementation.

This is essential for Member 3 because different experiments may
evaluate different false-positive-rate operating points.

------------------------------------------------------------------------

# 13. Complete Detection API

``` python
def detect(
    token_sequence: list[int],
    watermark_key: int | str,
    threshold: float,
    config: WatermarkConfig,
) -> DetectionResult:
    ...
```

### Internal flow

``` text
Token sequence
      ↓
Seed reconstruction
      ↓
g-value reconstruction
      ↓
Sequence score
      ↓
Threshold comparison
      ↓
DetectionResult
```

------------------------------------------------------------------------

# 14. Experimental Evaluation Interface for Member 3

Member 3 should not need to know the internal implementation of the
watermark.

Use:

``` python
def evaluate_text(
    token_sequence: list[int],
    watermark_key: int | str,
    config: WatermarkConfig,
    threshold: float,
) -> DetectionResult:
    ...
```

For generation experiments:

``` python
def generate_and_evaluate(
    model,
    prompt: str,
    config: WatermarkConfig,
    generation_config,
) -> GenerationResult:
    ...
```

This becomes the main Member 2 → Member 3 integration point.

------------------------------------------------------------------------

# 15. Evaluation Metrics Exposed to Member 3

Member 2 must provide enough information for Member 3 to calculate:

## 15.1 Detection Score

``` text
score
```

## 15.2 Number of Tokens

``` text
num_tokens
```

## 15.3 Threshold

``` text
threshold
```

## 15.4 Watermark Decision

``` text
is_watermarked
```

Member 3 calculates dataset-level metrics such as:

-   True-positive rate
-   False-positive rate
-   TPR at fixed FPR
-   Score distributions
-   Detection performance versus text length
-   Detection performance versus entropy

The paper defines detectability using **TPR at a fixed FPR**; for
TPR@FPR=1%, the threshold is selected from the upper 1% of scores on
unwatermarked text and the fraction of watermarked texts above that
threshold is measured.

------------------------------------------------------------------------

# 16. Evaluation Metadata Contract

Every generated/evaluated sample should be representable as:

``` python
@dataclass
class EvaluationRecord:
    model_name: str
    prompt_id: str
    text: str
    token_ids: list[int]
    watermarked: bool
    score: float
    threshold: float | None
    detected: bool | None
    num_tokens: int
    entropy: float | None
    perplexity: float | None
    generation_time_ms: float | None
    config_name: str
```

Member 3 can aggregate these records into tables and plots.

------------------------------------------------------------------------

# 17. Experimental Configurations

Member 3 must be able to vary:

``` python
@dataclass
class ExperimentConfig:
    model_name: str
    dataset_name: str
    num_samples: int
    generation_length: int
    temperature: float
    top_k: int | None
    top_p: float | None
    watermark_config: WatermarkConfig
```

The paper's evaluation includes Gemma-family models and Mistral 7B, with
ELI5 used for controlled text-generation experiments.

------------------------------------------------------------------------

# 18. Non-Distortionary vs Distortionary Interface

## Non-Distortionary

``` python
WatermarkConfig(
    competitors_per_match=2,
    distortion_mode="non_distortionary",
)
```

Purpose:

-   Preserve text quality.
-   Provide detectable statistical evidence.
-   Use repeated context masking where configured.

## Distortionary

``` python
WatermarkConfig(
    competitors_per_match=N,
    distortion_mode="distortionary",
)
```

where:

``` text
N > 2
```

Purpose:

-   Apply a stronger watermark.
-   Potentially improve detectability.
-   Accept some loss in text quality.

Member 3 compares these configurations experimentally.

------------------------------------------------------------------------

# 19. Repeated Context Masking

Member 2 must expose the configuration:

``` python
def apply_context_mask(
    token_history: list[int],
    masking_window: int = 1,
) -> list[int]:
    ...
```

The implementation must ensure that the same context window is not
repeatedly watermarked when repeated context masking is enabled.

For the project's default configuration:

``` text
masking_window = 1
```

The paper uses repeated context masking to obtain the stated
non-distortionary configuration.

------------------------------------------------------------------------

# 20. Temperature, Top-k and Top-p

Member 2 must keep the watermarking layer compatible with the LLM
sampling configuration.

``` python
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
```

The watermark algorithm operates on the resulting token distribution.

The paper states compatibility with:

-   top-k where `k >= 2`
-   top-p where `0 < p <= 1`
-   temperature greater than zero

------------------------------------------------------------------------

# 21. Latency and Scalability Hook

Member 3 needs generation-level timing without modifying the watermark
algorithm.

Recommended API:

``` python
@dataclass
class TimingResult:
    total_time_ms: float
    tokens_generated: int
    ms_per_token: float
```

Optional wrapper:

``` python
def benchmark_generation(
    generate_fn,
    *args,
    **kwargs,
) -> TimingResult:
    ...
```

Member 3 uses this to compare:

``` text
Unwatermarked generation
vs.
Watermarked generation
```

The paper evaluates computational overhead relative to the underlying
LLM forward pass and reports very small additional latency for the
watermarking layer.

------------------------------------------------------------------------

# 22. Speculative Sampling Interface

Speculative sampling is an evaluation/deployment integration rather than
a replacement for the core watermark API.

Member 2 should expose a compatible generation interface:

``` python
def generate_watermarked_speculative(
    target_model,
    draft_model,
    prompt_tokens: list[int],
    config: WatermarkConfig,
    lookahead: int = 3,
) -> GenerationResult:
    ...
```

Member 3 evaluates:

-   Acceptance rate
-   Overall latency
-   Detectability
-   Watermarked vs unwatermarked performance

The paper evaluates fast watermarked speculative sampling using Gemma
7B-IT as the target model and Gemma 2B-IT as the draft model, with three
lookahead tokens.

------------------------------------------------------------------------

# 23. Member 3 Experimental Matrix

Member 3 should be able to request experiments without changing Member 2
code.

### Detection

``` text
SynthID-Text
vs
Gumbel sampling
```

### Distortionary evaluation

``` text
SynthID-Text
vs
Soft Red List
```

### Variables

``` text
Text length
Entropy / temperature
Watermark type
Number of tournament layers
Model
```

### Quality

``` text
Perplexity
Human evaluation
Grammaticality/coherence
Relevance
Correctness
Helpfulness
Overall quality
```

### Scalability

``` text
Latency
ms/token
Watermark overhead
Model size
Vectorized implementation
```

------------------------------------------------------------------------

# 24. Baseline Interface

For fair evaluation, Member 3 may need alternative sampling methods.

The baseline wrapper should follow the same interface:

``` python
def generate_with_baseline(
    model,
    prompt: str,
    sampling_method: str,
    generation_config,
) -> GenerationResult:
    ...
```

The baseline implementation must not alter the Member 2 SynthID-Text
implementation.

For like-for-like comparisons, the paper keeps the same seed-generation
and repeated-context-masking components while comparing different
sampling algorithms.

------------------------------------------------------------------------

# 25. Error Handling

Member 2 must validate:

``` text
history_length >= 0
num_layers >= 1
competitors_per_match >= 2
temperature > 0
top_k is None or top_k >= 2
top_p is None or 0 < top_p <= 1
threshold is finite
token IDs are valid
probabilities are non-negative
probability distribution has non-zero total mass
```

Recommended exceptions:

``` python
ValueError
TypeError
```

No experiment should silently continue with an invalid watermark
configuration.

------------------------------------------------------------------------

# 26. Reproducibility Requirements

For the same:

``` text
model
prompt
watermark key
context
sampling configuration
random seed
watermark configuration
```

the watermark layer should produce reproducible behavior where
deterministic execution is requested.

Member 3 must record the configuration used for every experiment.

------------------------------------------------------------------------

# 27. End-to-End Example

``` python
config = WatermarkConfig(
    watermark_key="project-secret-key",
    history_length=4,
    num_layers=30,
    competitors_per_match=2,
    distortion_mode="non_distortionary",
    repeated_context_masking=True,
    masking_window=1,
)

token_id = generate_watermarked_token(
    token_probabilities=next_token_distribution,
    context=context_tokens,
    config=config,
)

score = calculate_score(
    token_sequence=generated_tokens,
    watermark_key=config.watermark_key,
    history_length=config.history_length,
    num_layers=config.num_layers,
    masking_window=config.masking_window,
)

result = detect(
    token_sequence=generated_tokens,
    watermark_key=config.watermark_key,
    threshold=threshold,
    config=config,
)

print(result.is_watermarked)
print(result.score)
```

------------------------------------------------------------------------

# 28. What Member 3 Must Receive From Member 2

At minimum:

``` text
1. Generated token IDs
2. Generated text
3. Watermark configuration
4. Watermark score
5. Detection result
6. Number of generated tokens
7. Generation timing
8. Optional per-token/per-layer scores for analysis
```

The preferred optional diagnostic structure is:

``` python
@dataclass
class WatermarkDiagnostics:
    seeds: list[int] | None
    token_scores: list[float] | None
    layer_scores: list[list[float]] | None
```

These diagnostics are useful for research analysis but should not be
required for ordinary detection.

------------------------------------------------------------------------

# 29. What Member 3 Must NOT Change

Member 3 must not independently modify:

-   Seed-generation logic
-   Watermark key interpretation
-   g-value generation
-   Tournament winner selection
-   Sequence scoring
-   Core detection implementation

If an experimental change is required, Member 3 should request a new
configuration parameter or a new versioned interface from Member 2.

------------------------------------------------------------------------

# 30. Testing Requirements

## Member 2 Unit Tests

### Seed

``` text
same context + same key → same seed
different key → different seed sequence
```

### g-values

``` text
same token + seed + layer → same g-value
different layer → independently generated layer-specific value
```

### Tournament

``` text
valid candidates → exactly one winner
```

### Detection

``` text
score >= threshold → True
score < threshold → False
```

### Configuration

``` text
invalid values → ValueError
```

------------------------------------------------------------------------

# 31. Integration Tests

The following must work before Member 3 begins large experiments:

``` text
LLM distribution
    ↓
seed generation
    ↓
g-values
    ↓
candidate sampling
    ↓
tournament sampling
    ↓
generated sequence
    ↓
sequence scoring
    ↓
threshold detection
```

The same watermark key and configuration must be usable by both
generation and detection.

------------------------------------------------------------------------

# 32. Definition of Done --- Member 2

Member 2 is complete when:

-   [ ] Seed generation is implemented.
-   [ ] Watermark key is supported.
-   [ ] Sliding-window context is implemented.
-   [ ] g-values are generated deterministically.
-   [ ] Candidate sampling works.
-   [ ] Single-layer tournament works.
-   [ ] Multilayer tournament works.
-   [ ] Non-distortionary configuration works.
-   [ ] Distortionary configuration works.
-   [ ] Sequence scoring works.
-   [ ] Threshold-based detection works.
-   [ ] Public APIs match this document.
-   [ ] Unit tests pass.
-   [ ] Member 3 can run an end-to-end generation/detection experiment
    without changing core code.

------------------------------------------------------------------------

# 33. Definition of Done --- Member 3

Member 3 is complete when:

-   [ ] Member 2 APIs are used directly.
-   [ ] Experimental configurations are reproducible.
-   [ ] Detection experiments are implemented.
-   [ ] TPR@FPR evaluation is implemented.
-   [ ] Text-length analysis is implemented.
-   [ ] Entropy analysis is implemented.
-   [ ] Quality/perplexity analysis is implemented.
-   [ ] Distortionary/non-distortionary comparisons are implemented.
-   [ ] Baseline comparisons are implemented.
-   [ ] Latency/overhead measurements are implemented.
-   [ ] Speculative-sampling evaluation is implemented where supported.
-   [ ] Results are stored in the agreed schema.
-   [ ] Tables and plots can be generated from the stored results.
-   [ ] Limitations are documented using the paper's findings.

------------------------------------------------------------------------

# 34. Final Interface Contract

The minimum stable public interface between Member 2 and Member 3 is:

``` python
# Configuration
WatermarkConfig
SamplingConfig

# Generation
generate_seed()
g_value()
compute_g_values()
sample_candidates()
tournament_sample_layer()
tournament_sample()
generate_watermarked_token()

# Detection
score_token()
calculate_score()
detect_watermark()
detect()

# End-to-end
generate_and_evaluate()

# Experimental support
benchmark_generation()
generate_watermarked_speculative()
```

The central integration principle is:

``` text
MEMBER 2
Core watermark algorithm
        │
        │ stable API
        ▼
MEMBER 3
Experiments → Metrics → Results → Analysis
```

Member 3 should therefore treat Member 2's watermark engine as a
**black-box research component with a stable interface**, while Member 2
should keep the implementation modular enough to support the
experimental variables required by the paper.

------------------------------------------------------------------------

# 35. Source Alignment

This interface specification is derived from:

1.  **Dathathri et al., 2024 --- "Scalable watermarking for identifying
    large language model outputs"**
    -   Generative watermarking framework
    -   Random seed generation
    -   Sampling algorithm
    -   g-values
    -   Tournament sampling
    -   Watermark scoring
    -   Threshold-based detection
    -   Non-distortionary and distortionary configurations
    -   Computational scalability
    -   Speculative sampling
    -   Experimental evaluation
2.  **AI WORK DIVISION --- Final Locked Three-Member Research Work
    Division**
    -   Member 2: Core Technical Method
    -   Member 3: Experimental Evaluation, Scalability, Results &
        Limitations

No unrelated research topic is introduced into the project scope.
