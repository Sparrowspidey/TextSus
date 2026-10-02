# Core Method Interfaces

This document defines the planned interfaces for the core technical method implementation of SynthID-Text in TextSus.

The interfaces covered are:

- Random seed generation
- G-value generation
- Tournament Sampling
- Mean watermark scoring
- Weighted mean watermark scoring
- Frequentist mean scoring
- Frequentist weighted mean scoring
- Parameterized Bayesian scoring
- Watermark detection

The mathematical and algorithmic requirements are derived from the SynthID-Text paper. Function names, Python representations, and exact input/output shapes are implementation-level interface decisions for TextSus.

---

## 1. Random Seed Generator

### Purpose

Generate the random seed used by the watermarking sampling algorithm at each generation step.

In SynthID-Text, the random seed at generation step `t` is generated using the preceding token context and the watermarking key.

The paper describes a sliding-window method in which the random seed is generated using the most recent `H` tokens together with the watermarking key. The paper uses `H = 4`.

### Interface

    generate_seed(
        context_tokens,
        watermarking_key
    ) -> seed

### Inputs

 `context_tokens` - Most recent tokens used by the sliding-window seed generator 
 `watermarking_key` - Watermarking key used by the seed generator 

### Output

`seed` - Random seed used for the current generation step 

### Configuration

    H = 4

For the paper's configuration:

    context_tokens = (x[t-4], x[t-3], x[t-2], x[t-1])

### Conceptual flow

    Previous token context + watermarking key
                        ↓
                      Hash
                        ↓
                  Random seed r_t

### Requirements

- The seed generator must be deterministic for the same context and key.
- The relevant previous-token context must be incorporated into the seed.
- The watermarking key must be incorporated into the seed.
- The implementation must support the sliding-window configuration used by SynthID-Text.

---

## 2. G-Value Generator

### Purpose

Generate the watermarking g-value for a candidate token using a random seed and a tournament layer.

SynthID-Text uses pseudorandom functions for the tournament layers. A g-value function assigns a score to a candidate token using the random seed.

### Interface

    generate_g_value(
        token_id,
        seed,
        layer
    ) -> g_value

### Inputs

`token_id` -  Candidate token ID 
`seed` - Random seed for the current generation step 
`layer`-  Tournament layer 

### Output

`g_value` - G-value assigned to the candidate token 

### G-value distributions

The implementation should support the g-value distributions described or explored in the paper:

- Bernoulli(0.5)
- Uniform `[0, 1]`

The default SynthID-Text configuration used in the paper's experiments is:

    Bernoulli(0.5)

### Layer functions

The implementation should support separate g-value functions for different tournament layers:

    g_1(token, seed)
    g_2(token, seed)
    ...
    g_m(token, seed)

Each layer produces its own g-value for a candidate token.

---

## 3. Tournament Sampling

### Purpose

Select the output token using candidate tokens sampled from the LLM distribution and watermarking g-values.

Tournament Sampling is the sampling algorithm used by SynthID-Text.

For an `m`-layer tournament, the paper uses:

    M = 2^m

candidate tokens.

### Interface

    tournament_sample(
        token_ids,
        seed,
        num_layers
    ) -> selected_token

### Inputs


 `token_ids` - Candidate token IDs sampled from the LLM distribution 
 `seed` - Random seed used by the watermarking functions 
 `num_layers` -  Number of tournament layers 

### Output


 `selected_token` - Final token selected by Tournament Sampling 

### Candidate count

For an `m`-layer tournament:

    M = 2^m

Examples:

    m = 1  →  M = 2
    m = 2  →  M = 4
    m = 3  →  M = 8

### Tournament process

    LLM distribution
           ↓
    Sample M = 2^m candidate tokens
           ↓
    Randomly form pairs
           ↓
    Tournament layer 1
           ↓
    Compare g_1 values
           ↓
    Retain winners
           ↓
    Tournament layer 2
           ↓
    Compare g_2 values
           ↓
    Retain winners
           ↓
          ...
           ↓
    Tournament layer m
           ↓
    One final winner
           ↓
    Output token

For each tournament layer:

1. Candidate tokens are grouped into pairs.
2. The corresponding layer's g-value function is applied.
3. The token with the higher g-value wins.
4. Ties are broken randomly.
5. The winners continue to the next layer.
6. The process continues until one token remains.

### Requirements

- The implementation must support a configurable number of tournament layers.
- The initial candidate count must be `2^m` for an `m`-layer tournament.
- Each tournament layer must use its corresponding g-value function.
- Ties must be handled according to the tournament algorithm.
- The final remaining token is the output token.

---

## 4. Mean Watermark Score

### Purpose

Compute the mean watermark evidence of a tokenized text across generation timesteps and tournament layers.

The score is calculated from the g-values associated with the generated tokens and their corresponding random seeds.

### Interface

    mean_score(
        token_ids,
        watermarking_key,
        num_layers
    ) -> score

### Inputs



 `token_ids` - Tokenized text containing `T` tokens 
 `watermarking_key` - Key used to reproduce the random seeds 
 `num_layers` - Number of tournament layers 

### Output


 `score` - Mean watermark score 

### Scoring process

For every timestep:

1. Generate the random seed from the preceding context and watermarking key.
2. Compute the g-value of the generated token for each tournament layer.
3. Aggregate the g-values across all timesteps and layers.
4. Calculate their mean.

### Mathematical form

For `T` tokens and `m` tournament layers:

    Score(x) = (1 / (T × m)) × Σ_t Σ_l g_l(x_t, r_t)

where:

- `T` = number of tokens
- `m` = number of tournament layers
- `x_t` = token at timestep `t`
- `r_t` = random seed at timestep `t`
- `g_l` = g-value function for layer `l`

---

## 5. Weighted Mean Watermark Score

### Purpose

Compute a watermark score in which evidence from different tournament layers can receive different weights.

The paper describes the weighted mean as a scoring function that re-weights the evidence contributed by each tournament layer.

### Interface

    weighted_mean_score(
        token_ids,
        watermarking_key,
        layer_weights
    ) -> score

### Inputs


 `token_ids` - Tokenized text containing `T` tokens 
 `watermarking_key` - Key used to reproduce the random seeds 
 `layer_weights` - Weight assigned to each tournament layer 

### Output


 `score` - Weighted watermark score 

### Requirements

- There must be one weight for each tournament layer.
- The scoring function must reproduce the random seeds using the watermarking key and token context.
- G-values must be obtained for the generated tokens.
- The corresponding layer weights must be applied to the layer evidence.

---

## 6. Frequentist Scoring

### Purpose

Compute frequentist versions of the watermark scores by performing a hypothesis test on the mean-based watermark evidence.

The paper proposes frequentist versions of the mean score and weighted mean score that produce a P value.

### 6.1 Frequentist Mean Score

#### Interface

    frequentist_mean_score(
        token_ids,
        watermarking_key,
        num_layers
    ) -> p_value

#### Inputs

`token_ids` - Tokenized text containing the generated tokens

`watermarking_key` - Key used to reproduce the random seeds

`num_layers` - Number of tournament layers

#### Output

`p_value` - P value produced by the hypothesis test on the mean watermark score

#### Requirements

- The watermark score must be computed from the g-values of the generated tokens.
- Random seeds must be reproduced using the watermarking key and token context.
- The function must perform the corresponding frequentist hypothesis test.
- The function must return a P value.

### 6.2 Frequentist Weighted Mean Score

#### Interface

    frequentist_weighted_mean_score(
        token_ids,
        watermarking_key,
        layer_weights
    ) -> p_value

#### Inputs

`token_ids` - Tokenized text containing the generated tokens

`watermarking_key` - Key used to reproduce the random seeds

`layer_weights` - Weight assigned to each tournament layer

#### Output

`p_value` - P value produced by the hypothesis test on the weighted mean watermark score

#### Requirements

- The weighted watermark score must be computed from the g-values of the generated tokens.
- Random seeds must be reproduced using the watermarking key and token context.
- The corresponding layer weights must be applied.
- The function must perform the corresponding frequentist hypothesis test.
- The function must return a P value.

---

## 7. Parameterized Bayesian Scoring

### Purpose

Compute the posterior probability that a text is watermarked using a parameterized Bayesian scoring function.

The paper describes this scoring function as learning from watermarked and unwatermarked texts.

### Interface

    bayesian_score(
        token_ids,
        watermarking_key,
        parameters
    ) -> posterior_probability

### Inputs

`token_ids` - Tokenized text containing the generated tokens

`watermarking_key` - Key used to reproduce the random seeds

`parameters` - Learned parameters of the Bayesian scoring function

### Output

`posterior_probability` - Posterior probability that the text is watermarked

### Requirements

- The scoring function must use the g-values associated with the generated tokens.
- Random seeds must be reproduced using the watermarking key and token context.
- The function must use learned parameters.
- The parameters are learned from watermarked and unwatermarked texts.
- The function must return the posterior probability that the text is watermarked.

---

## 8. Watermark Detection

### Purpose

Determine whether a tokenized text contains sufficient watermark evidence to be classified as watermarked.

The detection process uses the watermark score and a detection threshold.

### Interface

    detect_watermark(
        score,
        threshold
    ) -> is_watermarked

### Inputs

`score` - Watermark score calculated from the generated text

`threshold` - Detection threshold used to determine whether the score provides sufficient watermark evidence

### Output

`is_watermarked` - Boolean indicating whether the text is classified as watermarked

### Detection rule

    score >= threshold
        ↓
    watermarked

    score < threshold
        ↓
    non-watermarked

### Requirements

- The detection function must accept a watermark score.
- The detection function must accept a configurable threshold.
- The function must return a boolean detection decision.
- The threshold determines whether sufficient watermark evidence is present.

---

## 9. Interface Relationships

The core method follows this dependency chain:

    Previous token context
            +
    Watermarking key
            ↓
    Random Seed Generator
            ↓
    Random seed r_t
            ↓
    G-value Functions
            ↓
    G-values for candidate tokens
            ↓
    Tournament Sampling
            ↓
    Selected token x_t
            ↓
    Generated text
            ↓
    Scoring Function
            ↓
    Watermark Score
            ↓
    Detection
            ↓
    Watermarked / Non-watermarked

---

## 10. Data and Shape Conventions

| Symbol                | Meaning                               | Shape                 |
| `V`                   | LLM vocabulary                        | vocabulary-sized      |
| `x_t`                 | Token at timestep `t`                 | scalar token ID       |
| `x_<t`                | Tokens preceding timestep `t`         | sequence              |
| `H`                   | Sliding-window context size           | scalar                |
| `T`                   | Number of tokens in a text            | scalar                |
| `m`                   | Number of tournament layers           | scalar                |
| `M`                   | Number of tournament candidates       | scalar, `2^m`         |
| `r_t`                 | Random seed at timestep `t`           | seed value            |
| `g_l`                 | G-value function for layer `l`        | function              |
| `g_l(x_t, r_t)`       | G-value for token `x_t` at layer `l`  | scalar                |
| `Score(x)`            | Watermark score of text `x`           | scalar                |

---

## 11. Paper-Derived Requirements

The following concepts are based on the SynthID-Text paper:

- Random seed generation
- Watermarking key
- Sliding-window seed generation
- Context window size `H = 4`
- Hash-based random seed generation
- Pseudorandom g-value functions
- G-values
- Tournament Sampling
- `M = 2^m` candidates for an `m`-layer tournament
- Tournament layers
- Mean watermark score
- Weighted mean watermark score

---

## 12. TextSus Implementation Decisions

The following are implementation-level decisions for this repository and are not claimed to be literal Python APIs from the paper:

- Function names
- Python data structures
- Exact watermarking-key representation
- Exact seed representation
- Exact tensor/array implementation
- Exact type annotations
- Module organization

These decisions should remain consistent with this interface specification as implementation proceeds.

---

## 13. Scope

This interface specification covers the core-method task:

- Seed generation
- G-value generation
- Tournament Sampling
- Mean scoring
- Weighted mean scoring
- Frequentist mean scoring
- Frequentist weighted mean scoring
- Parameterized Bayesian scoring
- Watermark detection

The following are intentionally outside the scope of this interface specification and will be handled by later issues:

- Repeated context masking
- Distortionary variants
- Vectorized Tournament Sampling
- Complexity optimization
- Evaluation
- Experimental pipelines

---

## 14. References

Primary source:

Dathathri, S. et al. "Scalable watermarking for identifying large language model outputs." Nature 634, 818–824 (2024).

The interface specification follows the terminology and method described in the paper's sections on:

- Watermarking with SynthID-Text
- The LLM distribution
- Watermarking framework
- Random seed generator
- SynthID-Text's Tournament Sampling approach
- Watermark detection