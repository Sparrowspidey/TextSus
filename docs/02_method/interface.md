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

### Overview

Tournament Sampling is the sampling algorithm used by SynthID-Text to
select the next token from multiple independently sampled candidate
tokens.

The method assigns each candidate a pseudorandom g-value and selects
the candidate with the highest g-value.

The paper describes both a single-layer formulation and a multilayer
formulation.

---

### Single-Layer Tournament Sampling

For a single tournament layer, `N` candidate tokens are sampled
independently from the language model distribution.

For each candidate token `x`, a g-value is computed using:

- the candidate token,
- the random seed,
- the tournament layer.

The candidates with the maximum g-value are retained.

If multiple candidates have the same maximum g-value, one of them is
selected uniformly at random.

#### Process

    text
N candidate tokens
       |
       v
Compute g-values
       |
       v
Find maximum g-value
       |
       v
Keep all maximum-g candidates
       |
       v
Uniform random selection
       |
       v
Selected token

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

Compute frequentist versions of the mean and weighted mean
watermark scores by performing hypothesis tests.

### Frequentist Mean

    frequentist_mean_score(
        token_ids,
        watermarking_key,
        num_layers
    ) -> p_value

The returned P value measures the probability, under the
unwatermarked null hypothesis, of observing watermark evidence
at least as strong as the observed evidence.

### Frequentist Weighted Mean

    frequentist_weighted_mean_score(
        token_ids,
        watermarking_key,
        layer_weights
    ) -> p_value

The weighted version applies the corresponding layer weights
before performing the hypothesis test.

### Interpretation

Smaller P values provide stronger evidence against the
unwatermarked null hypothesis.

---

## 7. Parameterized Bayesian Scoring

### Purpose

Compute the posterior probability that a text is watermarked.

### Interface

    bayesian_score(
        token_ids,
        watermarking_key,
        parameters
    ) -> posterior_probability

### Inputs

`token_ids`
- Tokenized text.

`watermarking_key`
- Key used to reproduce the random seeds.

`parameters`
- Learned Bayesian detector parameters.

### Output

`posterior_probability`
- Posterior probability that the text is watermarked.

### Training requirement

The Bayesian parameters are learned from watermarked and
unwatermarked text.

The Bayesian detector therefore cannot be used with arbitrary
untrained parameters.

---
## 8. Watermark Detection

### Purpose

Determine whether sufficient watermark evidence is present.

### Score-based interface

    detect_watermark(
        score,
        threshold
    ) -> is_watermarked

### Rule

    score >= threshold
        ↓
    watermarked

    score < threshold
        ↓
    non-watermarked

This rule is used for scoring methods where larger scores indicate
stronger watermark evidence, including mean, weighted mean, and
Bayesian posterior scores.

### Frequentist detection

    detect_frequentist(
        p_value,
        alpha
    ) -> is_watermarked

### Rule

    p_value <= alpha
        ↓
    watermarked

    p_value > alpha
        ↓
    non-watermarked

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