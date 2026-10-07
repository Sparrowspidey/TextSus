# Non-Distortion in SynthID-Text

This document explains the non-distortion levels used by SynthID-Text and how they relate to Tournament Sampling and repeated context masking.

## Overview

The paper distinguishes non-distortion at the token level from non-distortion over one or more complete sequences. These are different guarantees:

- **Single-token non-distortion** preserves the distribution of one generated token when averaged over the random seed.
- **K-sequence non-distortion** preserves the probability of generating a set of `K` responses when averaged over the random seeds, using K-sequence repeated context masking.

## Single-Token Non-Distortion

At generation step `t`, let `p_LM(x_t | x_<t)` be the original language-model distribution for the next token, given the preceding context. A watermark sampling algorithm is single-token non-distortionary when, averaged over the random seed `r_t`, its distribution for `x_t` is equal to the original distribution:

```text
Average over r_t:
watermarked distribution of x_t = p_LM(x_t | x_<t)
```

The paper shows that Tournament Sampling with exactly two competitors per match has this property:

```text
num_competitors = 2
M = 2^m initial candidates for m tournament layers
```

This statement is about a token at a single generation step. It does not, by itself, establish that a complete response—or several responses—has the same probability distribution as the unwatermarked language model.

## Sequence Non-Distortion

Sequence non-distortion concerns the probability of generating one or more complete text sequences, rather than the distribution of one token in isolation. The paper's K-sequence definition concerns `K` responses generated under the repeated context masking procedure.

### K-Sequence Repeated Context Masking

The seed generator uses the preceding `H` tokens as a context window. Repeated context masking tracks context windows that have already been watermarked within the configured history.

At each generation step:

1. Form the current context window from the preceding `H` tokens.
2. If that context window has already been watermarked within the tracked history, skip watermarking at this step and sample from the original language-model distribution.
3. Otherwise, apply the configured watermark sampling algorithm and add the context window to the history.

The integer `K >= 1` controls the response history used for this check. `K = 1` keeps context history for one response. Larger `K` values check contexts across the last `K` responses. The paper reports using `K = 1` for most experiments.

The paper shows that K-sequence repeated context masking gives K-sequence non-distortion. In particular, `K = 1` gives single-sequence non-distortion.

## Generation Integration Review

Repeated context masking belongs in the generation loop, before seed-dependent watermark sampling. At each step, the implementation must decide whether the current context window is eligible for watermarking before it calls the seed generator and Tournament Sampler.

### Expected per-step flow

```text
Build the current H-token context window
                  ↓
Check it against context history for K responses
           ┌──────┴──────┐
     Already used     Not used
          ↓                ↓
Sample directly      Generate seed from
from p_LM             context and key
                           ↓
                    Generate g-values
                           ↓
                  Tournament Sampling
                           ↓
               Record context as watermarked
```

In words:

1. Build the current context window using the same `H`-token convention as seed generation.
2. Check the window against the repeated context history scoped by `K`.
3. If the context was already watermarked, bypass watermark sampling and sample from the original language-model distribution for this step.
4. If it is new in the tracked history, generate the seed, compute g-values, and apply Tournament Sampling; then record the context as watermarked.

### Review points for the generation path

- **Masking precedes watermark sampling.** The decision must occur before generating a watermark seed or running Tournament Sampling for the step.
- **The two context windows must agree.** The context checked by masking must use the same preceding-token window convention as the seed generator.
- **The unwatermarked branch must remain unwatermarked.** Repeated contexts use the original language-model distribution at that step.
- **`K` controls history scope.** `K = 1` keeps history for one response; larger values check contexts across the last `K` responses.
- **`N` and `K` control different things.** `N = 2` is the single-token non-distortionary sampling configuration; repeated context masking with `K` gives the sequence-level guarantee. `N > 2` remains distortionary even if masking is enabled.
- **Scoring must account for masked positions.** Positions where watermarking was skipped should not be treated as if the watermark sampler had been applied.

These points review the method-level generation integration described by the paper. The project’s actual generation implementation was not available here, so this document does not claim that its code has been inspected.

## Relationship Between the Guarantees

| Configuration | Guarantee | What it covers |
|---|---|---|
| Tournament Sampling with `N = 2` competitors per match | Single-token non-distortion | The distribution of a token at one generation step, averaged over the random seed |
| `N = 2` plus K-sequence repeated context masking | K-sequence non-distortion | The probability of generating `K` responses under the masking procedure |
| Tournament Sampling with `N > 2` | Distortionary sampling | The token distribution is biased toward candidates with higher g-values |

Repeated context masking is the mechanism used to extend the two-competitor single-token guarantee to a sequence-level guarantee. The `N > 2` distortionary configuration changes the token sampling distribution; repeated context masking should not be described as turning that configuration into the two-competitor non-distortionary configuration.

## Detectability and Quality Trade-Off

The paper describes a trade-off between preserving the original language-model distribution and watermark detectability. Using more than two competitors per tournament match increases the bias toward higher g-values, strengthening the watermark signal while distorting the original token distribution. Non-distortionary configurations preserve the relevant distribution guarantee but involve trade-offs in watermark detectability and, for larger `K`, context-history cost as discussed in the paper.

## Interface Requirements

- The sampling configuration must expose the number of competitors per tournament match so the `N = 2` and `N > 2` cases are explicit.
- Documentation must identify `N = 2` as the single-token non-distortionary Tournament Sampling configuration.
- Sequence-level non-distortion must be associated with K-sequence repeated context masking, not with `N = 2` alone.
- The repeated context mask must prevent watermark application when the current context window has already been watermarked within the configured history.
- The `K` setting and its response-history scope must be explicit.
- Distortionary sampling and sequence-level non-distortion must not be conflated.

## Reference

Dathathri, S. et al. “Scalable watermarking for identifying large language model outputs.” *Nature* 634, 818–824 (2024). See the sections on Tournament Sampling, repeated context masking, and Supplementary Information sections G.1–G.3.
