# Distortionary Tournament Sampling

## Purpose

Issue #17 extends Tournament Sampling so that the number of competitors
per tournament match can be configured.

The SynthID-Text paper defines multilayer Tournament Sampling using:

- `N` competitors per match
- `m` tournament layers
- `N^m` initial candidate tokens

The non-distortionary configuration uses exactly two competitors per match.

When more than two competitors are used in each match, the resulting
configuration is distortionary at the token level and provides stronger
watermarking.

## Tournament Structure

For `N` competitors and `m` layers:

    Initial candidates = N^m

At each layer:

    N candidates
        ↓
    select maximum-g candidate
        ↓
    1 winner

The process is repeated until one final token remains.

For example, with `N = 4` and `m = 3`:

    4^3 = 64 initial candidates

    Layer 1:
    64 → 16 winners

    Layer 2:
    16 → 4 winners

    Layer 3:
    4 → 1 winner

## Configurations

| Competitors | Layers | Initial Candidates |
|-------------|--------|--------------------|
| 2 | 3 | 8 |
| 2 | 5 | 32 |
| 3 | 3 | 27 |
| 4 | 3 | 64 |
| 5 | 2 | 25 |

## Selection Rule

For each match:

1. Compute the g-value of every candidate using the g-function for
   the current tournament layer.
2. Find the maximum g-value.
3. Keep every candidate whose g-value equals the maximum.
4. Sample uniformly among those candidates.

Different tournament layers use different g-functions.

## Distortionary Configuration

When exactly two competitors participate in every match, Tournament
Sampling is single-token non-distortionary.

When more than two competitors participate in each match, Tournament
Sampling becomes distortionary at the token level.

The reason is that selecting the highest-scoring candidate from a larger
set biases the output distribution more strongly toward tokens with
higher g-values.

## Detectability vs Quality Trade-off

Increasing the number of competitors strengthens the watermark.

This improves watermark detectability, but comes at the cost of text
quality.

Therefore:

    More competitors
          ↓
    Stronger watermark
          ↓
    Higher detectability
          ↓
    Greater distortion
          ↓
    Potential quality loss

The paper describes distortionary SynthID-Text as the configuration to
use when stronger watermark detectability is more important than
preserving the original token distribution.

The paper's evaluation compares distortionary SynthID-Text with Soft Red
List and shows a detectability-versus-quality trade-off.

## Implementation Parameters

The TextSus implementation exposes:

    num_competitors
    num_layers

The required number of initial candidates is:

    num_competitors ** num_layers

The default value of `num_competitors` is `2`, preserving the existing
non-distortionary configuration.

## Scope

This issue implements the configurable tournament structure.

It does not claim that the repository has reproduced the paper's
full quality/detectability experimental evaluation. Those measurements
belong to the evaluation stage.