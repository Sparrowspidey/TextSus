# Vectorized Tournament: Complexity Notes

Issue #31 asks for a faster Tournament Sampling implementation and a short complexity comparison. This note describes the cost of reducing `N^m` sampled candidates through `m` tournament layers. `N` is the number of competitors per match; `m` is the number of layers.

## Work Per Generation Step

At the start, each tournament has `N^m` candidate tokens. Each layer reduces the candidate count by a factor of `N`:

```text
N^m + N^(m-1) + ... + N
```

The exact number of candidate g-value evaluations per tournament is:

```text
N + N^2 + ... + N^m = (N^(m+1) - N) / (N - 1)
```

For `B` independent generation positions or tournaments, the work is:

```text
O(B × (N^m + N^(m-1) + ... + N)) = O(B × N^m)
```

This excludes the cost of computing one g-value and the language-model sampling cost for the initial candidates. If one g-value evaluation costs `C_g`, multiply the evaluation work by `C_g`.

## Scalar and Vectorized Implementations

| Implementation | Asymptotic tournament work | Main cost characteristic |
|---|---:|---|
| Scalar loops over tournaments, matches, and candidates | `O(B × N^m)` | Python dispatch and per-match loop overhead |
| Vectorized layer reduction | `O(B × N^m)` | Performs g-value and match operations on arrays; Python loop remains over `m` layers |

Vectorization does not change the algorithm's asymptotic work. It reduces interpreter overhead by processing the candidates and matches for a layer together. The actual wall-clock speedup depends on batch size, g-value implementation, array allocation, and hardware; this note does not claim a measured speedup.

## Memory

The initial candidate array contains `B × N^m` token IDs. The vectorized reduction also creates score, tie-priority, and grouped-view/intermediate arrays. Peak working memory is therefore `O(B × N^m)`; it is governed by the initial candidate count, not by the final single-token output.

For the paper's two-competitor setting, `N = 2`, so initial candidates scale as `2^m`. Increasing `N` or `m` increases initial candidate count exponentially in the other parameter. The vectorized implementation improves execution overhead but does not remove this candidate-count growth.

## Algorithm Semantics

The vectorized implementation keeps the method's selection rule: in every match, the highest g-value advances; ties are broken uniformly at random. Its candidate reduction is equivalent to grouping the independently sampled candidates into matches and applying the tournament layer by layer. The paper describes this tournament procedure and random tie-breaking in its Tournament Sampling section: [SynthID-Text paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499265/).
