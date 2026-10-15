## Chapter 3 — Core Technical Method


### 3.1 Overview

This chapter describes the core method implemented for TextSus, a project-level implementation based on SynthID-Text. SynthID-Text is a generative watermark: it introduces a statistical signal during token sampling, then detects that signal from the generated token sequence without requiring the language model at detection time. The method is organized around four connected operations: (1) derive a deterministic seed from recent token context and a secret watermarking key; (2) assign pseudorandom g-values to candidate tokens; (3) select a candidate with Tournament Sampling; and (4) reconstruct and aggregate g-value evidence to score and classify text.

At generation step \(t), let \(x_{<t}\) denote the preceding tokens, \(k\) the watermark key, and \(p_{LM}(\cdot\mid x_{<t})\) the model’s next-token distribution. A context-dependent seed \(r_t\) initializes the pseudorandom functions used by each tournament layer. Candidate tokens sampled from the model distribution are compared using their layer-specific g-values, and the tournament returns the next token \(x_t\). At detection, the same key and context reconstruct the seed and g-values for each observed token; an aggregate score is compared with a threshold.

The implementation follows the paper’s algorithmic structure, while its concrete hash functions, serialization, Python interfaces, and some scoring details are repository choices. It should therefore be understood as a project implementation, not a byte-for-byte reproduction of Google DeepMind’s production system or of all experiments reported in the paper.

### 3.2 Keyed, context-dependent seed generation

The seed generator uses a sliding window of the most recent \(H=4\) tokens and the watermarking key:

\[
r_t = h(x_{t-H},\ldots,x_{t-1},k), \qquad H=4.
\]

The repository implements this as a deterministic function: it takes the last four available token IDs, serializes them as comma-separated decimal integers, appends a vertical-bar separator and the integer key, hashes the resulting UTF-8 string with SHA-256, and converts the digest to a big-endian integer. Thus, the same context and key reproduce the same seed, while changes to either input generally produce a different seed.

SHA-256, integer key representation, delimiter-based serialization, and conversion of the digest to an integer are **TextSus implementation choices**. They specify this repository’s deterministic behavior; they are not presented as exact serialization or cryptographic-hash requirements of the paper. At the beginning of a sequence, fewer than four preceding tokens may be available, and the implementation hashes the available prefix rather than padding it to four tokens.

### 3.3 Pseudorandom g-values

For a candidate token ID \(v\), seed \(r_t\), and tournament layer \(\ell\), TextSus computes a deterministic pseudorandom value from the serialized tuple \((v,\ell,r_t)\). As with seed generation, the current code uses SHA-256 and a repository-defined UTF-8 representation. The resulting digest is mapped to one of two supported distributions:

- **Bernoulli option:** a value of 0 or 1 with probability one half each, obtained by dividing the hash range at its midpoint. This is the default and is the principal configuration used by the sampler and the score implementations.
- **Uniform option:** a deterministic value in \([0,1)\), obtained by normalizing the digest integer by the size of its range.

Distinct layers use distinct layer identifiers, so a token can receive different pseudorandom values at different tournament layers. The hash-based construction makes the values reproducible for scoring, conditional on the same token, seed, and layer. These concrete hash and mapping choices are implementation details; the method-level role is to supply pseudorandom g-values for candidate comparison and later detection.

### 3.4 Tournament Sampling

#### Single-layer tournament

For one layer and \(N\) competitors, the sampler starts with \(N\) independently drawn candidate tokens from the model distribution. It computes each candidate’s g-value for that layer, retains the candidate or candidates with the maximum value, and chooses uniformly among tied maxima. Uniform tie-breaking avoids an additional preference among tokens with equal g-values.

#### Multilayer tournament

For \(m\) layers and \(N\) competitors in each match, the initial candidate count is

\[
M=N^m.
\]

At each layer, candidates are arranged in groups of \(N\). The layer-specific g-value is computed for each candidate in a group, and one maximum-g candidate is selected, with uniform tie-breaking. The winners form the candidate set for the next layer. After \(m\) reductions, one token remains. For example, \(N=3\) and \(m=2\) require nine initial candidates and reduce them to three, then to one.

The core sampler accepts the already sampled token IDs and checks that their count is exactly \(N^m\). A separate distortionary sampler generalizes the number of competitors per match. The Hugging Face generation wrapper samples \(N^m\) candidates with replacement from the temperature-adjusted, top-k-filtered model distribution and passes them to the tournament sampler.

#### Non-distortionary and distortionary configurations

With exactly two competitors per match (\(N=2\)), Tournament Sampling is **single-token non-distortionary**: averaged over the random seed, the marginal distribution of the selected token matches the original model distribution. This property applies to an individual next-token choice; it does not by itself establish that a complete response or multiple responses retain their original sequence distribution.

When \(N>2\), selecting the largest g-value among a larger candidate set increases the selection bias toward high-g candidates. This is the **distortionary configuration**. It offers a method-level trade-off: more competitors can strengthen the watermark signal and improve detectability, while moving the output distribution further from the original model distribution and potentially reducing text quality or diversity. These effects are a rationale of the method and are not, by themselves, results of a TextSus controlled evaluation.

### 3.5 Repeated context masking and sequence non-distortion

Single-token non-distortion and sequence-level non-distortion are separate claims. The paper’s repeated context masking procedure extends the guarantee to \(K\) sequences by preventing a context window from being watermarked repeatedly within the tracked response history. With \(K=1\), the procedure targets single-sequence non-distortion; larger \(K\) values track context reuse across a correspondingly larger response history. In the intended generation flow, an eligible context is watermarked and recorded, while a repeated context bypasses watermark sampling for that step and uses the original model distribution.

TextSus includes helper functions to create a bounded history of response-context sets and to test whether a context has already occurred. The code dump does **not** show this history being populated and checked in the Hugging Face generation loop, nor does it show the bypass-to-original-distribution behavior integrated into that loop. Therefore, this chapter describes repeated context masking as a documented method/interface requirement with partial helper support, not as a completed generation feature. Likewise, the sequence-level non-distortion guarantee cannot be attributed to the current integrated generator based on the available implementation.

The distinction is consequential: \(N=2\) describes the sampler’s single-token marginal property; \(K\)-sequence non-distortion depends on repeated context masking in the generation process. Choosing \(N=2\) alone does not implement that sequence-level mechanism. The distortionary \(N>2\) setting remains distortionary even if a masking policy is added.

### 3.6 Watermark scoring

For a token sequence \(x_1,\ldots,x_T\), the mean score aggregates the g-values of each observed token under the reconstructed seed and across the \(m\) layers:

\[
S_{\mathrm{mean}}(x_{1:T})=
\frac{1}{Tm}\sum_{t=1}^{T}\sum_{\ell=1}^{m} g_{\ell}(x_t,r_t).
\]

The implementation obtains \(r_t\) from the preceding token prefix (the seed routine itself retains the last four tokens) and computes the default Bernoulli g-value for each layer. A higher mean indicates more observed g-value mass than the nominal Bernoulli null expectation of one half.

The weighted mean allows layers to contribute unequally. For nonnegative layer weights \(w_1,\ldots,w_m\), with positive total weight, the implemented score is

\[
S_{\mathrm{weighted}}(x_{1:T})=
\frac{1}{T\sum_{\ell=1}^{m}w_\ell}
\sum_{t=1}^{T}\sum_{\ell=1}^{m}w_\ell g_{\ell}(x_t,r_t).
\]

The repository checks that the weights are nonnegative and that at least one is positive. It also rejects empty token sequences and invalid layer counts in these score functions.

### 3.7 Frequentist and parameterized Bayesian evidence

The frequentist mean scorer treats the Bernoulli g-values under the unwatermarked null as Bernoulli\((0.5)\) observations. It counts the observed ones and returns the exact one-sided binomial upper-tail probability of obtaining at least that many successes. Smaller p-values indicate stronger evidence against the null. The weighted frequentist scorer applies the layer weights and standardizes the weighted mean using the implemented null mean and variance, then returns a one-sided normal upper-tail probability. The latter is a normal-approximation procedure, not an exact weighted-distribution calculation.

The Bayesian scorer computes a posterior probability from Bernoulli g-values and supplied parameters: layer biases \(\beta\), a square matrix of autoregressive layer-dependence parameters \(\delta\), and a prior probability (defaulting to 0.5 when omitted). It forms a layerwise likelihood ratio, accumulates log evidence, adds prior log-odds, and maps the posterior log-odds through a sigmoid. The scorer consumes these parameters; the repository code does not train or calibrate them. Its output is therefore parameterized evidence, not a trained or empirically validated Bayesian detector.

### 3.8 Threshold detection

For scores where larger values indicate stronger watermark evidence, the basic detector applies

\[
\text{watermarked} \iff S\geq \tau,
\]

where \(\tau\) is a caller-supplied threshold. For frequentist scores, the decision rule is

\[
\text{watermarked} \iff p\leq\alpha,
\]

where \(\alpha\) is the selected significance level. The detection module implements both threshold comparisons. Threshold calibration for a target false-positive rate is an evaluation task; the basic detector does not estimate an operating threshold from a corpus.

### 3.9 Module and interface organization

The implementation separates method components into focused modules:

| Component | Repository module or interface | Role |
|---|---|---|
| Seed generation | `textsus.seed.random_seed` | Context-window and key-dependent deterministic seed |
| G-value generation | `textsus.gvalues.gvalues` | Pseudorandom Bernoulli or uniform values by token, seed, and layer |
| Tournament sampling | `textsus.sampling.tournament` | Two-competitor single-layer and multilayer tournament logic |
| Distortionary sampling | `textsus.sampling.distortionary_tournament` | Configurable \(N\)-competitor matches |
| Context masking helpers | `textsus.sampling.context_masking` | Bounded history creation and repeated-context check |
| Generation wrapper | `textsus.generation.watermarked_generator` | Hugging Face next-token generation with tournament and baseline methods |
| Mean scoring | `textsus.scoring.mean_score`, `weighted_mean` | Unweighted and layer-weighted g-value aggregation |
| Statistical scoring | `textsus.scoring.frequentist`, `bayesian` | Frequentist p-values and parameterized Bayesian posterior |
| Detection | `textsus.detection.detector` | Score-threshold and p-value significance decisions |

The accompanying interface documents provide contracts and broader architectural intentions. Those contracts are not all implemented as complete public APIs. In particular, the dump does not establish a general production generation/detection service, repeated-context masking integration, Bayesian parameter training, or an end-to-end evaluation harness that reproduces the Nature paper’s full protocol.

### 3.10 Test status and implementation limitations

The latest test status reported for the project is **54 tests passing after the tournament generalization**. The code dump contains unit-test modules for the seed, g-values, tournament variants, context masking helpers, scoring, detector, and vectorized sampler, but it does not include a test-run log that independently confirms the reported count. This unit-test result supports software-level behavior covered by those tests; it is not evidence of scientific reproduction or paper-grade evaluation.

The repository also contains experiment scripts and result-table artifacts. Their presence alone does not establish that the paper’s models, datasets, operating points, comparison baselines, sample sizes, or statistical protocol were reproduced. No paper-grade detectability/quality result is claimed here. Other concrete scope limits include:

- The SHA-256 algorithms and string serialization are local implementation choices, not exact paper requirements.
- The repeated-context functions are helpers and are not wired into the generation loop.
- Bayesian scoring accepts parameters, but no parameter-training pipeline is established by the dump.
- The generation wrapper supports a Hugging Face model, temperature and top-k filtering, and watermarked or unwatermarked generation, but it is a limited project wrapper rather than proof of a complete production-grade or paper-faithful end-to-end system.
- The available vectorized tournament module vectorizes the tournament reduction over candidate arrays; its existence does not establish that the full generation pipeline is vectorized or that the paper’s computational-overhead claims have been reproduced.

These boundaries should guide interpretation of the method: the repository implements and unit-tests core components, while complete sequence-level masking integration, parameter learning, and rigorous empirical reproduction require additional work.

### References

Dathathri, S., See, A., Ghaisas, S., et al. “Scalable watermarking for identifying large language model outputs.” *Nature* 634, 818–823 (2024). [Nature article and DOI](https://doi.org/10.1038/s41586-024-08025-4).