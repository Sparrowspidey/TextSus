# Chapter 4: Experimental Evaluation and Results

This chapter describes how the TextSus implementation of SynthID-Text was evaluated and what the experiments showed. All numbers come from the scripts in `experiments/` and are stored in `results/tables/` (figures in `results/figures/`).

## 4.1 Experimental setup

| Item | Setting |
|---|---|
| Model | GPT-2 (CPU). Gemma access was set up, but runs were not performed on Gemma. |
| Prompts | ELI5 questions (`sentence-transformers/eli5`, "question" column), split into dev/test files by `experiments/prepare_eli5_data.py` |
| Sampling | top-k = 100 (500 where stated), temperature 0.7 |
| Watermark key / seed window | key = 42; sliding window H = 4 |
| Tournament layers | m = 4 unless stated (16 candidates per token) |
| Detection score | mean g-value (Bernoulli(0.5)) over tokens and layers |
| Detectability metric | TPR at a fixed 1% FPR; the threshold is the 99th percentile of unwatermarked scores |
| Quality metrics | perplexity, Self-BLEU (diversity) |

The paper uses m = 30 layers and thousands of prompts on production hardware. This project uses far fewer layers and prompts so that every experiment runs on a laptop CPU. The sample sizes below (10 to 50 prompts) are therefore much smaller than the paper's, and the results should be read as a small-scale reproduction.

The watermarked generator (`WatermarkedGenerator`) runs a standard token-by-token loop. At each step it draws 2^m candidate tokens from the (top-k, temperature-filtered) model distribution and picks one with Tournament sampling. The same loop supports the Gumbel and Soft Red List baselines, and an unwatermarked mode that samples directly.

## 4.2 Detectability

For each prompt, one watermarked and one unwatermarked response were generated and scored with the mean g-value.

| Run (GPT-2) | Prompts | Tokens | TPR @ 1% FPR |
|---|---|---|---|
| First end-to-end run | 5 | 60 | 1.000 |
| Larger run | 50 | 100 | 0.960 (threshold 0.577) |
| Re-run with the team's final modules | 20 | 80 | 0.900 (threshold 0.574) |

Watermarked responses scored about 0.60–0.70, while unwatermarked responses stayed around 0.45–0.58, close to the 0.5 expected for Bernoulli(0.5) g-values. The watermark is therefore clearly detectable with no access to the model. One degenerate pair (scores 0.284 for both) appeared in the 20-prompt run, which suggests a very short or repetitive generation.

## 4.3 Text quality and diversity

Using 15 prompts and 80 tokens:

| | Watermarked | Unwatermarked |
|---|---|---|
| Mean perplexity | 8.50 | 10.11 |
| Self-BLEU (higher = less diverse) | 3.90 | 4.27 |

The differences go in the "watermarked is slightly better" direction, which is not meaningful at n = 15 given the very large spread of per-sample perplexities (from about 3.9 to over 20). The correct reading is that no degradation of fluency or diversity was observed, which matches the paper's claim for the non-distortionary configuration. This is not evidence that the watermark improves quality. No human evaluation was done.

## 4.4 Latency

Timing 100 generated tokens, 3 runs per method (GPT-2, CPU):

| Method | ms / token | Overhead vs unwatermarked |
|---|---|---|
| Unwatermarked | 88.71 | - |
| Tournament (m = 4) | 86.60 | -2.4% |
| Gumbel | 87.87 | -0.9% |
| Soft Red List | 87.27 | -1.6% |

The standard deviations (0.8 to 1.9 ms) are as large as the differences, so the apparent speed-ups are timing noise. The practical conclusion is that the watermarking computation (a few SHA-256 hashes per candidate) is negligible next to the model forward pass of about 88 ms, consistent with the paper's report of well under 1% overhead. A tighter measurement would need more runs.

## 4.5 Detectability versus text length (SynthID-Text vs Gumbel)

At temperature 0.7 with 10 prompts, one long generation per method was truncated to 50, 100 and 200 tokens:

| Tokens | Tournament TPR | Gumbel TPR |
|---|---|---|
| 50 | 0.9 | 1.0 |
| 100 | 0.9 | 1.0 |
| 200 | 0.9 | 1.0 |

Both curves are flat, unlike the paper's Fig. 3a where TPR rises with length. This is a statistical artifact of the sample size: with 10 unwatermarked scores, the 1%-FPR threshold is essentially the maximum of those 10 values, so the TPR saturates and cannot show a length trend. The experiment confirms that the pipeline works but does **not** reproduce the paper's length dependence or the ranking between methods. A reliable comparison would need hundreds of prompts.

## 4.6 Ablation: number of tournament layers

Mean watermark score of watermarked text (10 prompts, 80 tokens, GPT-2):

| m | top-k = 100 (initial run) | top-k = 500 |
|---|---|---|
| 1 | 0.675 | 0.641 |
| 2 | 0.646 | 0.653 |
| 4 | 0.649 | 0.653 |
| 6 | 0.631 | 0.640 |
| 8 | 0.620 | 0.636 |

TPR@1%FPR with top-k = 500 was 0.6 at m = 1 and 1.0 for every m from 2 to 8. With top-k = 100 it was 1.0 throughout.

The initial run showed a steady **decline** in score as m grew, contrary to the paper's diminishing-returns curve. The cause is that 2^m candidates are drawn with replacement from only top-k distinct tokens. At m = 8 there are 256 candidates drawn from at most 100 distinct tokens, so many matches are between identical tokens. Identical tokens have identical g-values, so the match is a pure coin flip and gives no watermark signal, which pulls the average toward the 0.5 baseline. With top-k = 500, which comfortably exceeds the largest candidate pool, the score rises from m = 1 to a peak around m = 2 to 4 and then levels off, and TPR climbs from 0.6 to 1.0. This is the expected diminishing-returns pattern. The finding shows that the entropy of the candidate pool bounds how many layers are useful, in line with the paper's statement that detectability depends on the entropy of the model's distribution. The top-k = 100 numbers come from the first run; its JSON was later overwritten by the top-k = 500 run, so only the second run is stored in the repository.

## 4.7 Distortionary SynthID-Text versus Soft Red List

Both methods have a strength setting that trades quality for detectability. With 10 prompts and 80 tokens, the unwatermarked mean log-perplexity was 2.406.

| Method | Setting | Score | TPR @ 1% FPR | Mean log-perplexity |
|---|---|---|---|---|
| Tournament | 2 competitors/match (non-distortionary) | 0.633 | 1.0 | 2.409 |
| Tournament | 3 competitors | 0.728 | 1.0 | 2.420 |
| Tournament | 4 competitors | 0.763 | 1.0 | 2.474 |
| Soft Red List | delta = 0 | z = -0.46 | 0.1 | 2.412 |
| Soft Red List | delta = 1 | z = 2.46 | 0.9 | 2.538 |
| Soft Red List | delta = 2 | z = 4.83 | 1.0 | 2.508 |
| Soft Red List | delta = 4 | z = 7.85 | 1.0 | 2.703 |

Three observations follow.
1. Soft Red List with delta = 0 applies no bias and is correctly undetectable (TPR 0.1, z close to 0), which is a useful sanity check of the implementation.
2. Tournament sampling reached TPR 1.0 at every strength with a log-perplexity increase of at most 0.07 above unwatermarked, while Soft Red List needed delta of 2 or more for TPR 1.0 and cost between 0.10 and 0.30. This matches the direction of the paper's Fig. 3c: SynthID-Text gives a better detectability-quality trade-off.
3. The Tournament TPR values are all 1.0, so they cannot show the rise in watermark strength within the sweep. That rise is visible in the mean score, which grows steadily from 0.633 to 0.763. The small sample (n = 10) also makes the log-perplexity values noisy: for example, Soft Red List delta = 1 shows a higher log-perplexity than delta = 2.

## 4.8 Summary of findings

- Tournament sampling produces text that is reliably detectable from the tokens and key alone (TPR 0.90 to 1.0 at 1% FPR in the detection runs).
- No degradation of perplexity or diversity was observed for the non-distortionary configuration.
- Watermarking overhead was below measurement noise relative to the model forward pass.
- The number of useful layers is limited by the candidate pool size (top-k); with top-k too small, extra layers reduce the signal.
- At a comparable detection level, distortionary Tournament sampling cost less quality than Soft Red List in this small-scale test.

---

# Chapter 5: Limitations, Conclusion and Future Work

## 5.1 Limitations of this study

- **Model and scale.** All experiments used GPT-2 on CPU with 5 to 50 prompts. Gemma 2B/7B and Mistral 7B, used in the paper, were not run. With samples this small, the 1%-FPR threshold is unstable and TPR saturates at 0.9 or 1.0, which is why the length sweep and the layer ablation cannot show fine trends in TPR.
- **Reduced scale of the method.** The paper uses 30 layers on production infrastructure. Here m is at most 8, and candidates are sampled directly with replacement, which exposes the duplicate-token effect described in Section 4.6. A vectorized implementation (`vectorized_tournament.py`) exists in the repository, but the experiments use the scalar version, and no speed-up was measured.
- **Context masking.** The repeated-context-masking helpers and their tests exist, but the generator does not call them. The sequence-level non-distortion guarantee therefore does not apply to the generated text in these experiments; only the single-token guarantee (two competitors per match) is relevant.
- **Quality evidence.** Quality was measured with perplexity and Self-BLEU on 15 prompts. No human evaluation and no large-scale user study were done, unlike the paper's roughly 20 million Gemini responses.
- **Speculative sampling and robustness.** Integration with speculative sampling, and the effect of editing, paraphrasing, stealing or spoofing attacks, were not evaluated. The paper notes that edits and paraphrasing weaken the watermark.
- **Practical constraints inherited from the method.** Generative watermarking needs the text provider to apply it, so it cannot identify text from unwatermarked or open-source models, and detection is weaker when the model's distribution has low entropy.

## 5.2 Conclusion

TextSus re-implements the SynthID-Text pipeline end to end: a keyed sliding-window seed generator, pseudorandom g-values, multilayer Tournament sampling (with a configurable number of competitors), several scoring and detection methods, and Gumbel and Soft Red List baselines. A generation and evaluation framework was built around it. At small scale, the experiments support the paper's main qualitative claims: watermarked text is detectable without the model, shows no measurable quality or latency cost in its non-distortionary form, and offers a more favorable detectability-quality trade-off than Soft Red List when made distortionary. They also show a practical constraint not obvious from the paper's description: the useful number of layers is limited by how many distinct candidates the sampling distribution can supply. The experiments are too small to reproduce the paper's quantitative curves, and this is stated openly.

## 5.3 Future work

1. Repeat the detectability, length and temperature experiments with Gemma 2B/7B and hundreds of prompts, so that TPR curves are no longer saturated.
2. Integrate repeated context masking into the generator and test K-sequence non-distortion.
3. Use the vectorized tournament in generation and measure its speed-up and its behavior at larger m.
4. Add human or large-scale preference evaluation of quality.
5. Evaluate robustness to paraphrasing and editing, and implement watermarked speculative sampling.
6. Evaluate the Bayesian and frequentist scorers (already implemented) in the detection experiments, in place of the plain mean score.

---



## Results

All results are from GPT-2 on CPU with 10 to 50 prompts, so they are a small-scale check of the method, not a reproduction of the paper's numbers. Full tables are in `results/tables/` and plots in `results/figures/`.

| Experiment | Script | Main result |
|---|---|---|
| Detectability | `run_detectability.py` | TPR @ 1% FPR = 0.90 to 1.0; watermarked scores ~0.6 to 0.7 vs ~0.5 unwatermarked |
| Quality | `run_quality.py` | Mean perplexity 8.5 (watermarked) vs 10.1; Self-BLEU 3.9 vs 4.3; no degradation observed |
| Latency | `run_latency.py` | ~87 ms/token for all methods; overhead within noise |
| Length / temperature | `run_length_temperature_sweep.py` | Flat TPR (0.9 Tournament, 1.0 Gumbel) because n = 10 is too small for a stable 1% threshold |
| Layer ablation | `run_layer_ablation.py` | With top-k = 500, score rises then plateaus; with top-k = 100 it declines because of duplicate candidates |
| Distortionary vs Soft Red List | `run_distortionary_comparison.py` | Tournament reaches TPR 1.0 with log-perplexity +0.003 to +0.07; Soft Red List needs delta >= 2 at +0.10 to +0.30 |

Known limitations: GPT-2 only, small samples, context masking not wired into the generator, and speculative sampling and paraphrase robustness not evaluated.