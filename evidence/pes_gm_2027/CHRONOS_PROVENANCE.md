# Chronos checkpoint provenance

The study uses a frozen `amazon/chronos-t5-small` checkpoint in zero-shot mode on the delayed hourly-max HB_HUBAVG series. A pre-2021 tree-training split does **not** establish a pre-2021 foundation-model training cutoff. Do not claim that it does.

## Checkpoint

| Field | Value | Evidence |
| --- | --- | --- |
| Hugging Face id | `amazon/chronos-t5-small` | `models/chronos_delayed_cache.py`; `models/pes2027/chronos_cache.py` |
| Git revision pinned for new caches | `a971ba21945c4f1796b17a91fe69214b5f4ad472` | `evidence/pes_gm_2027/env/chronos_t5_small_revision.txt`; `models/pes2027/chronos_cache.py` `REVISION` |
| Architecture | T5 encoder-decoder, 46M parameters, vocab 4096 | `env/chronos_t5_small_config.json`; Hugging Face model card (accessed 2026-10-26) |
| `chronos_config.context_length` | 512 | same config |
| `chronos_config.prediction_length` | 64 (model default; the run uses horizon `1+d`) | same config |
| `num_samples` | 20 | config `chronos_config.num_samples`; cache builder |
| `temperature` | 1.0 | config |
| `top_k` | 50 | config |
| `top_p` | 1.0 | config |
| Tokenizer | `MeanScaleUniformBins`, limits [−15, 15] | config |
| Runtime dtype | bfloat16 | cache builder `torch_dtype=torch.bfloat16` |
| Device | CUDA, `CUDA_VISIBLE_DEVICES=0` for new runs | protocol |

The original cache builder (`models/chronos_delayed_cache.py`) calls `from_pretrained("amazon/chronos-t5-small")` **without** an explicit revision. The files actually loaded for the saved caches are the snapshot hashed in `env/chronos_snapshot_sha256.txt`. New E4 caches pin the revision above.

## Library versions in the evaluation environment

From `env/pip_freeze_ercot_venv.txt`:

| Package | Version |
| --- | --- |
| chronos-forecasting | 2.3.2 |
| torch | 2.6.0+cu124 |
| transformers | 4.57.6 |
| xgboost | 2.1.4 |

## Inference settings used in this study

- Context: 512 **clock** hours ending at t − (1 + d). Missing hours stay NaN; they are not dropped to compress the window.
- Horizon: `prediction_length = 1 + d`. The stored summaries are the samples at step index `d` (0-based), which is the path value for hour t.
- Seed: one `torch.manual_seed` call per run. The original run used seed 0 and could resume from a partial file (the random stream then depends on which hours remain). New E4 runs do not resume.
- Batch size: 256, same origin order as `_contexts`.
- Summaries written to the cache:
  - `chronos_pred`: `torch.median` (lower median of 20 even-count samples).
  - `chronos_q10`, `chronos_q90`: `torch.quantile`, linear interpolation.
  - `chronos_mean`: sample mean.
  - `chronos_p_gt200`: share of samples strictly greater than 200.
  - `chronos_spread`: q90 − q10, computed later in `chronos_block`.

## Pretraining corpus, as documented

Ansari et al., “Chronos: Learning the Language of Time Series,” *Transactions on Machine Learning Research*, 2024, https://openreview.net/forum?id=gerNCVqqtR, arXiv:2403.07815v1 (html retrieved 2026-10-26).

The paper’s Table 2 and Appendix B partition datasets into (a) pretraining only, (b) Benchmark I (in-domain train and test), and (c) Benchmark II (zero-shot evaluation only). **ERCOT Load is in Benchmark II.** Appendix B:

> ERCOT Load contains hourly energy load in 8 US regions between 2004 and 2021.

Tables 8 and 9 report WQL and MASE for ERCOT Load under Benchmark II. The paper states that Benchmark II datasets “were not used during Chronos models’ training.” A footnote adds that, rigorously, a zero-shot series should start after the last pretraining timestamp, and that the authors “consider the risk to be minimal given that the datasets bear no overlap beyond high-level conceptual categorization.”

ERCOT **prices** do not appear in Table 2. Spanish Energy and Weather includes Spanish prices; that is a different market.

What is **not** documented in the paper or the model card: a pretraining cutoff date, a statement that ERCOT prices were or were not crawled from some other public dump, or the exact snapshot date of the Hugging Face weights relative to 2021. Those items are [MISSING].

## Overlap with this study

| Series | In Chronos pretraining (published list) | In Chronos Benchmark II | Used here |
| --- | --- | --- | --- |
| ERCOT hourly load, 8 weather zones, 2004–2021 | No | Yes, zero-shot | Realized load is an XGBoost feature, not a Chronos input |
| ERCOT HB_HUBAVG 15-minute / hourly-max prices | Not listed | Not listed | Chronos input and target |
| February 2021 prices | Not listed | Not listed | Retrospective stress diagnostic only (E8 / original crisis run) |

The Chronos context in this study is the delayed **price** series. Seeing ERCOT Load in Benchmark II is not evidence that Chronos was trained on ERCOT prices, and it is not evidence that it was not. The published corpus list simply does not include those prices.

## 2021 is a retrospective stress diagnostic

The February 2021 window is a single historically salient episode used to describe tail behavior. It is not an operational real-time test, and it is not a claim that the foundation-model weights predate 2021. Tree models in the original crisis run train on pre-2021 rows; that fact does not transfer to Chronos.

## Citation record (verified)

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D. C., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., and Wang, Y. (2024). Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=gerNCVqqtR
