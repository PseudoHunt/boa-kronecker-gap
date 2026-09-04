# Phase B — does H_out's length dependence show up in loss at long context?

Qwen2.5-0.5B, W3, seed 0. Arms differ ONLY in q/k H_out; damping, grid, ordering, calibration budget all identical to the banked `boa_w3_s0`.

**PG19 was skipped by request** (the DeepMind loader was still fetching after 40 min), so the long-context result rests on concatenated wikitext2 alone. One corpus, not two.

## Standard 2k PPL

| arm | wiki2 | c4-new | wiki2 vs arm 1 |
|---|---|---|---|
| FP | 13.078 | 20.407 |  |
| two-sided-2048 | 19.717 | 39.072 | +0.000 |
| one-sided | 20.345 | 40.827 | +0.628 |
| two-sided-Lext | 19.797 | 39.022 | +0.080 |
| two-sided-longcalib | 19.614 | 38.742 | -0.103 |

## Gate 0 — is FP itself flat to 32k?

| bucket | FP nll | vs 0-2k |
|---|---|---|
| 0-2048 | 2.5182 | 1.000x |
| 2048-4096 | 2.4061 | 0.955x |
| 4096-8192 | 2.3657 | 0.939x |
| 8192-16384 | 2.2377 | 0.889x |
| 16384-32768 | 2.4728 | 0.982x |

**Gate 0: PASS** — FP's worst bucket is 1.000x its 0-2k loss.

## Per-position loss, ratio to FP in the same bucket

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 |
|---|---|---|---|---|---|
| two-sided-2048 | 1.1625 | 1.1911 | 1.2072 | 1.2358 | 1.2871 |
| one-sided | 1.1747 | 1.1923 | 1.2037 | 1.2284 | 1.2562 |
| two-sided-Lext | 1.1655 | 1.1765 | 1.1911 | 1.2208 | 1.2713 |
| two-sided-longcalib | 1.1678 | 1.1938 | 1.2042 | 1.2437 | 1.2863 |

## Per-position KL(FP || arm)

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 |
|---|---|---|---|---|---|
| two-sided-2048 | 0.4322 | 0.4613 | 0.5128 | 0.5355 | 0.7261 |
| one-sided | 0.4639 | 0.4657 | 0.5034 | 0.5207 | 0.6464 |
| two-sided-Lext | 0.4363 | 0.4367 | 0.4754 | 0.5010 | 0.6885 |
| two-sided-longcalib | 0.4514 | 0.4817 | 0.5058 | 0.5513 | 0.7207 |

