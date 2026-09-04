# Phase B — long-document eval on PG19, against wikitext2

Qwen2.5-0.5B W3 seed 0. Arms differ only in q/k H_out. PG19 test via the emozilla parquet mirror, 16 books, first 32768 tokens of each. proof-pile skipped: its HF loader errored, and the brief allowed at most five minutes.

## Gate 0 on PG19 — is FP flat to 32k?

| bucket | FP nll | vs 0-2k |
|---|---|---|
| 0-2048 | 2.9797 | 1.000x |
| 2048-4096 | 3.0257 | 1.015x |
| 4096-8192 | 2.9658 | 0.995x |
| 8192-16384 | 2.9389 | 0.986x |
| 16384-32768 | 2.9034 | 0.974x |

**Gate 0: PASS** — worst bucket 1.015x the 0-2k loss.

## Loss ratio to FP — PG19 (16 books)

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| two-sided-2048 | 1.2737 | 1.2415 | 1.2672 | 1.2884 | 1.3625 | 1.0697 |
| one-sided | 1.2897 | 1.2454 | 1.2675 | 1.2844 | 1.3267 | 1.0287 |
| two-sided-Lext | 1.2823 | 1.2422 | 1.2615 | 1.2804 | 1.3517 | 1.0541 |
| two-sided-longcalib | 1.2809 | 1.2413 | 1.2598 | 1.2829 | 1.3489 | 1.0531 |

## Loss ratio to FP — wikitext2 (9 seqs)

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| two-sided-2048 | 1.1625 | 1.1911 | 1.2072 | 1.2358 | 1.2871 | 1.1072 |
| one-sided | 1.1747 | 1.1923 | 1.2037 | 1.2284 | 1.2562 | 1.0693 |
| two-sided-Lext | 1.1655 | 1.1765 | 1.1911 | 1.2208 | 1.2713 | 1.0908 |
| two-sided-longcalib | 1.1678 | 1.1938 | 1.2042 | 1.2437 | 1.2863 | 1.1015 |

## KL(FP || arm) — PG19

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| two-sided-2048 | 0.8256 | 0.7495 | 0.8143 | 0.8694 | 1.0714 | 1.0697 |
| one-sided | 0.8733 | 0.7578 | 0.8108 | 0.8548 | 0.9653 | 1.0287 |
| two-sided-Lext | 0.8489 | 0.7501 | 0.7947 | 0.8448 | 1.0393 | 1.0541 |
| two-sided-longcalib | 0.8527 | 0.7514 | 0.7893 | 0.8523 | 1.0336 | 1.0531 |

## KL(FP || arm) — wikitext2

| arm | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| two-sided-2048 | 0.4322 | 0.4613 | 0.5128 | 0.5355 | 0.7261 | 1.1072 |
| one-sided | 0.4639 | 0.4657 | 0.5034 | 0.5207 | 0.6464 | 1.0693 |
| two-sided-Lext | 0.4363 | 0.4367 | 0.4754 | 0.5010 | 0.6885 | 1.0908 |
| two-sided-longcalib | 0.4514 | 0.4817 | 0.5058 | 0.5513 | 0.7207 | 1.1015 |

## Standard 2k PPL

| arm | wiki2 | c4-new |
|---|---|---|
| FP | 13.078 | 20.407 |
| two-sided-2048 | 19.717 | 39.072 |
| one-sided | 20.345 | 40.827 |
| two-sided-Lext | 19.797 | 39.022 |
| two-sided-longcalib | 19.614 | 38.742 |

## Pre-registered reading

The finding stands only if the 16-32k ordering `one-sided < Lext < two-sided-2048` reproduces on PG19 AND the arm1-vs-arm2 gap is at least as large as on wikitext2.

| corpus | one-sided | Lext | two-sided-2048 | ordering holds | arm1-arm2 gap |
|---|---|---|---|---|---|
| PG19 | 1.3267 | 1.3517 | 1.3625 | yes | +0.0358 |
| wikitext2 | 1.2562 | 1.2713 | 1.2871 | yes | +0.0309 |

**Verdict: HOLDS** — ordering on PG19 reproduces; gap +0.0358 vs +0.0309 on wikitext2 (>= required).


## H_infinity check (task 3)

Claim: as L -> infinity, off-diagonal frequency-pair blocks of H_out vanish and each
diagonal block collapses to alpha_i * I2. The J component of a diagonal block is
(m21 - m12)/2, which is exactly 0 for a symmetric C, and the pre-RoPE covariance is
symmetric to 0.0e+00 -- so the claim is structurally right.

It is NOT reached at L = 1e6 for Qwen2.5-0.5B. Convergence is set by 1/theta_min
and by the closest frequency pair:

    1/theta_min                 = 6.49e5
    1/min|theta_i - theta_j|    = 1.20e6   (the pair i=30, j=31)

    L        off-diag / ||H||max    diag vs alpha*I2   min|dtheta|*L   min(th+th)*L
    2048            1.71e-01            1.71e-01            0.00           0.01
    32768           1.48e-01            1.67e-01            0.03           0.10
    1e6             7.84e-02            2.78e-03            0.83           3.08
    1e7             1.95e-03            6.97e-05            8.31          30.80
    1e8             1.89e-05            5.72e-07           83.14         307.99
    1e9             2.01e-07            7.13e-09          831.45        3079.85

Tolerance 1e-3 on the off-diagonal blocks needs L ~ 1e7; 1e-5 needs L ~ 1e8. The
error falls as (L*theta)^-2.

The exception the brief asked about DOES occur: exactly one pair, (i,j) = (30,31),
has |theta_i - theta_j| * L < 1 at L = 1e6 -- |dtheta|*L = 0.831, F_L = 0.9437, i.e.
that off-diagonal block is essentially UNattenuated and is what holds the residual
at 7.8e-2. No sum-frequency pair is affected (min (theta_i + theta_j)*L = 3.08).

So H_infinity = diagonal is asymptotically true but is not a good description at any
context length a model will see; at 32k the off-diagonal mass is still 1.5e-1.
