# Seed-1 replication — arms 1 and 3 at long context

Paired within seed: `two-sided-Lext(32k)` minus `two-sided-2048`, same seed, same corpus. Negative = Lext better.

## pg19

| pair | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| seed 0 Lext-base | +0.0086 | +0.0007 | -0.0057 | -0.0080 | -0.0109 | -0.0156 |
| seed 1 Lext-base | -0.0040 | -0.0075 | -0.0089 | +0.0023 | +0.0159 | +0.0156 |
| seed 0 Lext(8k)-base | +0.0116 | +0.0117 | +0.0084 | +0.0032 | -0.0040 | -0.0128 |

## wikitext2

| pair | 0-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | slope |
|---|---|---|---|---|---|---|
| seed 0 Lext-base | +0.0030 | -0.0146 | -0.0161 | -0.0150 | -0.0158 | -0.0163 |
| seed 1 Lext-base | -0.0031 | -0.0154 | -0.0174 | -0.0137 | +0.0068 | +0.0086 |
| seed 0 Lext(8k)-base | +0.0054 | -0.0066 | -0.0085 | -0.0027 | -0.0006 | -0.0057 |

## Raw 16-32k loss ratios

| arm | pg19 | wikitext2 |
|---|---|---|
| two-sided-2048 | 1.3625 | 1.2871 |
| one-sided | 1.3267 | 1.2562 |
| two-sided-Lext | 1.3517 | 1.2713 |
| two-sided-longcalib | 1.3489 | 1.2863 |
| two-sided-2048_s1 | 1.3319 | 1.2626 |
| two-sided-Lext_s1 | 1.3477 | 1.2694 |
| two-sided-Lext8k_s0 | 1.3585 | 1.2864 |

## Verdict — the Lext effect does not replicate

16-32k loss ratio to FP:

                          pg19    wikitext2
  two-sided-2048  seed0  1.3625     1.2871
  two-sided-2048  seed1  1.3319     1.2626
  Lext(32k)       seed0  1.3517     1.2713
  Lext(32k)       seed1  1.3477     1.2694
  one-sided       seed0  1.3267     1.2562

Paired Lext - baseline at 16-32k:
  pg19       s0 -0.0109   s1 +0.0159   mean +0.0025   SIGN FLIPS
  wikitext2  s0 -0.0158   s1 +0.0068   mean -0.0045   SIGN FLIPS

The slope difference flips with it (seed 0: Lext shallower by 0.016 on both corpora;
seed 1: Lext STEEPER by 0.016 / 0.009). So the seed-0 result -- Lext closing 51% of
the arm1-arm2 gap on wikitext2 and 30% on PG19 -- was within seed noise, and the
two-seed mean is ~0 on both corpora.

The scale that makes this legible: the BASELINE's own seed-to-seed spread at 16-32k
is 0.0307 (pg19) and 0.0244 (wikitext2), i.e. two to three times the Lext effect
that was being claimed. The whole Phase B comparison was being read at a resolution
finer than the seed noise of a single arm.

That also undercuts the one-sided result, which only ever had seed 0. Its gap to the
baseline is -0.0358 (pg19) and -0.0309 (wikitext2), only 1.17x and 1.26x the
baseline's seed spread. It is the largest effect seen here and it is still the same
order as the noise; it needs seeds 1-2 of the one-sided arm before it can be called
anything. Those were not run -- the brief's next-step list specified seeds for arms
1 and 3 only.

What survives: nothing about Lext. The Phase A length dependence of H_out is real and
analytically verified, but its effect on long-context loss is not resolvable at n=2
seeds on this model.
