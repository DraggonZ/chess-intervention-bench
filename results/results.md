# Results

100 episodes; the unassisted side wins 0, draws 64 and loses 36. 3211 eligible turns (32.1 per episode). The Expert's move differs from the Worker's on 34% of turns; intervening helps on 8.7% and hurts on 0.5%.

| Policy | Mean score [95% CI] | Minus random timing [95% CI] | Share of achievable gain [95% CI] |
| --- | --- | --- | --- |
| Never intervene | 0.320 [0.270, 0.365] | -0.043 [-0.053, -0.034] | 0.00 [0.00, 0.00] |
| Random timing | 0.363 [0.317, 0.407] | 0.000 [0.000, 0.000] | 0.08 [0.06, 0.10] |
| Fixed move 30 | 0.390 [0.340, 0.440] | 0.027 [-0.004, 0.060] | 0.13 [0.07, 0.19] |
| First disagreement (uses Expert) | 0.495 [0.435, 0.555] | 0.132 [0.074, 0.192] | 0.33 [0.21, 0.44] |
| Random disagreement (uses Expert) | 0.452 [0.408, 0.494] | 0.089 [0.072, 0.107] | 0.25 [0.20, 0.29] |
| Best in hindsight | 0.855 [0.805, 0.900] | 0.492 [0.472, 0.515] | 1.00 [1.00, 1.00] |
| GPT-6 Luna | 0.375 [0.325, 0.420] | 0.012 [-0.018, 0.046] | 0.10 [0.04, 0.17] |
| gpt-oss-120b | 0.375 [0.320, 0.430] | 0.012 [-0.026, 0.052] | 0.10 [0.03, 0.18] |
| Qwen3 30B A3B | 0.380 [0.325, 0.430] | 0.017 [-0.019, 0.056] | 0.11 [0.04, 0.19] |
| Gemma 4 31B | 0.350 [0.300, 0.395] | -0.013 [-0.040, 0.018] | 0.06 [0.00, 0.12] |
| Jev 1.13 | 0.330 [0.280, 0.380] | -0.033 [-0.048, -0.015] | 0.02 [0.00, 0.05] |

| Model | Intervened | Median move | Token spent on an unchanged move (random timing) | Intervention helped (random timing) | Smallest detectable difference from random | Episodes to detect +0.066 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 100/100 | 22 | 70% (66%) | 12% (9%) | ±0.046 | 50 |
| gpt-oss-120b | 98/100 | 22 | 60% (66%) | 14% (9%) | ±0.055 | 69 |
| Qwen3 30B A3B | 100/100 | 18 | 64% (66%) | 13% (9%) | ±0.056 | 73 |
| Gemma 4 31B | 99/100 | 25 | 71% (66%) | 8% (9%) | ±0.039 | 35 |
| Jev 1.13 | 71/100 | 29 | 82% (66%) | 3% (9%) | ±0.024 | 13 |

| Model | Errors | Input tokens | Output tokens | Cost (USD) |
| --- | --- | --- | --- | --- |
| GPT-6 Luna | 0 | 415,906 | 705,736 | 0.39 |
| gpt-oss-120b | 0 | 439,851 | 2,102,337 | 1.33 |
| Qwen3 30B A3B | 0 | 81,297 | 963 | 0.01 |
| Gemma 4 31B | 0 | 843,492 | 1,581,769 | 0.60 |
| Jev 1.13 | 0 | 2,205,901 | 72,874 | 0.09 |
| GPT-6 Luna (repeats) | 0 | 1,230,035 | 2,076,911 | 1.16 |

GPT-6 Luna repeated 3 times on 100 episodes: mean scores 0.390, 0.370, 0.365; the same turn every time in 7% of episodes.
