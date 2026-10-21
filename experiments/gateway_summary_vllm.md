## Gateway kill trials

| run_id | engine | replicas | mode | fsync | temperature | kill_at | trials | exactly_once | duplicated_positions | missing_positions | complete | matches_reference | reference_repeatable | recovery_s | attempts | git_commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261021T010704Z_gw_vllm_r1_ids_T0.0_k25 | vllm | 1 | ids | False | 0.0 | 25 | 10 | 10 | 0 | 0 | 10 | 0 | False | 107.62 (105.62-109.12) | 1 (1-1) | 8f7d849 |
| 20261021T012835Z_gw_vllm_r1_ids_T0.0_k128 | vllm | 1 | ids | False | 0.0 | 128 | 10 | 10 | 0 | 0 | 10 | 0 | False | 107.63 (106.62-108.65) | 1 (1-1) | 8f7d849 |
| 20261021T015008Z_gw_vllm_r1_ids_T0.0_k230 | vllm | 1 | ids | False | 0.0 | 230 | 10 | 10 | 0 | 0 | 10 | 0 | False | 108.41 (107.16-109.66) | 1 (1-1) | 8f7d849 |
| 20261021T021147Z_gw_vllm_r1_ids_T0.8_k25 | vllm | 1 | ids | False | 0.8 | 25 | 10 | 10 | 0 | 0 | 10 | 0 | False | 108.42 (107.15-109.74) | 1 (1-1) | 8f7d849 |
| 20261021T023327Z_gw_vllm_r1_ids_T0.8_k128 | vllm | 1 | ids | False | 0.8 | 128 | 10 | 10 | 0 | 0 | 10 | 0 | False | 181.81 (163.77-188.31) | 1 (1-1) | 8f7d849 |
| 20261021T030730Z_gw_vllm_r1_ids_T0.8_k230 | vllm | 1 | ids | False | 0.8 | 230 | 10 | 10 | 0 | 0 | 10 | 0 | False | 184.83 (158.30-187.85) | 1 (1-1) | 8f7d849 |
| 20261021T034247Z_gw_vllm_r1_ids_T0.0_k128 | vllm | 1 | ids | False | 0.0 | 128 | 10 | 10 | 0 | 0 | 10 | 10 | True | 173.03 (121.17-177.29) | 1 (1-1) | 8f7d849 |

## Gateway overhead (failure-free; median (min-max) over pairs; diff = gateway - direct per pair)

| run_id | engine | mode | fsync | pairs | direct_ttft_ms | gateway_ttft_ms | diff_ttft_ms | direct_tpot_ms | gateway_tpot_ms | diff_tpot_ms | direct_e2e_s | gateway_e2e_s | diff_e2e_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261021T041803Z_gw_vllm_r1_ids_T0.0_overhead | vllm | ids | False | 30 | 40.84 (40.62-41.56) | 79.60 (79.28-80.60) | 38.80 (37.98-39.74) | 33.11 (33.11-33.11) | 33.16 (33.16-33.16) | 0.05 (0.04-0.05) | 8.48 (8.48-8.48) | 8.54 (8.53-8.54) | 0.05 (0.05-0.05) |
| 20261021T042944Z_gw_vllm_r1_ids_fsync_T0.0_overhead | vllm | ids | True | 30 | 40.90 (40.53-41.50) | 85.63 (84.92-88.77) | 44.76 (44.11-47.80) | 33.11 (33.11-33.11) | 33.17 (33.16-33.17) | 0.06 (0.05-0.06) | 8.48 (8.48-8.49) | 8.54 (8.54-8.55) | 0.06 (0.06-0.06) |
