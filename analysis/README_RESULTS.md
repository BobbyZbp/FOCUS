# BT-CCQ Result Checklist

## Main task
- `antmaze-large-diverse-v2`
- (later) `antmaze-large-play-v2`

## Runs (per env)
- `reduced_wsrl_seed{0,1,2}`
- `reduced_btccq_seed{0,1,2}`
- `full_wsrl_seed{0,1,2}`
- `full_btccq_seed{0,1,2}`

## Required plots
1. **Online return / success curve** — `evaluation/average_normalized_return` vs step
2. **Q diagnostic** — `q_diag/q_eval_mean` and `q_diag/q_eval_max` over online steps
3. **BT-CCQ gate stats** — `btccq/gate_frac`, `btccq/gate_mean`, `btccq/delta_mean`, `btccq/q_hat`
4. **Calibration sanity** — at step 0: `btccq/calib_e_mean`, `btccq/calib_e_p90`, `btccq/calib_zero_frac`

## Main table

| Method | Warmup | Early-Online Q-Dip (∆Q at 5k) | Final Return | Gate Peak | Notes |
|---|---|---|---|---|---|
| WSRL    | reduced | ? | ? | n/a | baseline |
| BT-CCQ  | reduced | ? | ? | ?   | test     |
| WSRL    | full    | ? | ? | n/a | upper-bound baseline |
| BT-CCQ  | full    | ? | ? | ?   | matched comparison   |

## Story

- **Reduced-warmup**: BT-CCQ should reduce early Q-dip and recover return faster than WSRL.
- **Full-warmup**: BT-CCQ should at minimum not hurt; ideally still improve early Q-dip.

## Health checks before trusting numbers

| Metric | Bad sign | Action |
|---|---|---|
| `btccq/q_hat == 0` and `calib_zero_frac > 0.95` | calibration degenerate | lower `--btccq_alpha` to 0.05 or floor q_hat |
| `btccq/gate_frac == 0` always | gate never fires | raise `--btccq_alpha` to 0.2 |
| `btccq/gate_frac > 0.8` always | gate too aggressive | raise `--btccq_w_out` to 0.4 |
| `critic_loss` NaN | numerical issue | lower lr or check reward_scale |
| `q_diag/q_eval_max` exploding (>1e5) | classic Q-collapse | this is what BT-CCQ should mitigate — note it for the paper |
