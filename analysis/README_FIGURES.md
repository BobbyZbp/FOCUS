# Figure generation

After all 4 cells finish, run:

```bash
cd /workspace/wsrl
python analysis/make_figures.py
```

Figures written to `analysis/figures/` (PDF + PNG):
- `fig1_learning_curves.pdf` — success_rate vs step, 4 cells overlay
- `fig2_q_collapse.pdf` — q_eval_max vs step, 4 cells overlay
- `fig3_gate_trajectory.pdf` — gate_frac and gate_mean over training, BT-CCQ only
- `fig4_calibration.pdf` — q_hat / q_tail / q_scale / positive_frac at transition

## Required wandb run names (use as `--exp_name`)

The script looks for runs containing these substrings in their name:

| Cell | exp_name to use |
|---|---|
| Full WSRL baseline | `wsrl_full_medium_redq_seed0` |
| Full BT-CCQ | `btccq_full_medium_redq_seed0` |
| Reduced WSRL baseline | `wsrl_reduced_medium_redq_seed0` |
| Reduced BT-CCQ | `btccq_reduced_medium_redq_seed0` |

If you used different names, override with CLI:
```bash
python analysis/make_figures.py \
    --wsrl_full <substring1> \
    --btccq_full <substring2> \
    --wsrl_reduced <substring3> \
    --btccq_reduced <substring4>
```

## Expected wandb metrics (script will warn if any missing)

- `evaluation/success_rate` (every eval_interval, all 4 cells)
- `q_diag/q_diag/q_eval_max` (every eval_interval, all 4 cells)
- `training/critic/btccq/gate_frac` (every log_interval, BT-CCQ cells only)
- `training/critic/btccq/gate_mean` (every log_interval, BT-CCQ cells only)
- `btccq_calibration/btccq/q_hat` (single value at transition, BT-CCQ cells only)
- `btccq_calibration/btccq/q_tail`
- `btccq_calibration/btccq/q_scale`
- `btccq_calibration/btccq/calib_positive_frac`
- `btccq_calibration/btccq/calib_zero_frac`
