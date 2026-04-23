# visual-factor-timing

Replication of **Jia, Li, Zhang & Zhao (2025), "Timing the Factor Zoo via Deep Visualization"** applied to the T2 and GDELT factor panels.

## Approach
Per-factor historical return trajectories are rendered as two complementary images — a JKX-style OHLC candlestick chart with a moving-average overlay and a monthly realized-volatility strip, and an MXX-style cumulative-return bar chart — then processed by CNN and CNN-LSTM networks to produce factor-specific timing weights.

## Data sources
External xlsx files live outside the repo:
- **T2 daily**   — `A Complete/T2 Factor Timing Fuzzy Daily/T2_Optimizer.xlsx`
- **T2 monthly** — `A Complete/T2 Factor Timing Fuzzy/T2_Optimizer.xlsx`
- **GDELT daily**   — `A Complete/GDELT Factor Timing Fuzzy Daily/T2-Factor-Timing-Daily/GDELT_Optimizer.xlsx`
- **GDELT monthly** — `A Complete/T2 GDELT/GDELT_Optimizer.xlsx`

Daily returns are stored in percent; the loader rescales before cumulating.

## Pipeline
```
factor_timing/
  data/factor_loader.py      daily returns → synthetic price → monthly OHLC + vol parquet
  imaging/build_cache.py     monthly OHLC → JKX image memmap cache (window=12 or 24)
  (planned)
  imaging/mxx_renderer.py    cumulative-return trajectory (1D CNN input)
  models/cnn.py              1D CNN (64 filters, kernel 5, SiLU, dropout 0.25)
  models/cnn_lstm.py         CNN + LSTM(64) variant
  train/targets.py           raw, σ-standardized, Φ⁻¹, percentile targets; EW/EWPM weighting
  train/ensemble.py          64-model IC-weighted ensemble
  train/clustering.py        expanding-window Ward k=8 clustering for GDELT pooled training
  backtest/factor_timing.py  IC tables, α tables, OOS R², HML quintile sort
  backtest/mechanism.py      LTA, risk aversion, macro PC analysis
```

## Build

```bash
# T2
python -m factor_timing.data.factor_loader --panel t2
python -m factor_timing.imaging.build_cache --panel t2 --window 12
python -m factor_timing.imaging.build_cache --panel t2 --window 24

# GDELT
python -m factor_timing.data.factor_loader --panel gdelt
python -m factor_timing.imaging.build_cache --panel gdelt --window 12
python -m factor_timing.imaging.build_cache --panel gdelt --window 24
```

## Training strategy
- **T2** (83 factors, ~270 months each): per-factor training per the paper.
- **GDELT** (86 factors, ~129 months each): pooled training in 8 groups discovered by expanding-window Ward clustering on the factor return correlation matrix — no forward leakage since the clustering is refit at each retrain date using only data available at that point.

## Reference
Jia, Y., Li, J., Zhang, H., & Zhao, J. (2025). Timing the Factor Zoo via Deep Visualization. *Working paper, March 24, 2025.*
