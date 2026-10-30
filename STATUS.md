# FabricInspect-IoT – project status (2026-10-30, evening)

Target journal: **IEEE Internet of Things Journal** (IEEEtran LaTeX in `paper/ieee/`, compile on Overleaf).

## Done (final, valid)
- Detector results: `fewshot.json`, `threshold.json`, `ablation.json`, `robust.json` (perturbations at 256 px),
  `iot_cascade.json` (gate trade-off), `iot_profile.json` (profile sizes/distribution) – accuracy results do not
  depend on host speed.
- Paper text: sec1–5, sec6a (few-shot + qualitative fig), sec6b (gate), sec6c (resources, ablation),
  sec6e (profile distribution, robustness), sec7 (discussion), sec8 (conclusion with one [TBD]); refs.bib (37).
- Figures: fig_architecture, fig_fewshot_image/aupro, fig_gate, fig_qualitative. Tables in `paper/ieee/tables/`.
- Tools: `tools/progress.py` (progress window), `experiments/device_bench.py` (real-device station benchmark).

## Why the live-system timing is not done
The laptop throttles (battery / Balanced mode / persistent low-power state after many hours of load) by 4-8x;
all hour-long live sweeps were invalidated by the speed check (`*_INVALID.json`, `*_balanced_mode.json`,
`iot_live_run*.json` – do not use). Decision: **trace-driven simulation validated on short live runs**.

## To do after a full Windows restart (AC power, OMEN Performance, Windows "Best performance")
1. `python experiments/run_iot.py speedcheck`   -> must show ok=True (station ~37 ms, edge ~16-20 ms)
2. `python experiments/run_traces.py`            (~3 min) -> results/traces.json
3. Short live validation: `FI_LINKS=wifi,constrained FI_NS=1,4 FI_REPS=1 FI_OUT=iot_live_validation.json python experiments/run_iot.py live`
4. Calibrate overhead c in `experiments/sim.py` on wifi edge-only N=1 (live p50 - sim p50 with c=0),
   check sim vs. live on the other validation configs, then `python experiments/sim.py <c>` -> results/sim.json
5. make_figures: fig/table for sim (p95, misses, edge utilisation vs N; nominal and edge x2) + validation table
   (replace fig_live/tab_live which used the invalid live sweeps); latency table from a speed-checked run.
6. Write `paper/ieee/sec6d_results.tex` (validated simulation; load model Eq. 6), update sec5 setup text
   (live sweep -> validation + simulation), fill [TBD] in sec8 and the abstract in main.tex; check every number.
7. Ask user: authors/affiliations, repo URL, AI-use statement; optional real-device `device_bench.py` run.
