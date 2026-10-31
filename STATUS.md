# FabricInspect-IoT – project status (2026-10-31)

Target journal: **IEEE Internet of Things Journal**. Manuscript: `paper/ieee/` (upload the whole folder to
Overleaf, main file `main.tex`, IEEEtran class). Repository (public):
https://github.com/nasin0764-bit/FabricInspect-IoT

## Manuscript: complete draft
All sections, abstract (237 words), 6 figures, 10 tables, 37 Crossref-verified references, author list,
author contributions, AI-use acknowledgment. Every number comes from `results/*.json` via the scripts.

## Before submission (authors)
1. Compile on Overleaf and read the PDF end to end (no local LaTeX was available to compile it here).
2. Confirm/edit the AI-use statement and add funding/acknowledgments (sec8_conclusion.tex).
3. Optional but valuable for IoT-J reviewers: run `experiments/device_bench.py` on a real station device
   (Raspberry Pi / Android-Termux) and add the numbers to Section VI-C.
4. Cover letter + suggested reviewers (IEEE IoT-J submission system).

## Notes
- Timing runs only on AC power with the original charger, Windows "Best performance"; the scripts speed-check
  before/after and discard throttled runs (`*_INVALID*.json` are kept locally, not in git).
- Known weak spots, reported in the paper: grid (LOO threshold too conservative), station model on carpet/leather.
