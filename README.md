Author: Gracie Grimsrud  
Date Created: 09/30/2026  
Overall Project: PFM Compare  
Analysis: Infomap vs Template Matching Visual Comparison

## Goal

The scripts in `code/tm_vs_infomap/`:

1. Load Template Matching (TM) and Infomap PFMs for each discovery subject, separately for rest and task.
2. Plot the three maps together for visual comparison: TM above filtered + dilated Infomap and filtered-only Infomap, with left lateral, left medial, right lateral, and right medial views on the same subject's inflated fsLR 32k surfaces.
3. Save one PNG per subject/condition and display it inline in the notebooks. Each of the three rows has its own network legend.

Subjects: **s03, s10, s19, s29, s43**. Conditions: **rest, task** (10 figures).
Cortex only; no Dice/overlap metrics or network matching across methods.

## Maps Used

Template Matching : `*_ReproTM_template-ABCC2026-a3-9to16_refine-SCAN_minsize-30.dscalar.nii`
Infomap, filtered + dilated : `*_desc-spatialfiltered+dilated50mm_networks.dlabel.nii`

The third row uses `*_desc-spatialfiltered_networks.dlabel.nii` (spatial filtering without dilation). Enable it with `include_filtered=True` or `--include-filtered`; the notebooks enable it by default.

## Input Dir

All paths below are relative to `PROJECT = /oak/stanford/groups/russpold/users/grimsrud/projects/pfm_compare`:

- TM: `analysis/temp_match_results/april_2026_fmriprep_xcpd/sub-<subject>/task-<rest|task>/sub-<subject>/ses-concatenated/func/`
- Infomap: `analysis/pypfm_xcpd/sub-<subject>/<rest|task>/discovery-100trials-seed42-blas/postprocess/area50mm2-dil50mm/`
- Surfaces: `derivatives/xcpd_26.0.2/april_2026_proc/sub-<subject>/anat/sub-<subject>_hemi-<L|R>_space-fsLR_den-32k_desc-hcp_inflated.surf.gii`

The default batch checks 60 file references: 30 network maps, 10 TM companion dlabels, and 20 surface references (10 unique surfaces). It validates ordered cortical vertex indices, 32k surface dimensions, integral labels, label tables, and agreement between the TM scalar and its companion dlabel. Cortical rows are selected explicitly; the loader does not assume cortex occupies the first rows. Missing or incompatible inputs raise an error.

## Output Dir

Default notebook setting:
`/oak/stanford/groups/russpold/users/grimsrud/projects/pfm_compare/analysis/tm_vs_infomap/30Sept2026_discovery_with_filtered/`

- `sub-<subject>_task-<condition>_TM-vs-Infomap.png`: 3 × 4 surface panels and three legends.
- Matching `.json`: input paths, file sizes/modification times, settings, package versions, and per-method cortical counts.

Inputs are read-only. Existing outputs require `OVERWRITE=True` in a notebook or `--overwrite` on the command line; alternatively choose a new output directory.

## Environment

Use the existing **pypfm (group env)** Jupyter kernel (`pypfm`):
`$GROUP_HOME/grimsrud/envs/pypfm/bin/python`.

Installed versions checked on 09/30/2026: NumPy 2.4.3, nibabel 5.4.2, nilearn 0.14.1, matplotlib 3.11.2, pandas 2.3.2, nbformat 5.11.1. The notebooks also use IPython. The helper does not import the pypfm package or load time series/FC matrices.

Render in a Sherlock compute allocation (`sh_dev` or an Open OnDemand compute session). Allow several minutes for all ten figures, with one CPU and a few GB of memory; no GPU is needed. Rendering is sequential, and notebooks close figures after display.

## Files

1. `tm_vs_infomap/tm_vs_infomap_visual.ipynb`: step-by-step single-case notebook (default sub-s03 rest): settings, dependency check, validated loading, descriptive counts, figure saving/display, and review notes.
2. `tm_vs_infomap/tm_vs_infomap_allsubs.ipynb`: all five subjects × rest/task, with dependency/content checks before rendering and inline display of every saved PNG.
3. `tm_vs_infomap/tm_vs_infomap.py`: shared paths, inventory, CIFTI validation, cortical mapping, legends, surface rendering, provenance, and command-line entry point.
4. `tm_vs_infomap/test_tm_vs_infomap.py`: synthetic checks for alignment errors, missing inputs, masking, and discrete colour mapping.
5. `tm_vs_infomap/proposed_TM_vs_infomap.md`: original analysis proposal, with an implementation note appended.

## Run

Open either notebook from `code/` or `code/tm_vs_infomap/`, select the pypfm kernel, edit the settings cell, then run the cells in order. Start with the single-case notebook to inspect the figure design.
