# Proposed: visual comparison of Template Matching vs Infomap network maps

Status: brainstorm / plan only (2026-09-30). No code yet.

## Goal

Visually inspect, for each subject, how the Template Matching (TM) network map
compares with the Infomap network map for the same data, separately for
**rest** and **task**. No comparison metrics for now: the output is figures to
look at.

## What to compare

| | Template Matching | Infomap (pypfm) |
|---|---|---|
| Map | `..._ReproTM_template-ABCC2026-a3-9to16_refine-SCAN_minsize-30.dscalar.nii` | `..._desc-spatialfiltered+dilated50mm_networks.dlabel.nii` |
| Location | `analysis/temp_match_results/april_2026_fmriprep_xcpd/sub-<sub>/task-<rest\|task>/sub-<sub>/ses-concatenated/func/` | `analysis/pypfm_xcpd/sub-<sub>/<rest\|task>/discovery-100trials-seed42-blas/postprocess/area50mm2-dil50mm/` |
| Labels and colours | From the matching `..._refine-SCAN_minsize-30.dlabel.nii` label table | From the dlabel's own label table (colours from `data/priors.mat`) |
| Network set | ABCC template: 18 label slots, 15 named networks (Default_Mode, Visual, Frontoparietal, Dorsal/Ventral Attention, Salience, Action-Mode, Sensorimotor Dorsal/Lateral, Auditory, Temporal Pole, Medial Temporal Lobe, Parietal Memory, Parietal Occipital, Somato-Cognitive-Action) plus `unlabelled4/6/17` | Lynch priors: 20 networks (4 Default subnetworks, 4 Visual, Frontoparietal, DorsalAttention, Premotor/DorsalAttentionII, Language, Salience, CinguloOpercular/Action-mode, MedialParietal, 3 Somatomotor, Auditory, SomatoCognitiveAction) |
| Coverage | Every cortical vertex labelled (no zeros in sub-s03) | 0 = unassigned at ineligible vertices (~740–1,140 per subject) |

Subjects: s03, s10, s19, s29, s43. Conditions: rest, task. Both maps are on the
fsLR 32k grid (91,282 grayordinates; 59,412 cortical vertices). Cortex only.

## Figure design

**One PNG per subject per condition** (10 PNGs):
`sub-<sub>_task-<rest|task>_TM-vs-Infomap.png`.

```
                 Left lateral   Left medial   Right lateral   Right medial  | Legend
Row 1  TM        [surface]      [surface]     [surface]       [surface]     | TM networks present
Row 2  Infomap   [surface]      [surface]     [surface]       [surface]     | Infomap networks present
```

- Same rendering as notebook 05: nilearn `plot_surf_roi` on the subject's
  inflated surface
  (`derivatives/xcpd_26.0.2/april_2026_proc/sub-<sub>/anat/sub-<sub>_hemi-{L,R}_space-fsLR_den-32k_desc-hcp_inflated.surf.gii`),
  four views, the same camera for both rows so panels line up vertically.
- Column titles: "Left hemisphere, lateral", etc. (as now in notebook 05).
- Row labels on the left saying what each map is, e.g. "TM (ABCC2026, refine-SCAN, minsize-30)"
  and "Infomap (pypfm v2, filtered + dilated 50 mm)".
- Figure title: subject and condition.
- **Legend per row**, drawn in its own axis to the right of that row, using that
  method's own names and colours. List only networks present in the map.
  Possibly add vertex counts or % of cortex after each name to show which
  networks dominate.
- Also display each figure inline in the notebook.

Optional later: an all-subjects PNG per condition (10 rows: TM/Infomap for each
subject), for a one-page overview.

## Things to handle

1. **Different numbers of networks and different taxonomies.** TM has about 15
   named networks, Infomap 20, and the names don't match one to one (e.g.
   Infomap splits Default into 4 subnetworks and Visual into 4; TM has
   Temporal Pole, Medial Temporal Lobe and Parietal Memory, which the Lynch
   priors don't). Each row therefore gets its own legend and its own colours;
   no attempt to match networks across methods for now.
2. **Colours can mislead.** The two colour schemes are independent, so a similar
   colour in both rows does not mean the same network. Say so in the figure
   caption or notebook text. (Future option: a name crosswalk table, e.g.
   Infomap Default_* → TM Default_Mode, so matched networks share a colour.)
3. **Background and unassigned vertices.**
   - Infomap 0 (ineligible or unassigned): leave blank (transparent).
   - TM `unlabelled4/6/17`: decide whether to mask them like
     `plot_pfm_overlap/pfm_overlap.py` does (it treats 0, 4, 6, 17 as
     background) or show them in grey with a legend entry. They're not
     negligible: label 4 covers 555 vertices in sub-s03 rest.
   - TM Salience is black (0, 0, 0) in the ABCC table. Check that it reads
     against the surface shading.
4. **Same vertex order.** Before plotting, check that the TM and Infomap files
   have the same cortical BrainModelAxis (vertex indices), then map cortical
   rows to the 32k surface with `surface_vertices`, the same as notebook 05.
5. **Surface shading.** Optional: add a sulcal-depth or curvature background
   (`bg_map`) so anatomy is visible under blank vertices. Would need an fsLR
   32k sulc file; the XCP-D `anat` folder only has parcel-level morph TSVs.
6. **Run time.** About 8 surface renders per PNG, 10 PNGs: a few minutes on a
   compute node (`sh_dev`). Not for a login node.

## Existing code to reuse

- `prep_pypfm_inputs/05_check_postprocess.ipynb`: `to_surface` and
  `plot_on_surface` (surface rendering, hemisphere titles, prior colours).
- `code/plot_pfm_overlap/pfm_overlap.py`: `load_label_table`,
  `make_network_cmap`, `make_background_mask`, `extract_cortical_values`,
  `plot_surface_grid`, already written for TM dscalar + dlabel pairs.
  Either import it by path or copy the needed functions into a small helper
  module next to the new notebook.

## Proposed notebook

`code/tm_vs_infomap/tm_vs_infomap_visual.ipynb`, kernel "pypfm (group env)"
(has nilearn 0.14; `pfm_overlap.py` was developed in the network-fmri venv,
so check it imports there).

1. Settings: subjects, conditions, TM file pattern, Infomap run/post labels and
   variant, output folder, masking choice for TM unlabelled labels.
2. Path builder and a check that all 20 input files exist (report missing ones,
   e.g. before the post-processing batch finishes).
3. Loaders: cortical values plus the label table (name, RGBA) for each map.
4. Plot function: one subject/condition → 2 × 4 panels + 2 legends → PNG.
5. Loop over subjects and conditions; save PNGs; show them inline.

Output folder suggestion:
`analysis/tm_vs_infomap/<date or label>/sub-<sub>_task-<cond>_TM-vs-Infomap.png`.

## Out of scope for now

Dice or overlap metrics, network matching across methods, subcortex, other
Infomap variants (raw, filtered-only) and other TM variants (without
refine-SCAN / minsize). Easy to add later as extra rows.

## Open questions

- One PNG per subject and condition (as above), or one per condition with all
  subjects stacked? Or both?
- TM `unlabelled` labels: mask or show in grey?
- Infomap variant: dilated only, or add the raw `infomap` row too (the
  pfm-mefmri-style labeling)?
- Add vertex counts or percentages to the legends?
- Output location and file naming OK?

## Implementation update — 2026-09-30

The workflow is implemented in `tm_vs_infomap.py`, with single-case and all-subjects notebooks. See [the completed analysis README](../README.md). The proposal above is preserved as the original plan.

Resolved settings: one PNG per subject/condition; TM labels 0/4/6/17 masked (user preference); filtered + dilated 50 mm Infomap only; independent legends with percentages of all represented cortical vertices; output folder `analysis/tm_vs_infomap/30Sept2026_discovery`. Input checks also include TM companion dlabels and inflated surfaces. Optional montage/crosswalk/raw-map additions remain deferred.
