"""Cortex-only visual comparison of TM and Infomap; no network crosswalk."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, LabelAxis, ScalarAxis
import numpy as np
import pandas as pd
from nilearn import plotting, surface


SUBJECTS = ("s03", "s10", "s19", "s29", "s43")
CONDITIONS = ("rest", "task")
STRUCTURES = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT", "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}
ROW_SHORT_NAMES = {"Infomap": "filtered + dilated", "Infomap unfiltered": "unfiltered"}
TM_SUFFIX = "ReproTM_template-ABCC2026-a3-9to16_refine-SCAN_minsize-30"


@dataclass(frozen=True)
class Settings:
    project: Path
    run_label: str = "discovery-100trials-seed42-blas"
    post_label: str = "area50mm2-dil50mm"
    tm_unlabelled: str = "mask"
    legend_stats: str = "percent"
    dpi: int = 150
    include_unfiltered: bool = False

    def __post_init__(self):
        object.__setattr__(self, "project", Path(self.project))
        if self.tm_unlabelled not in ("mask", "grey"):
            raise ValueError("tm_unlabelled must be mask or grey")
        if self.legend_stats not in ("percent", "count", "none"):
            raise ValueError("legend_stats must be percent, count, or none")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")


def default_project():
    return Path(os.environ["OAK"]) / "users/grimsrud/projects/pfm_compare"


def case_paths(settings, subject, condition):
    subject = subject.removeprefix("sub-")
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    prefix = f"sub-{subject}"
    tm_dir = (settings.project / "analysis/temp_match_results/april_2026_fmriprep_xcpd"
              / prefix / f"task-{condition}" / prefix / "ses-concatenated/func")
    tm = tm_dir / f"{prefix}_ses-concatenated_task-{condition}_{TM_SUFFIX}.dscalar.nii"
    im_dir = (settings.project / "analysis/pypfm_xcpd" / prefix / condition
              / settings.run_label / "postprocess" / settings.post_label)
    anat = settings.project / "derivatives/xcpd_26.0.2/april_2026_proc" / prefix / "anat"
    return {
        "tm": tm,
        "tm_labels": tm.with_name(tm.name.replace(".dscalar.nii", ".dlabel.nii")),
        **({"infomap_unfiltered": im_dir / f"{prefix}_task-{condition}_desc-infomap_networks.dlabel.nii"}
           if settings.include_unfiltered else {}),
        "infomap": im_dir / f"{prefix}_task-{condition}_desc-spatialfiltered+dilated50mm_networks.dlabel.nii",
        **{f"surface_{h}": anat / f"{prefix}_hemi-{h}_space-fsLR_den-32k_desc-hcp_inflated.surf.gii"
           for h in STRUCTURES},
    }


def input_inventory(settings, subjects=SUBJECTS, conditions=CONDITIONS):
    """Check every dependency, including companion label tables and surfaces."""
    return pd.DataFrame([
        dict(subject=s, condition=c, role=role, path=str(path), exists=path.is_file())
        for s in subjects for c in conditions
        for role, path in case_paths(settings, s, c).items()
    ])


def require_inputs(inventory):
    missing = inventory.loc[~inventory.exists]
    if len(missing):
        raise FileNotFoundError("Missing inputs:\n" + missing.to_string(index=False))


def _load_image(path, axis_type):
    img = nib.load(path)
    if not isinstance(img, nib.Cifti2Image) or len(img.shape) != 2 or img.shape[0] != 1:
        raise ValueError(f"Expected a single-map CIFTI: {path}")
    if not isinstance(img.header.get_axis(0), axis_type):
        raise ValueError(f"Unexpected map axis in {path}")
    axis = img.header.get_axis(1)
    if not isinstance(axis, BrainModelAxis):
        raise ValueError(f"Missing BrainModelAxis: {path}")
    return img, axis


def cortical_layout(axis):
    """Use explicit brain-model rows; never assume cortex is a prefix."""
    layout = {}
    for hemi, structure in STRUCTURES.items():
        rows = np.flatnonzero(axis.name == structure)
        if not len(rows):
            raise ValueError(f"Missing {structure}")
        vertices = axis.vertex[rows]
        nvertices = axis.nvertices[structure]
        if nvertices != 32492:
            raise ValueError(f"Expected fsLR 32k surface (32492 vertices), got {nvertices}")
        if np.any(vertices < 0) or np.any(vertices >= nvertices) or len(np.unique(vertices)) != len(vertices):
            raise ValueError(f"Invalid or repeated surface vertices in {structure}")
        layout[hemi] = dict(rows=rows, vertices=vertices, nvertices=nvertices)
    return layout


def assert_same_cortex(first, second):
    """Compare ordered cortical brain models, ignoring unrelated subcortex."""
    a, b = cortical_layout(first), cortical_layout(second)
    mask_a = np.isin(first.name, list(STRUCTURES.values()))
    mask_b = np.isin(second.name, list(STRUCTURES.values()))
    if not (np.array_equal(first.name[mask_a], second.name[mask_b])
            and np.array_equal(first.vertex[mask_a], second.vertex[mask_b])):
        raise ValueError("Cortical BrainModelAxis vertex order differs between inputs")
    for h in STRUCTURES:
        if a[h]["nvertices"] != b[h]["nvertices"]:
            raise ValueError(f"Surface size differs for {h}")


def _cortical_labels(img, axis):
    layout = cortical_layout(axis)
    values = np.asarray(img.dataobj)[0]
    result = {}
    for h, info in layout.items():
        raw = values[info["rows"]]
        if not np.all(np.isfinite(raw)) or not np.allclose(raw, np.rint(raw), rtol=0, atol=1e-6):
            raise ValueError("Cortical labels must be finite integers")
        if np.any(raw < 0):
            raise ValueError("Cortical labels must be nonnegative")
        result[h] = raw.astype(np.int64)
    return result, layout


def load_case(settings, subject, condition):
    paths = case_paths(settings, subject, condition)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    tm, tm_axis = _load_image(paths["tm"], ScalarAxis)
    labels, labels_axis = _load_image(paths["tm_labels"], LabelAxis)
    im, im_axis = _load_image(paths["infomap"], LabelAxis)
    assert_same_cortex(tm_axis, labels_axis)
    assert_same_cortex(tm_axis, im_axis)
    tm_values, layout = _cortical_labels(tm, tm_axis)
    companion_values, _ = _cortical_labels(labels, labels_axis)
    if any(not np.array_equal(tm_values[h], companion_values[h]) for h in STRUCTURES):
        raise ValueError("TM dscalar and companion dlabel cortical labels disagree")
    im_values, _ = _cortical_labels(im, im_axis)
    variants = [("TM", tm_values, labels), ("Infomap", im_values, im)]
    if settings.include_unfiltered:
        unfiltered, unfiltered_axis = _load_image(paths["infomap_unfiltered"], LabelAxis)
        assert_same_cortex(tm_axis, unfiltered_axis)
        unfiltered_values, _ = _cortical_labels(unfiltered, unfiltered_axis)
        variants.append(("Infomap unfiltered", unfiltered_values, unfiltered))
    maps = {}
    for name, values, label_img in variants:
        table = {int(k): (str(v[0]), tuple(v[1]))
                 for k, v in label_img.header.get_axis(0).label[0].items()}
        present = set(np.unique(np.concatenate(list(values.values()))))
        unknown = present - set(table) - {0}
        if unknown:
            raise ValueError(f"{name}: labels absent from label table: {sorted(unknown)}")
        for key, (_, color) in table.items():
            if len(color) != 4 or not np.all(np.isfinite(color)) or np.any(np.asarray(color) < 0) or np.any(np.asarray(color) > 1):
                raise ValueError(f"{name}: invalid RGBA for label {key}")
        hidden = {0}
        if name == "TM":
            if settings.tm_unlabelled == "mask":
                hidden.update((4, 6, 17))
            else:
                for key in (4, 6, 17):
                    if key in table:
                        table[key] = (table[key][0], (0.6, 0.6, 0.6, 1.0))
        maps[name] = dict(values=values, table=table, hidden=hidden)
    meshes = {h: surface.load_surf_mesh(str(paths[f"surface_{h}"])) for h in STRUCTURES}
    for h, mesh in meshes.items():
        if len(mesh.coordinates) != layout[h]["nvertices"]:
            raise ValueError(f"Surface vertex count disagrees with CIFTI: {h}")
    return dict(subject=subject.removeprefix("sub-"), condition=condition,
                paths=paths, layout=layout, maps=maps, meshes=meshes)


def coverage_table(case):
    """Descriptive counts, not overlap metrics; denominator includes hidden cortex."""
    rows = []
    for method, data in case["maps"].items():
        values = np.concatenate(list(data["values"].values()))
        ids, counts = np.unique(values, return_counts=True)
        for key, count in zip(ids, counts):
            rows.append(dict(method=method, label=int(key),
                             name=data["table"].get(key, ("Unassigned", None))[0],
                             vertices=int(count), percent=100 * count / len(values),
                             displayed=key not in data["hidden"]))
    return pd.DataFrame(rows)


def surface_map(case, method, hemi):
    """Dense categorical display indices, preserving native label colors."""
    data = case["maps"][method]
    keys = sorted(set(np.concatenate(list(data["values"].values()))) - data["hidden"])
    colors = [data["table"][k][1] for k in keys] or [(0, 0, 0, 0)]
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    info = case["layout"][hemi]
    out = np.full(info["nvertices"], np.nan)
    for display_index, key in enumerate(keys, start=1):
        out[info["vertices"][data["values"][hemi] == key]] = display_index
    return out, cmap, len(colors)


def _plot_legends(fig, grid, case, coverage, settings):
    """One legend for TM and one shared legend spanning the Infomap rows."""
    methods = list(case["maps"])
    groups = [[m for m in methods if m == "TM"], [m for m in methods if m != "TM"]]
    for group in filter(None, groups):
        tables = [case["maps"][m]["table"] for m in group]
        shared = set.intersection(*(set(t) for t in tables))
        if any(t[k] != tables[0][k] for t in tables for k in shared):
            raise ValueError(f"Label tables disagree across {group}; cannot share a legend")
        rows = [methods.index(m) for m in group]
        legend_ax = fig.add_subplot(grid[min(rows):max(rows) + 1, 4])
        legend_ax.axis("off")
        subset = coverage[coverage.method.isin(group) & coverage.displayed]
        handles = []
        for label in sorted(subset.label.unique()):
            stats = []
            for m in group:
                hit = subset[(subset.method == m) & (subset.label == label)]
                stats.append("–" if hit.empty else
                             f"{hit.percent.iloc[0]:.1f}%" if settings.legend_stats == "percent" else
                             f"n={hit.vertices.iloc[0]:,}")
            suffix = "" if settings.legend_stats == "none" else f" ({' / '.join(stats)})"
            table = next(t for t in tables if label in t)
            handles.append(Patch(facecolor=table[label][1], edgecolor="0.3",
                                 label=table[label][0] + suffix))
        title = f"{group[0]}: networks present"
        if len(group) > 1:
            title = "Infomap: networks present"
            if settings.legend_stats != "none":
                title += "\n(" + " / ".join(ROW_SHORT_NAMES.get(m, m) for m in group) + ")"
        if handles:
            legend_ax.legend(handles=handles, loc="center left", frameon=False,
                             fontsize=9, title=title)


def plot_case(case, settings):
    """Return matched surface views and a legend for each map variant."""
    nrows = len(case["maps"])
    fig = plt.figure(figsize=(22, 4.5 * nrows), facecolor="white")
    grid = fig.add_gridspec(nrows, 5, width_ratios=[1, 1, 1, 1, 1.3],
                            left=0.075, right=0.995, bottom=0.09, top=0.90,
                            wspace=0, hspace=0.12)
    coverage = coverage_table(case)
    views = [("L", "lateral"), ("L", "medial"), ("R", "lateral"), ("R", "medial")]
    for row, method in enumerate(case["maps"]):
        for col, (h, view) in enumerate(views):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            values, cmap, ncolors = surface_map(case, method, h)
            plotting.plot_surf_roi(
                case["meshes"][h], values,
                hemi="left" if h == "L" else "right", view=view,
                cmap=cmap, vmin=0.5, vmax=ncolors + 0.5,
                avg_method="median", bg_on_data=False,
                alpha=1, colorbar=False, axes=ax, figure=fig)
            if row == 0:
                ax.set_title(f"{'Left' if h == 'L' else 'Right'} hemisphere, {view}", fontsize=11)
    _plot_legends(fig, grid, case, coverage, settings)
    row_labels = {
        "TM": "TM\nABCC2026\nrefine-SCAN\nminsize-30",
        "Infomap": "Infomap\npypfm v2\nfiltered +\ndilated 50 mm",
        "Infomap unfiltered": "Infomap\npypfm v2\nno filter\nno dilation",
    }
    for row, method in enumerate(case["maps"]):
        bounds = grid[row, 0].get_position(fig)
        fig.text(0.012, (bounds.y0 + bounds.y1) / 2, row_labels[method],
                 va="center", fontsize=11)
    fig.suptitle(f"sub-{case['subject']} | {case['condition']} | TM vs Infomap", fontsize=17)
    fig.text(0.075, 0.03,
             "Independent network taxonomies and colours: similar colours do not imply matching networks.\n"
             "Percentages use all cortical vertices in the CIFTI (excluding the medial wall); blank regions show the surface background.",
             fontsize=10)
    return fig


def save_case(case, settings, output_dir, *, overwrite=False):
    """Save PNG and input/settings provenance, returning (figure, PNG path)."""
    output_dir = Path(output_dir)
    stem = f"sub-{case['subject']}_task-{case['condition']}_TM-vs-Infomap"
    png, provenance = output_dir / f"{stem}.png", output_dir / f"{stem}.json"
    if not overwrite and (png.exists() or provenance.exists()):
        raise FileExistsError(f"Output already exists: {stem}; choose a new folder or overwrite=True")
    fig = plot_case(case, settings)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=settings.dpi, facecolor="white")
        import importlib.metadata
        record = dict(subject=case["subject"], condition=case["condition"],
                      settings={**vars(settings), "project": str(settings.project)},
                      inputs={k: dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                              for k, p in case["paths"].items()},
                      versions={k: importlib.metadata.version(k) for k in ("numpy", "nibabel", "nilearn", "matplotlib")},
                      cortical_counts=coverage_table(case).to_dict(orient="records"))
        provenance.write_text(json.dumps(record, indent=2) + "\n")
    except Exception:
        plt.close(fig)
        raise
    return fig, png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-unfiltered", action="store_true",
                        help="Add regular (not spatially filtered or dilated) Infomap as a third row")
    parser.add_argument("--tm-unlabelled", choices=("mask", "grey"), default="mask")
    parser.add_argument("--legend-stats", choices=("percent", "count", "none"), default="percent")
    parser.add_argument("--run-label", default=Settings.run_label)
    parser.add_argument("--post-label", default=Settings.post_label)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    settings = Settings(args.project or default_project(), args.run_label, args.post_label,
                        args.tm_unlabelled, args.legend_stats, args.dpi, args.include_unfiltered)
    inventory = input_inventory(settings, args.subjects, args.conditions)
    print(inventory.to_string(index=False))
    require_inputs(inventory)
    if not args.check_only and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Render inside a Slurm allocation (e.g. sh_dev)")
    if not args.check_only and args.output_dir is None:
        parser.error("Rendering requires --output-dir")
    for subject in args.subjects:
        for condition in args.conditions:
            case = load_case(settings, subject, condition)
            print(f"Validated sub-{subject} {condition}", flush=True)
            if not args.check_only:
                fig, png = save_case(case, settings, args.output_dir, overwrite=args.overwrite)
                plt.close(fig)
                print(png, flush=True)


if __name__ == "__main__":
    main()
