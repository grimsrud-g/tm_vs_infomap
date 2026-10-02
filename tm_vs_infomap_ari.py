"""Matched adjusted Rand index (ARI) between TM and Infomap PFMs.

Native labels are converted to shared classes with network_matching.csv before
any comparison, so every ARI here is a matched ARI. TM Default Mode (1),
Temporal Pole (13), Medial Temporal Lobe (14), and Parietal Occipital (16)
join Infomap Default subnetworks (1-4) in the Default (DMN) shared class.
Vertices whose label maps
to EXCLUDE (TM 0/4/6/17, Infomap 0) in either map are dropped from that pair.
"""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from tm_vs_infomap import CONDITIONS, STRUCTURES, load_case


EXCLUDE = "EXCLUDE"
MATCHING_CSV = Path(__file__).with_name("network_matching.csv")
# load_case map name -> method column in network_matching.csv.
MATCHING_METHOD = {"TM": "TM", "Infomap": "Infomap", "Infomap unfiltered": "Infomap"}
MAP_SHORT_NAMES = {"TM": "TM", "Infomap": "Infomap filt+dil",
                   "Infomap unfiltered": "Infomap unfilt"}
# Sequential blue ramp (light -> dark) from the dataviz reference palette.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#1f2328", "#6b7280"


def comparisons():
    """(name, (map_a, condition_a), (map_b, condition_b)) for one subject."""
    pairs = []
    for infomap in ("Infomap", "Infomap unfiltered"):
        for condition in CONDITIONS:
            pairs.append((f"TM vs {MAP_SHORT_NAMES[infomap]} ({condition})",
                          ("TM", condition), (infomap, condition)))
    for name in ("TM", "Infomap", "Infomap unfiltered"):
        pairs.append((f"{MAP_SHORT_NAMES[name]}: rest vs task", (name, "rest"), (name, "task")))
    return pairs


def load_matching(path=MATCHING_CSV):
    matching = pd.read_csv(path, dtype={"label": int}, keep_default_na=False)
    if list(matching.columns) != ["method", "label", "name", "shared_class"]:
        raise ValueError(f"Unexpected columns in {path}: {list(matching.columns)}")
    if matching.duplicated(["method", "label"]).any():
        raise ValueError("Duplicate (method, label) rows in matching table")
    if (matching.shared_class.str.strip() == "").any():
        raise ValueError("Every matching row needs a shared_class (or EXCLUDE)")
    if set(matching.method) != set(MATCHING_METHOD.values()):
        raise ValueError(f"Matching methods must be {sorted(set(MATCHING_METHOD.values()))}")
    return matching


def matching_digest(path=MATCHING_CSV):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_classes(matching):
    """Shared classes in table order (first appearance), excluding EXCLUDE."""
    return list(dict.fromkeys(c for c in matching.shared_class if c != EXCLUDE))


def lookup(matching, method):
    rows = matching.loc[matching.method == method]
    return dict(zip(rows.label, rows.shared_class)), dict(zip(rows.label, rows.name))


def check_matching(matching, case):
    """Each label table must agree exactly (labels and names) with the CSV."""
    for map_name, data in case["maps"].items():
        _, names = lookup(matching, MATCHING_METHOD[map_name])
        table = {k: v[0] for k, v in data["table"].items()}
        if table != names:
            diff = sorted(set(table.items()) ^ set(names.items()))
            raise ValueError(f"sub-{case['subject']} {case['condition']} {map_name}: "
                             f"label table and matching CSV disagree: {diff}")


def to_shared(values, matching, method):
    """Map native integer labels to shared-class strings."""
    classes, _ = lookup(matching, method)
    unknown = set(np.unique(values)) - set(classes)
    if unknown:
        raise ValueError(f"{method}: labels missing from matching table: {sorted(unknown)}")
    return np.array([classes[v] for v in values], dtype=object)


def cortex(case, map_name, hemi="both"):
    values = case["maps"][map_name]["values"]
    if hemi == "both":
        return np.concatenate([values[h] for h in STRUCTURES])
    return values[hemi]


def matched_ari(a_shared, b_shared):
    """ARI over vertices labelled in both maps; returns (ari, n_kept)."""
    if len(a_shared) != len(b_shared):
        raise ValueError("Maps have different vertex counts")
    keep = (a_shared != EXCLUDE) & (b_shared != EXCLUDE)
    if not keep.any():
        raise ValueError("No vertices are labelled in both maps")
    return adjusted_rand_score(a_shared[keep], b_shared[keep]), int(keep.sum())


def load_cases(settings, subjects, conditions=CONDITIONS, matching=None):
    cases = {}
    for subject in subjects:
        for condition in conditions:
            case = load_case(settings, subject, condition)
            if matching is not None:
                check_matching(matching, case)
            cases[subject, condition] = case
    return cases


def ari_table(cases, matching, subjects):
    rows = []
    for subject in subjects:
        for name, (map_a, cond_a), (map_b, cond_b) in comparisons():
            case_a, case_b = cases[subject, cond_a], cases[subject, cond_b]
            for hemi in ("both", *STRUCTURES):
                a = to_shared(cortex(case_a, map_a, hemi), matching, MATCHING_METHOD[map_a])
                b = to_shared(cortex(case_b, map_b, hemi), matching, MATCHING_METHOD[map_b])
                ari, kept = matched_ari(a, b)
                rows.append(dict(subject=subject, comparison=name,
                                 map_a=map_a, condition_a=cond_a, map_b=map_b, condition_b=cond_b,
                                 hemisphere=hemi, n_cortex=len(a), n_kept=kept,
                                 pct_kept=100 * kept / len(a), ari=ari))
    return pd.DataFrame(rows)


def native_confusion(case, matching, infomap="Infomap"):
    """Vertex counts, native TM labels x native Infomap labels, both maps labelled.

    Rows and columns are ordered by shared class so matched blocks are contiguous.
    """
    tm, im = cortex(case, "TM"), cortex(case, infomap)
    keep = ((to_shared(tm, matching, "TM") != EXCLUDE)
            & (to_shared(im, matching, "Infomap") != EXCLUDE))
    order = {c: i for i, c in enumerate(shared_classes(matching))}

    def axis_labels(method, present):
        classes, names = lookup(matching, method)
        labels = sorted((l for l in present if classes[l] != EXCLUDE),
                        key=lambda l: (order[classes[l]], l))
        return labels, [names[l] for l in labels], [classes[l] for l in labels]

    tm_labels, tm_names, tm_classes = axis_labels("TM", set(tm[keep]))
    im_labels, im_names, im_classes = axis_labels("Infomap", set(im[keep]))
    # sklearn uses the same labels on both axes; select each map's native
    # labels afterward to retain the rectangular TM x Infomap table.
    labels = sorted(set(tm_labels) | set(im_labels))
    matrix = (confusion_matrix(tm[keep], im[keep], labels=labels) if labels
              else np.zeros((0, 0), dtype=int))
    counts = pd.DataFrame(matrix, index=labels, columns=labels).loc[tm_labels, im_labels]
    counts.index = pd.MultiIndex.from_arrays([tm_classes, tm_labels, tm_names],
                                             names=["shared_class", "label", "name"])
    counts.columns = pd.MultiIndex.from_arrays([im_classes, im_labels, im_names],
                                               names=["shared_class", "label", "name"])
    return counts


def plot_confusion(counts, title):
    """Vertex-count heatmap; outlines mark the matched shared-class blocks."""
    values = counts.to_numpy()
    vmax = max(int(values.max()), 1)
    fig, ax = plt.subplots(figsize=(0.42 * counts.shape[1] + 4.5, 0.42 * counts.shape[0] + 2.5))
    image = ax.imshow(values, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:,d}", ha="center", va="center", fontsize=7,
                color="white" if values[i, j] > 0.55 * vmax else INK)
    rows = counts.index.get_level_values("shared_class")
    cols = counts.columns.get_level_values("shared_class")
    for cls in dict.fromkeys(rows):
        r, c = np.flatnonzero(rows == cls), np.flatnonzero(cols == cls)
        if len(c):
            ax.add_patch(Rectangle((c[0] - 0.5, r[0] - 0.5), len(c), len(r),
                                   fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(counts.shape[1]),
                  [f"{l} {n}" for _, l, n in counts.columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(counts.shape[0]), [f"{l} {n}" for _, l, n in counts.index], fontsize=8)
    ax.set_xlabel("Infomap network")
    ax.set_ylabel("TM network")
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Cortical vertex count", color=MUTED)
    bar.outline.set_visible(False)
    fig.text(0.01, 0.005, "Outlined blocks = shared classes from network_matching.csv; "
             "counts include vertices labeled in both maps", fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout()
    return fig


def plot_ari(table, subjects, hemisphere="both"):
    """Annotated heatmap: comparison x subject, matched ARI."""
    data = table.loc[table.hemisphere == hemisphere]
    names = [name for name, *_ in comparisons()]
    grid = data.pivot(index="comparison", columns="subject", values="ari").loc[names, list(subjects)]
    fig, ax = plt.subplots(figsize=(1.1 * len(subjects) + 4.5, 0.5 * len(names) + 1.6))
    image = ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                color="white" if value > 0.55 else INK)
    ax.set_xticks(range(len(subjects)), [f"sub-{s}" for s in subjects])
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Matched ARI (hemisphere: {hemisphere})", fontsize=10, loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("ARI", color=MUTED)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def provenance(settings, cases, matching_path=MATCHING_CSV):
    import importlib.metadata
    return dict(
        settings={**vars(settings), "project": str(settings.project)},
        matching=dict(path=str(matching_path), sha256=matching_digest(matching_path),
                      rows=load_matching(matching_path).to_dict(orient="records")),
        inputs={f"sub-{s}_{c}": {k: str(p) for k, p in case["paths"].items() if not k.startswith("surface")}
                for (s, c), case in cases.items()},
        versions={k: importlib.metadata.version(k)
                  for k in ("numpy", "nibabel", "pandas", "scikit-learn", "matplotlib")},
    )


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
