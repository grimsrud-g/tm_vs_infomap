"""Synthetic checks for the matched-ARI helpers (run with pytest)."""

import numpy as np
import pandas as pd
import pytest

from tm_vs_infomap_ari import (
    EXCLUDE, check_matching, load_matching, matched_ari, native_confusion, to_shared,
)


MATCHING = pd.DataFrame([
    ("TM", 0, "none", EXCLUDE), ("TM", 1, "DMN", "Default"), ("TM", 2, "PO", "Default"),
    ("TM", 3, "SMd", "SM"), ("TM", 4, "SMl", "SM"), ("TM", 5, "TP", "Other"),
    ("Infomap", 0, "Unassigned", EXCLUDE), ("Infomap", 1, "Def_A", "Default"),
    ("Infomap", 2, "Def_B", "Default"), ("Infomap", 3, "Hand", "SM"),
], columns=["method", "label", "name", "shared_class"])


def make_case(tm_values, im_values, tm_names=None):
    tm_table = {l: (n, (0, 0, 0, 1)) for _, l, n in
                MATCHING.loc[MATCHING.method == "TM", ["label", "name"]].itertuples()}
    if tm_names:
        tm_table.update({l: (n, (0, 0, 0, 1)) for l, n in tm_names.items()})
    im_table = {l: (n, (0, 0, 0, 1)) for _, l, n in
                MATCHING.loc[MATCHING.method == "Infomap", ["label", "name"]].itertuples()}
    half = len(tm_values) // 2
    split = lambda v: {"L": np.asarray(v[:half]), "R": np.asarray(v[half:])}
    return dict(subject="x", condition="rest", maps={
        "TM": dict(values=split(tm_values), table=tm_table),
        "Infomap": dict(values=split(im_values), table=im_table)})


def test_matched_networks_give_ari_one():
    # TM splits Default (1, 2) and SM (3, 4) differently from Infomap, but the
    # crosswalk merges them, so the matched partitions are identical.
    tm = to_shared(np.array([1, 2, 1, 3, 4, 3]), MATCHING, "TM")
    im = to_shared(np.array([1, 2, 2, 3, 3, 3]), MATCHING, "Infomap")
    ari, kept = matched_ari(tm, im)
    assert ari == pytest.approx(1.0) and kept == 6


def test_excluded_vertices_are_dropped_from_both_maps():
    tm = to_shared(np.array([0, 1, 1, 3, 3]), MATCHING, "TM")
    im = to_shared(np.array([3, 1, 1, 3, 0]), MATCHING, "Infomap")
    ari, kept = matched_ari(tm, im)
    assert kept == 3 and ari == pytest.approx(1.0)


def test_other_class_counts_as_its_own_cluster():
    tm = to_shared(np.array([1, 1, 5, 5]), MATCHING, "TM")
    im = to_shared(np.array([1, 1, 1, 1]), MATCHING, "Infomap")
    ari, _ = matched_ari(tm, im)
    assert ari < 1


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="missing from matching table"):
        to_shared(np.array([1, 99]), MATCHING, "TM")


def test_label_table_mismatch_raises():
    case = make_case([1, 1], [1, 1], tm_names={1: "Renamed"})
    with pytest.raises(ValueError, match="disagree"):
        check_matching(MATCHING, case)


def test_check_matching_accepts_exact_tables():
    check_matching(MATCHING, make_case([1, 3], [1, 3]))


def test_confusion_orders_by_shared_class_and_drops_excluded():
    case = make_case([2, 1, 3, 0, 5, 4], [1, 2, 3, 3, 0, 3])
    counts = native_confusion(case, MATCHING)
    assert list(counts.index.get_level_values("label")) == [1, 2, 3, 4]
    assert counts.to_numpy().sum() == 4


def test_csv_matching_table_is_valid():
    matching = load_matching()
    assert set(matching.loc[matching.shared_class == EXCLUDE, "label"]) == {0, 4, 6, 17}
    infomap = matching.loc[matching.method == "Infomap"].set_index("label").shared_class
    assert infomap[11] == "Somatomotor" and infomap[10] == "Dorsal_Attention"


def test_tm_temporal_pole_and_mtl_join_default():
    matching = load_matching()
    assert to_shared(np.array([1, 13, 14, 16]), matching, "TM").tolist() == ["Default"] * 4
    tm = to_shared(np.array([1, 13, 14, 16, 2, 2]), matching, "TM")
    im = to_shared(np.array([1, 2, 3, 4, 5, 5]), matching, "Infomap")
    ari, kept = matched_ari(tm, im)
    assert ari == pytest.approx(1.0) and kept == 6
