"""Tests for template-aware splits and q-error analysis."""

from learned_ce.splits import assign_template_splits, stable_query_id


def test_assign_template_splits_are_disjoint():
    templates = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    split_map = assign_template_splits(templates, train_ratio=0.6, val_ratio=0.2, seed=1)
    assert set(split_map.values()) <= {"train", "val", "test"}
    assert len(set(split_map.keys())) == len(templates)


def test_stable_query_id_is_deterministic():
    params = {"min_price": 500, "segment": "AUTOMOBILE"}
    assert stable_query_id("tpch_orders_customer", params, 0) == stable_query_id(
        "tpch_orders_customer", params, 0
    )
