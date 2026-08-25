"""Parameterized join/filter workload templates for training-data generation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterator


@dataclass(frozen=True)
class WorkloadSpec:
    template: str
    benchmark: str
    relations: tuple[str, ...]
    sql: str
    params: dict[str, object]


JOIN_TEMPLATES: dict[str, str] = {
    "tpch_orders_customer": """
SELECT COUNT(*)
FROM orders o
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE o.o_totalprice > {min_price}
  AND c.c_mktsegment = '{segment}'
""".strip(),
    "tpch_lineitem_orders_customer": """
SELECT COUNT(*)
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE l.l_discount > {min_discount}
  AND l.l_quantity < {max_quantity}
  AND c.c_mktsegment = '{segment}'
""".strip(),
    "tpch_lineitem_part_supplier": """
SELECT COUNT(*)
FROM lineitem l
JOIN partsupp ps ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
JOIN part p ON p.p_partkey = ps.ps_partkey
WHERE l.l_shipdate >= DATE '{shipdate}'
  AND p.p_type LIKE '{part_type}%'
""".strip(),
    "tpch_orders_lineitem_filter": """
SELECT COUNT(*)
FROM orders o
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE o.o_orderdate >= DATE '{orderdate}'
  AND l.l_extendedprice > {min_extendedprice}
""".strip(),
}


def iter_tpch_parameter_grid(max_variants_per_template: int = 250) -> Iterator[WorkloadSpec]:
    segments = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
    discounts = [0.02, 0.04, 0.06, 0.08]
    quantities = [10, 20, 30, 40]
    prices = [500, 1000, 2000, 5000]
    shipdates = ["1995-01-01", "1996-01-01", "1997-01-01"]
    part_types = ["STANDARD", "PROMO", "ECONOMY"]
    orderdates = ["1995-01-01", "1996-06-01", "1997-01-01"]
    extended_prices = [500, 1000, 2500]

    grids: dict[str, list[dict[str, object]]] = {
        "tpch_orders_customer": [
            {"min_price": p, "segment": s}
            for p, s in itertools.product(prices, segments)
        ],
        "tpch_lineitem_orders_customer": [
            {"min_discount": d, "max_quantity": q, "segment": s}
            for d, q, s in itertools.product(discounts, quantities, segments)
        ],
        "tpch_lineitem_part_supplier": [
            {"shipdate": sd, "part_type": pt}
            for sd, pt in itertools.product(shipdates, part_types)
        ],
        "tpch_orders_lineitem_filter": [
            {"orderdate": od, "min_extendedprice": ep}
            for od, ep in itertools.product(orderdates, extended_prices)
        ],
    }

    relations_map = {
        "tpch_orders_customer": ("orders", "customer"),
        "tpch_lineitem_orders_customer": ("lineitem", "orders", "customer"),
        "tpch_lineitem_part_supplier": ("lineitem", "partsupp", "part"),
        "tpch_orders_lineitem_filter": ("orders", "lineitem"),
    }

    emitted = 0
    for template, param_list in grids.items():
        for params in param_list[:max_variants_per_template]:
            sql = JOIN_TEMPLATES[template].format(**params)
            yield WorkloadSpec(
                template=template,
                benchmark="tpch",
                relations=relations_map[template],
                sql=sql,
                params=params,
            )
            emitted += 1
            if emitted >= max_variants_per_template * len(grids):
                return
