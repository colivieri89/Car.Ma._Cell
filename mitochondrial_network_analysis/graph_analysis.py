"""Skeleton graph construction and branch extraction (8-connectivity)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

Pixel = Tuple[int, int]

# 8-connectivity neighbour offsets and Euclidean edge weights
_NEIGH_OFFSETS: Tuple[Tuple[int, int, float], ...] = (
    (-1, -1, math.sqrt(2.0)),
    (-1,  0, 1.0),
    (-1,  1, math.sqrt(2.0)),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    ( 1, -1, math.sqrt(2.0)),
    ( 1,  0, 1.0),
    ( 1,  1, math.sqrt(2.0)),
)


@dataclass
class Branch:
    """A single skeleton branch / filament between two special nodes."""

    branch_id: int
    component_id: int
    start_node: Pixel
    end_node: Pixel
    start_node_type: str
    end_node_type: str
    pixels: List[Pixel]
    length_px: float

    @property
    def n_pixels(self) -> int:
        return len(self.pixels)


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """Build an 8-connected pixel graph of a binary skeleton.

    Nodes are ``(row, col)`` tuples; edges have a ``weight`` attribute equal
    to the Euclidean distance between adjacent pixels (1 or sqrt(2)).
    """
    graph = nx.Graph()
    coords = np.argwhere(skeleton)
    h, w = skeleton.shape
    pixel_set = {(int(r), int(c)) for r, c in coords}

    for r, c in pixel_set:
        graph.add_node((r, c))
        for dr, dc, weight in _NEIGH_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pixel_set:
                graph.add_edge((r, c), (nr, nc), weight=weight)

    logger.info(
        "Skeleton graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def classify_skeleton_nodes(graph: nx.Graph) -> Dict[str, List[Pixel]]:
    """Classify nodes by topological role.

    Returns
    -------
    dict
        Keys: ``"endpoints"`` (deg=1), ``"junctions"`` (deg>=3),
        ``"slab"`` (deg=2), ``"isolated"`` (deg=0).
    """
    endpoints, junctions, slab, isolated = [], [], [], []
    for node, deg in graph.degree():
        if deg == 0:
            isolated.append(node)
        elif deg == 1:
            endpoints.append(node)
        elif deg == 2:
            slab.append(node)
        else:
            junctions.append(node)
    return {
        "endpoints": endpoints,
        "junctions": junctions,
        "slab": slab,
        "isolated": isolated,
    }


def _node_type(degree: int) -> str:
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree == 2:
        return "slab"
    return "junction"


def component_is_isolated(
    nodes: Sequence[Pixel], degree: Dict[Pixel, int]
) -> bool:
    """A component is an *isolated filament* if it has no junction (deg >= 3)."""
    return not any(degree[n] >= 3 for n in nodes)


def extract_branches(graph: nx.Graph) -> Tuple[List[Branch], List[List[Pixel]]]:
    """Extract branches and connected components from the skeleton graph.

    Special nodes are nodes with degree != 2 (endpoints, junctions or
    isolated pixels). Slab nodes (degree 2) are intermediate pixels.
    Pure cycles made only of slab nodes are also captured as a single
    closed branch.

    Returns
    -------
    branches : list of Branch
    components_sorted : list of node lists
        Connected components sorted by size (descending). Component IDs in
        ``Branch.component_id`` correspond to the index in this list.
    """
    raw_components = list(nx.connected_components(graph))
    components_sorted: List[List[Pixel]] = [
        sorted(c) for c in sorted(raw_components, key=lambda c: -len(c))
    ]
    component_id_of: Dict[Pixel, int] = {}
    for cid, nodes in enumerate(components_sorted):
        for n in nodes:
            component_id_of[n] = cid

    degree: Dict[Pixel, int] = dict(graph.degree())
    special_nodes = {n for n, d in degree.items() if d != 2}
    branches: List[Branch] = []
    visited_edges: set = set()
    branch_id = 0

    def _walk(start: Pixel, first_neighbor: Pixel) -> Tuple[List[Pixel], float]:
        path = [start, first_neighbor]
        length = float(graph[start][first_neighbor]["weight"])
        prev, curr = start, first_neighbor
        while degree[curr] == 2 and curr not in special_nodes:
            nbrs = [n for n in graph.neighbors(curr) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            length += float(graph[curr][nxt]["weight"])
            path.append(nxt)
            prev, curr = curr, nxt
            if curr == start:
                break
        return path, length

    # 1) Walks from every special node along every incident edge.
    for node in special_nodes:
        for nbr in graph.neighbors(node):
            edge = tuple(sorted((node, nbr)))
            if edge in visited_edges:
                continue
            path, length = _walk(node, nbr)
            for a, b in zip(path[:-1], path[1:]):
                visited_edges.add(tuple(sorted((a, b))))
            start_n, end_n = path[0], path[-1]
            branches.append(
                Branch(
                    branch_id=branch_id,
                    component_id=component_id_of[start_n],
                    start_node=start_n,
                    end_node=end_n,
                    start_node_type=_node_type(degree[start_n]),
                    end_node_type=_node_type(degree[end_n]),
                    pixels=path,
                    length_px=length,
                )
            )
            branch_id += 1

    # 2) Pure cycles consisting only of slab nodes (no special node).
    for nodes in components_sorted:
        if not nodes or not all(degree[n] == 2 for n in nodes):
            continue
        sub_edges = list(graph.subgraph(nodes).edges())
        if any(tuple(sorted((a, b))) in visited_edges for a, b in sub_edges):
            continue
        start = nodes[0]
        nbr = next(iter(graph.neighbors(start)))
        path, length = _walk(start, nbr)
        for a, b in zip(path[:-1], path[1:]):
            visited_edges.add(tuple(sorted((a, b))))
        branches.append(
            Branch(
                branch_id=branch_id,
                component_id=component_id_of[start],
                start_node=start,
                end_node=start,
                start_node_type="cycle",
                end_node_type="cycle",
                pixels=path,
                length_px=length,
            )
        )
        branch_id += 1

    # 3) Isolated single-pixel components.
    for nodes in components_sorted:
        if len(nodes) == 1 and degree[nodes[0]] == 0:
            n = nodes[0]
            branches.append(
                Branch(
                    branch_id=branch_id,
                    component_id=component_id_of[n],
                    start_node=n,
                    end_node=n,
                    start_node_type="isolated",
                    end_node_type="isolated",
                    pixels=[n],
                    length_px=0.0,
                )
            )
            branch_id += 1

    logger.info(
        "Extracted %d branches across %d components.",
        len(branches),
        len(components_sorted),
    )
    return branches, components_sorted
