"""Graph algorithm correctness, including the recursion-depth failure."""

import pytest

from algokit.graphs import (
    bfs,
    connected_components,
    dfs,
    dfs_recursive,
    dijkstra,
    has_cycle,
    kruskal_mst,
    shortest_path_unweighted,
    topological_sort,
)

# a -- b -- d
# |    |
# c ---+     e (isolated)
GRAPH = {"a": ["b", "c"], "b": ["a", "c", "d"], "c": ["a", "b"], "d": ["b"], "e": []}

WEIGHTED = {
    "a": [("b", 1.0), ("c", 4.0)],
    "b": [("c", 2.0), ("d", 6.0)],
    "c": [("d", 3.0)],
    "d": [],
}


class TestTraversal:
    def test_bfs_visits_by_level(self):
        assert bfs(GRAPH, "a") == ["a", "b", "c", "d"]

    def test_dfs_goes_deep_first(self):
        assert dfs(GRAPH, "a") == ["a", "b", "c", "d"]

    def test_iterative_and_recursive_dfs_agree(self):
        assert dfs(GRAPH, "a") == dfs_recursive(GRAPH, "a")

    def test_isolated_node(self):
        assert bfs(GRAPH, "e") == ["e"]

    def test_recursive_dfs_overflows_where_iterative_does_not(self):
        """The concrete cost of recursion: a deep path breaks it.

        This is why the iterative form is the default in this package.
        """
        n = 20_000
        path = {i: [i + 1] for i in range(n)}
        path[n] = []

        assert len(dfs(path, 0)) == n + 1
        with pytest.raises(RecursionError):
            dfs_recursive(path, 0)

    def test_connected_components(self):
        graph = {1: [2], 2: [1], 3: [4], 4: [3], 5: []}
        assert sorted(len(c) for c in connected_components(graph)) == [1, 2, 2]


class TestShortestPaths:
    def test_bfs_path(self):
        assert shortest_path_unweighted(GRAPH, "a", "d") == ["a", "b", "d"]

    def test_same_node(self):
        assert shortest_path_unweighted(GRAPH, "a", "a") == ["a"]

    def test_unreachable(self):
        assert shortest_path_unweighted(GRAPH, "a", "e") is None

    def test_dijkstra_distances(self):
        dist, _ = dijkstra(WEIGHTED, "a")
        assert dist == {"a": 0.0, "b": 1.0, "c": 3.0, "d": 6.0}

    def test_dijkstra_prefers_the_indirect_cheaper_route(self):
        """a->c direct is 4; a->b->c is 3. The shorter hop count is not the answer."""
        dist, parents = dijkstra(WEIGHTED, "a")
        assert dist["c"] == 3.0
        assert parents["c"] == "b"

    def test_dijkstra_rejects_negative_weights(self):
        """Better to fail loudly than return a silently wrong answer."""
        with pytest.raises(ValueError, match="negative weight"):
            dijkstra({"a": [("b", -1.0)], "b": []}, "a")

    def test_dijkstra_handles_unorderable_nodes(self):
        """Distance ties must not force a comparison between the nodes themselves."""
        graph = {(0, 0): [((1, 1), 1.0), ((2, 2), 1.0)], (1, 1): [], (2, 2): []}
        dist, _ = dijkstra(graph, (0, 0))
        assert dist[(1, 1)] == dist[(2, 2)] == 1.0


class TestOrderingAndCycles:
    def test_topological_order_respects_edges(self):
        dag = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        order = topological_sort(dag)
        position = {node: i for i, node in enumerate(order)}
        for node, targets in dag.items():
            for target in targets:
                assert position[node] < position[target]

    def test_topological_sort_detects_cycles(self):
        with pytest.raises(ValueError, match="cycle"):
            topological_sort({"a": ["b"], "b": ["c"], "c": ["a"]})

    def test_directed_cycle_detection(self):
        assert has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]})
        assert not has_cycle({"a": ["b"], "b": ["c"], "c": []})

    def test_diamond_is_not_a_cycle(self):
        """The case a two-colour DFS gets wrong: a re-visit is not a cycle."""
        assert not has_cycle({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []})

    def test_undirected_edge_is_not_a_cycle(self):
        assert not has_cycle({"a": ["b"], "b": ["a"]}, directed=False)
        triangle = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        assert has_cycle(triangle, directed=False)


class TestKruskal:
    def test_mst_weight_and_edge_count(self):
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 4.0)]
        chosen, total = kruskal_mst(4, edges)
        assert len(chosen) == 3          # a spanning tree has V-1 edges
        assert total == 7.0
        assert (0, 2, 3.0) not in chosen  # the cycle-closing edge is skipped

    def test_rejects_the_heavier_parallel_edge(self):
        chosen, total = kruskal_mst(2, [(0, 1, 5.0), (0, 1, 1.0)])
        assert chosen == [(0, 1, 1.0)] and total == 1.0

    def test_disconnected_graph_yields_a_forest(self):
        chosen, _ = kruskal_mst(4, [(0, 1, 1.0), (2, 3, 1.0)])
        assert len(chosen) == 2          # fewer than V-1: no spanning tree exists
