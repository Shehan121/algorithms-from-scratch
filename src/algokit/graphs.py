"""Graph algorithms over an adjacency-list representation.

Graphs are ``dict[node, list[(neighbour, weight)]]`` for weighted algorithms and
``dict[node, list[neighbour]]`` for unweighted ones. Adjacency lists cost
O(V + E) space against O(V^2) for a matrix, which is the right trade for the
sparse graphs that occur in practice. A matrix wins only when the graph is dense
or when O(1) "is there an edge?" matters more than iterating neighbours.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Hashable

from algokit.structures.heap import MinHeap
from algokit.structures.union_find import UnionFind

Node = Hashable
Graph = dict[Node, list[Node]]
WeightedGraph = dict[Node, list[tuple[Node, float]]]

__all__ = [
    "bfs",
    "dfs",
    "dfs_recursive",
    "shortest_path_unweighted",
    "dijkstra",
    "topological_sort",
    "has_cycle",
    "connected_components",
    "kruskal_mst",
]


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------


def bfs(graph: Graph, start: Node) -> list[Node]:
    """Breadth-first traversal. O(V + E).

    A ``deque`` is essential, not a stylistic choice: ``list.pop(0)`` is O(n),
    which would turn this into O(V^2 + E). It is the single most common way to
    accidentally ruin a BFS.

    Nodes are marked visited **when enqueued**, not when dequeued. Marking on
    dequeue lets a node be queued several times before it is first processed,
    which still terminates but does redundant work — and breaks the guarantee
    that the first time you reach a node is via a shortest path.
    """
    seen = {start}
    queue = deque([start])
    order: list[Node] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return order


def dfs(graph: Graph, start: Node) -> list[Node]:
    """Depth-first traversal, iterative. O(V + E).

    Structurally identical to :func:`bfs` — the *only* difference is popping
    from the same end that pushes (stack) rather than the opposite end (queue).
    That one change is what turns breadth-first into depth-first.

    Neighbours are pushed reversed so the traversal visits them left to right,
    matching the recursive version. Without it the iterative and recursive forms
    disagree on order, which makes them look like different algorithms.
    """
    seen: set[Node] = set()
    stack = [start]
    order: list[Node] = []

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in seen:
                stack.append(neighbour)
    return order


def dfs_recursive(graph: Graph, start: Node, seen: set[Node] | None = None) -> list[Node]:
    """Recursive DFS — shorter, but costs O(V) stack depth.

    On a path graph of 10,000 nodes this raises ``RecursionError`` while
    :func:`dfs` completes. The tests assert exactly that, because "use the
    iterative form on deep graphs" is only convincing once you have seen the
    recursive one fail.
    """
    if seen is None:
        seen = set()
    if start in seen:
        return []
    seen.add(start)
    order = [start]
    for neighbour in graph.get(start, ()):
        order.extend(dfs_recursive(graph, neighbour, seen))
    return order


def connected_components(graph: Graph) -> list[list[Node]]:
    """Components of an undirected graph, via repeated BFS. O(V + E)."""
    seen: set[Node] = set()
    components: list[list[Node]] = []
    for node in graph:
        if node not in seen:
            component = bfs(graph, node)
            seen.update(component)
            components.append(component)
    return components


# --------------------------------------------------------------------------
# Shortest paths
# --------------------------------------------------------------------------


def shortest_path_unweighted(graph: Graph, start: Node, goal: Node) -> list[Node] | None:
    """Shortest path by edge count, via BFS. O(V + E).

    On an unweighted graph BFS *is* the shortest-path algorithm, and there is no
    reason to reach for Dijkstra — BFS is O(V + E) against Dijkstra's
    O((V + E) log V). Every edge having equal weight means the frontier expands
    in distance order for free.

    Parent pointers reconstruct the path, which is cheaper than storing a path
    per queue entry (that would be O(V^2) memory in the worst case).
    """
    if start == goal:
        return [start]

    parents: dict[Node, Node | None] = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbour in graph.get(node, ()):
            if neighbour in parents:
                continue
            parents[neighbour] = node
            if neighbour == goal:
                return _rebuild(parents, goal)
            queue.append(neighbour)
    return None


def _rebuild(parents: dict[Node, Node | None], goal: Node) -> list[Node]:
    path = [goal]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def dijkstra(graph: WeightedGraph, start: Node) -> tuple[dict[Node, float], dict[Node, Node | None]]:
    """Single-source shortest paths for **non-negative** weights.

    O((V + E) log V) with a binary heap. Returns distances and parent pointers.

    Negative weights break it, and not subtly: Dijkstra finalises a node the
    first time it is popped, assuming no later path can improve it. A negative
    edge invalidates that assumption, so the answer is silently wrong rather
    than an error. Bellman-Ford is the O(VE) algorithm that tolerates them. This
    implementation raises instead of returning a wrong answer.

    Rather than decrease-key (which a binary heap does not support cheaply),
    stale entries are pushed and skipped on pop via the ``dist`` check. The heap
    can hold O(E) entries instead of O(V), which is the standard trade and does
    not change the asymptotic bound.
    """
    for node, edges in graph.items():
        for neighbour, weight in edges:
            if weight < 0:
                raise ValueError(
                    f"negative weight {weight} on edge {node}->{neighbour}; "
                    "Dijkstra requires non-negative weights"
                )

    dist: dict[Node, float] = {start: 0.0}
    parents: dict[Node, Node | None] = {start: None}
    heap = MinHeap([(0.0, _Key(start))])

    while heap:
        d, key = heap.pop()
        node = key.value
        if d > dist.get(node, float("inf")):
            continue  # a stale entry, already improved on
        for neighbour, weight in graph.get(node, ()):
            candidate = d + weight
            if candidate < dist.get(neighbour, float("inf")):
                dist[neighbour] = candidate
                parents[neighbour] = node
                heap.push((candidate, _Key(neighbour)))
    return dist, parents


class _Key:
    """Wrap a node so heap tuples never compare the nodes themselves.

    ``(distance, node)`` tuples fall back to comparing nodes when distances tie.
    That raises ``TypeError`` for unorderable nodes and, worse, imposes an
    arbitrary tie-break. Comparing equal here keeps ties resolved by heap order
    alone.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: "_Key") -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Key) and other.value == self.value


# --------------------------------------------------------------------------
# Ordering and cycles
# --------------------------------------------------------------------------


def topological_sort(graph: Graph) -> list[Node]:
    """Kahn's algorithm. O(V + E). Raises if the graph has a cycle.

    Repeatedly emit a node with in-degree zero and decrement its neighbours.
    The termination check is the useful part: if fewer than V nodes were emitted,
    the remainder all have incoming edges, which can only happen inside a cycle.
    So the algorithm doubles as cycle detection — a topological order exists if
    and only if the graph is acyclic.
    """
    in_degree: dict[Node, int] = {node: 0 for node in graph}
    for node in graph:
        for neighbour in graph[node]:
            in_degree.setdefault(neighbour, 0)
            in_degree[neighbour] += 1

    queue = deque(sorted((n for n, d in in_degree.items() if d == 0), key=repr))
    order: list[Node] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, ()):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(in_degree):
        raise ValueError("graph contains a cycle; no topological order exists")
    return order


def has_cycle(graph: Graph, directed: bool = True) -> bool:
    """Cycle detection by DFS colouring. O(V + E).

    Three colours, not two, and that is the crux. White = unvisited,
    grey = on the current recursion path, black = fully explored. A cycle is an
    edge to a **grey** node. Using only visited/unvisited would flag any
    re-encounter, including a diamond ``A->B, A->C, B->D, C->D``, which has no
    cycle at all.

    For undirected graphs the immediate parent must be excluded, otherwise every
    single edge looks like a two-node cycle.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[Node, int] = {node: WHITE for node in graph}
    for edges in graph.values():
        for neighbour in edges:
            colour.setdefault(neighbour, WHITE)

    def visit(node: Node, parent: Node | None) -> bool:
        colour[node] = GREY
        for neighbour in graph.get(node, ()):
            if not directed and neighbour == parent:
                continue
            if colour.get(neighbour, WHITE) == GREY:
                return True
            if colour.get(neighbour, WHITE) == WHITE and visit(neighbour, node):
                return True
        colour[node] = BLACK
        return False

    return any(colour[node] == WHITE and visit(node, None) for node in list(colour))


def kruskal_mst(nodes: int, edges: list[tuple[int, int, float]]) -> tuple[list[tuple[int, int, float]], float]:
    """Minimum spanning tree by Kruskal. O(E log E), dominated by the sort.

    Sort every edge by weight, then take each one whose endpoints are not
    already connected. Union-find is what makes "already connected?" cheap —
    with a BFS check per edge this would be O(E * V).

    The greedy choice is provably safe by the cut property: the lightest edge
    crossing any partition belongs to some MST. This is the clearest case in the
    repo of a greedy algorithm being *optimal* rather than merely a heuristic.
    """
    uf = UnionFind(nodes)
    chosen: list[tuple[int, int, float]] = []
    total = 0.0

    for u, v, weight in sorted(edges, key=lambda e: e[2]):
        if uf.union(u, v):
            chosen.append((u, v, weight))
            total += weight
            if len(chosen) == nodes - 1:
                break  # a spanning tree always has exactly V-1 edges
    return chosen, total
