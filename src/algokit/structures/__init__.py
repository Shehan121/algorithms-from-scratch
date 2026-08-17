"""Data structures built from scratch.

Each module implements one structure with the operations that show why it
exists — not a complete container API. The goal is understanding the trade-off
each structure makes, so the comments concentrate on *why* an operation costs
what it costs.
"""

from algokit.structures.bst import BST
from algokit.structures.hash_table import HashTable
from algokit.structures.heap import MinHeap
from algokit.structures.linked_list import LinkedList
from algokit.structures.queue_stack import Queue, Stack
from algokit.structures.trie import Trie
from algokit.structures.union_find import UnionFind

__all__ = ["BST", "HashTable", "MinHeap", "LinkedList", "Queue", "Stack", "Trie", "UnionFind"]
