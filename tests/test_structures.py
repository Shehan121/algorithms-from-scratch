"""Data structure behaviour, including the failure modes worth knowing."""

import random

import pytest

from algokit.structures import BST, HashTable, LinkedList, MinHeap, Queue, Stack, Trie, UnionFind


class TestLinkedList:
    def test_push_pop_order(self):
        ll = LinkedList()
        for v in (1, 2, 3):
            ll.push_front(v)
        assert list(ll) == [3, 2, 1]
        assert ll.pop_front() == 3
        assert len(ll) == 2

    def test_constructor_preserves_order(self):
        assert list(LinkedList([1, 2, 3])) == [1, 2, 3]

    def test_reverse(self):
        ll = LinkedList([1, 2, 3, 4])
        ll.reverse()
        assert list(ll) == [4, 3, 2, 1]

    def test_remove(self):
        ll = LinkedList([1, 2, 3])
        assert ll.remove(2) and list(ll) == [1, 3]
        assert ll.remove(1) and list(ll) == [3]      # head removal path
        assert not ll.remove(99)

    def test_middle(self):
        assert LinkedList([1, 2, 3]).middle() == 2
        assert LinkedList([1, 2, 3, 4]).middle() == 2   # lower middle when even
        with pytest.raises(IndexError):
            LinkedList().middle()

    def test_cycle_detection(self):
        ll = LinkedList([1, 2, 3, 4])
        assert not ll.has_cycle()
        # Splice the tail back onto the head to create a cycle.
        node = ll._head
        while node.next is not None:
            node = node.next
        node.next = ll._head
        assert ll.has_cycle()


class TestStackQueue:
    def test_stack_is_lifo(self):
        s = Stack()
        for v in (1, 2, 3):
            s.push(v)
        assert s.peek() == 3
        assert [s.pop() for _ in range(3)] == [3, 2, 1]

    def test_queue_is_fifo(self):
        q = Queue()
        for v in (1, 2, 3):
            q.enqueue(v)
        assert q.peek() == 1
        assert [q.dequeue() for _ in range(3)] == [1, 2, 3]

    def test_queue_interleaved_operations(self):
        """The two-stack transfer must survive pushes arriving mid-drain."""
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        assert q.dequeue() == 1     # triggers the transfer
        q.enqueue(3)                # arrives in _in while _out still has 2
        assert q.dequeue() == 2
        assert q.dequeue() == 3
        assert not q

    def test_empty_errors(self):
        with pytest.raises(IndexError):
            Stack().pop()
        with pytest.raises(IndexError):
            Queue().dequeue()


class TestMinHeap:
    def test_pops_in_sorted_order(self):
        rng = random.Random(5)
        data = [rng.randrange(100) for _ in range(50)]
        heap = MinHeap(data)
        assert heap.sorted_drain() == sorted(data)

    def test_heapify_matches_repeated_push(self):
        data = [9, 4, 7, 1, 8, 2]
        built = MinHeap(data)
        pushed = MinHeap()
        for v in data:
            pushed.push(v)
        assert built.sorted_drain() == pushed.sorted_drain()

    def test_peek_is_minimum(self):
        heap = MinHeap([5, 3, 8])
        assert heap.peek() == 3
        heap.push(1)
        assert heap.peek() == 1

    def test_empty_errors(self):
        with pytest.raises(IndexError):
            MinHeap().pop()


class TestBST:
    def test_in_order_is_sorted(self):
        rng = random.Random(11)
        keys = list({rng.randrange(200) for _ in range(60)})
        tree = BST(keys)
        assert tree.in_order() == sorted(keys)

    def test_search_and_size(self):
        tree = BST([5, 3, 8])
        assert 5 in tree and 3 in tree and 99 not in tree
        assert len(tree) == 3

    def test_duplicates_ignored(self):
        tree = BST([1, 1, 1])
        assert len(tree) == 1

    def test_delete_all_three_cases(self):
        #        5
        #      3   8
        #     2 4 7 9
        tree = BST([5, 3, 8, 2, 4, 7, 9])
        assert tree.delete(2)                 # leaf
        assert tree.in_order() == [3, 4, 5, 7, 8, 9]
        assert tree.delete(3)                 # one child
        assert tree.in_order() == [4, 5, 7, 8, 9]
        assert tree.delete(5)                 # two children, root
        assert tree.in_order() == [4, 7, 8, 9]
        assert not tree.delete(999)

    def test_min_max(self):
        tree = BST([5, 2, 9])
        assert tree.min() == 2 and tree.max() == 9

    def test_degenerate_tree_is_a_list(self):
        """Sorted insertion collapses the tree - the motivation for AVL trees.

        Height equals n-1, so every operation is O(n) rather than O(log n).
        """
        n = 200
        tree = BST(range(n))
        assert tree.height() == n - 1
        assert tree.in_order() == list(range(n))   # iterative, so no stack overflow

    def test_random_tree_height_is_logarithmic(self):
        rng = random.Random(2)
        keys = list(range(2000))
        rng.shuffle(keys)
        tree = BST(keys)
        # Expected height for random insertion is ~4.31 log2(n); allow headroom.
        assert tree.height() < 4 * 11


class TestHashTable:
    def test_put_get_delete(self):
        h = HashTable()
        h.put("a", 1)
        h.put("b", 2)
        assert h.get("a") == 1 and "b" in h
        h.put("a", 99)
        assert h.get("a") == 99 and len(h) == 2   # overwrite, not insert
        assert h.delete("a") and "a" not in h
        assert not h.delete("a")

    def test_get_default(self):
        assert HashTable().get("missing", "fallback") == "fallback"

    def test_resizes_and_keeps_load_factor_bounded(self):
        h = HashTable(capacity=4)
        for i in range(200):
            h.put(i, i)
        assert len(h) == 200
        assert h.resizes > 0
        assert h.load_factor <= 0.75
        assert all(h.get(i) == i for i in range(200))

    def test_matches_dict_under_random_operations(self):
        rng = random.Random(9)
        h, reference = HashTable(), {}
        for _ in range(2000):
            key = rng.randrange(50)
            if rng.random() < 0.7:
                value = rng.randrange(1000)
                h.put(key, value)
                reference[key] = value
            else:
                assert h.delete(key) == (reference.pop(key, "!") != "!")
        assert len(h) == len(reference)
        assert all(h.get(k) == v for k, v in reference.items())

    def test_negative_hashes_are_handled(self):
        """hash() returns negatives; the bucket index must stay in range."""
        h = HashTable()
        for value in (-1, -999999, "text", (1, 2)):
            h.put(value, str(value))
        assert all(h.get(v) == str(v) for v in (-1, -999999, "text", (1, 2)))


class TestTrie:
    def test_insert_and_search(self):
        t = Trie(["car", "cart", "dog"])
        assert t.search("car") and t.search("cart")
        assert not t.search("ca")          # a prefix is not a word
        assert t.starts_with("ca")         # but it is a prefix

    def test_prefix_of_another_word_survives(self):
        """Inserting 'carpet' must not erase 'car'."""
        t = Trie(["car", "carpet"])
        assert t.search("car") and t.search("carpet")
        assert len(t) == 2

    def test_with_prefix(self):
        t = Trie(["car", "cart", "carbon", "dog"])
        assert t.with_prefix("car") == ["car", "carbon", "cart"]
        assert t.with_prefix("z") == []

    def test_longest_common_prefix(self):
        assert Trie(["interview", "internal", "internet"]).longest_common_prefix() == "inter"
        assert Trie(["abc", "xyz"]).longest_common_prefix() == ""

    def test_duplicate_insert_counted_once(self):
        t = Trie(["a", "a"])
        assert len(t) == 1

    def test_empty_string(self):
        t = Trie([""])
        assert t.search("") and len(t) == 1


class TestUnionFind:
    def test_union_and_connected(self):
        uf = UnionFind(6)
        assert uf.components == 6
        assert uf.union(0, 1) and uf.union(1, 2)
        assert uf.connected(0, 2)
        assert not uf.connected(0, 3)
        assert uf.components == 4

    def test_redundant_union_returns_false(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        assert not uf.union(1, 0)
        assert uf.components == 2

    def test_path_compression_keeps_the_forest_flat(self):
        """A chain of 1,000 unions must not leave a 1,000-deep path."""
        n = 1000
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        for i in range(n):
            uf.find(i)
        assert uf.max_depth() <= 2
        assert uf.components == 1

    def test_rejects_negative_size(self):
        with pytest.raises(ValueError):
            UnionFind(-1)
