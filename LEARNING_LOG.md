# Learning log

What measuring these algorithms actually taught me — including the places where
my expectations were wrong, and one place where my *benchmark* was wrong before
any algorithm was.

Every number here comes from `reports/`, reproducible with
`python scripts/run_benchmarks.py`.

---

## 1. My benchmark was wrong before my algorithms were

The one worth putting first, because it nearly invalidated everything else.

Bubble sort measured a cost ratio of **7.17 per doubling of n** on nearly-sorted
input. That is impossible. Quadratic growth means 4.0; a ratio above 4 says the
work is growing faster than n², which bubble sort cannot do — its comparison
count is bounded by n².

So the algorithm was not the variable. My input generator was:

```python
for _ in range(max(1, n // 100)):
    i, j = rng.randrange(n), rng.randrange(n)   # arbitrary positions
    a[i], a[j] = a[j], a[i]
```

Both the *number* of swaps and the *distance* each element travels grew with n.
Counting inversions confirmed it:

| n | inversions | ratio |
|---|---:|---:|
| 64 | 7 | — |
| 256 | 258 | 15.2 |
| 1024 | 5,726 | 22.2 |
| 2048 | 24,838 | 4.3 |

Inversions grew roughly quadratically. The input was getting harder faster than
it was getting bigger, so the sweep measured two variables at once and attributed
both to n.

The fix was to bound displacement — each element moves at most `window`
positions — which makes inversions grow as O(n·window), i.e. linearly. Bubble
sort then measured 2.39, and insertion sort 2.10: adaptive, linear-ish, exactly as
theory says.

**What I took from it:** a measurement that disagrees with a *proven bound* is
almost always a broken measurement. The impossible number was the useful one,
because it was impossible — a plausible-but-wrong number would have shipped.
I now check every result against its theoretical ceiling before believing it.

---

## 2. Selection sort really has no best case

I knew "selection sort is O(n²) in all cases" as a sentence. The measurement made
it concrete in a way the sentence did not.

Comparisons at n = 2048, on **already sorted** input:

| algorithm | comparisons | vs insertion |
|---|---:|---:|
| selection | 2,096,128 | 1,024× |
| insertion | 2,047 | — |

Selection sort's doubling ratio is **4.01 on every one of the five input shapes**,
including sorted. It cannot exploit order because finding each minimum requires
scanning the entire remaining suffix — there is no early exit to add. Insertion
sort on the same input does exactly n−1 comparisons.

**What I took from it:** "best case" is not a property of a problem, it is a
property of whether the algorithm's structure lets it stop early. Bubble sort gets
an O(n) best case from one `swapped` flag. Selection sort has nowhere to put such
a flag.

---

## 3. The "switch to insertion sort below 32" rule is about input shape, not size

I had absorbed the folklore that library sorts fall back to insertion sort for
small arrays. I assumed the threshold was a property of constant factors. It is
also a property of the data.

| input shape | crossover | insertion at n = 8192 |
|---|---|---|
| random | **n = 64** | 92× *slower* than merge |
| nearly sorted | none in range | 8.2× *faster* than merge |

On nearly-sorted input insertion sort won at **every size tested up to 8192**, and
the ratio stayed flat at ~0.13 the whole way — the two lines are parallel on
log-log axes, meaning the same complexity class, not merely a constant-factor
edge. With bounded displacement k, insertion sort is O(n·k), which is linear in n.

My first attempt at this benchmark stopped at n = 256 and insertion sort won every
size. I nearly wrote that up as "insertion sort is surprisingly good." It only
meant my range was too narrow to contain the answer.

**What I took from it:** this is why Timsort detects existing runs rather than just
checking a size threshold. It is not optimising for small arrays; it is optimising
for *ordered* ones, and real data is very often nearly ordered.

---

## 4. A bad pivot does not make quicksort slow, it makes it crash

I expected fixed-pivot quicksort to be quadratic on sorted input. It is — 130,816
comparisons at n = 512 against the randomised version's 18,830, a 6.9× gap.

What I did not expect: at **n = 1024 on reversed input it raised
`RecursionError`.** Not slow — dead.

Every partition puts n−1 elements on one side, so recursion depth is n rather than
log n. Python's default limit is 1000 frames, so the algorithm hits a hard wall at
an input size any laptop should handle instantly. The randomised version sorts the
same input without noticing.

**What I took from it:** O(n²) *time* was the textbook answer, and it was the less
important half. The O(n) *space* is what actually kills the program. Analysing time
and ignoring space had been a habit.

---

## 5. Asymptotically better often means practically worse

The result that most changed how I read complexity claims.

String matching at n = 16,000:

| input | naive | KMP | Rabin-Karp |
|---|---:|---:|---:|
| adversarial (`aaaa…` for `aaa…b`) | 38.39 ms | **1.96 ms** | 2.85 ms |
| random text | **1.06 ms** | 0.90 ms | 2.75 ms |

On adversarial input KMP is 19.6× faster, exactly as advertised. On random text:

* KMP's advantage shrinks to 1.19× — its O(n+m) guarantee buys almost nothing,
  because naive search already breaks on the first character of nearly every
  alignment.
* **Rabin-Karp is 2.6× *slower* than the naive algorithm** it is supposed to
  improve on. Same asymptotic class, far heavier constant: modular arithmetic per
  character versus a single character comparison.

**What I took from it:** big-O describes the worst case as n grows. It does not
promise anything about the input you actually have, and a "better" algorithm can
lose badly on typical data. Rabin-Karp earns its place through multi-pattern
search, not through beating naive search on one pattern.

---

## 6. One decorator, a 16,981× speedup

Naive Fibonacci at n = 30 takes **78.5 ms**. Memoised, it takes **4.62 µs**. Same
recursion, same base case, one `@lru_cache`.

| n | naive | memoised | ratio |
|---|---:|---:|---:|
| 25 | 6.99 ms | 3.79 µs | 1,844× |
| 30 | 78.5 ms | 4.62 µs | **16,981×** |

The ratio itself grows with n, because the two functions are in different
complexity classes — O(φⁿ) against O(n). I could not benchmark n = 35 naively;
the projected runtime made it not worth the wall time.

**What I took from it:** dynamic programming stopped feeling like a technique to
memorise and started feeling like an observation to make — *is this recursion
recomputing subproblems?* The call count for naive `fib(n)` is `2·fib(n+1) − 1`,
so the algorithm's own inefficiency is measured by the sequence it computes.

The same shape shows up in longest increasing subsequence, where the speedup from
O(n²) to O(n log n) grows from 7.1× at n = 100 to **149× at n = 3200**. An
asymptotic improvement is not a fixed discount; it compounds.

---

## 7. An unbalanced BST is a linked list wearing a costume

Insert sorted keys into a plain BST and every node becomes a right child.

| n | random insertion | sorted insertion | log₂ n |
|---|---:|---:|---:|
| 3200 | 25 | **3,199** | 11.64 |

**128× taller.** Height is exactly n−1, so every search degrades from O(log n) to
O(n) — the structure still satisfies the search property, still returns correct
answers, and has thrown away its entire reason for existing.

Two follow-on details I had not appreciated:

* The recursive `in_order` traversal would overflow the stack on exactly the input
  most likely to produce a degenerate tree. Writing traversal iteratively is not
  stylistic here; it is what keeps the failure mode from compounding.
* Sorted insertion is not an adversarial edge case. Loading records from an indexed
  database, or inserting timestamps, produces it by default.

**What I took from it:** AVL and red-black trees stopped looking like optional
sophistication. They pay rotation cost on every write to buy a guarantee, and the
guarantee is what you were assuming you had.

---

## 8. What the load factor is actually protecting

Hash table lookup is O(1) "on average" — a claim I had accepted without asking
average over what. Disabling resizing and letting the table fill answers it:

| load factor | longest chain |
|---|---:|
| 0.25 | 1 |
| **0.75** (resize threshold) | **4** |
| 4.0 | 8 |
| 8.0 | 13 |

The chain length *is* the worst-case lookup cost, and it tracks the load factor
directly. O(1) is not a property of hashing; it is a property of keeping n/m
bounded, which is what the resize policy does and the only reason the constant
stays constant.

**What I took from it:** "amortised O(1)" is a claim with a precondition attached.
The doubling-and-rehash policy is not an implementation detail underneath the
complexity guarantee — it *is* the guarantee.

---

## 9. Where my measurement method breaks down

Being honest about the limits of the approach, because a tool that always agrees
with you is not measuring anything.

The complexity fitter classified randomised quicksort as **O(n)** on random input,
with a doubling ratio of 2.20. It is O(n log n). Across n = 256…16,384, `n log n`
grows only ~1.5× faster than `n`, so the two models fit almost equally well and a
small constant offset decides the winner. R² cannot separate adjacent classes over
one decade of n.

The doubling ratio is the more honest statistic: 2.20 is visibly above the 2.0 that
linear growth requires, and matches the 2.19–2.24 that merge sort and heap sort
(both provably O(n log n)) produce. So I read the ratio and treat the fitted label
as a hint.

One case where the near-linear reading is *real*, not an artifact: randomised
quicksort on few-unique input measured **1.96**. My implementation uses a three-way
partition, so duplicate keys land in the middle group and are never recursed into.
With only 5 distinct values the recursion depth is bounded by 5, making it
genuinely near-linear. Same number, opposite meaning — and only reading the
implementation distinguishes them.

**What I took from it:** curve fitting cannot tell you what an algorithm *is*, only
what the data is consistent with. Two adjacent classes need either a much wider
range of n or a proof. I stopped treating the fitted label as an answer.

---

## 10. Measuring without disturbing what you measure

A design problem I did not anticipate. Threading a counter through the algorithms
would have meant this:

```python
def insertion_sort(a, counter):
    counter.comparisons += 1        # in every branch
    if a[j] > key: ...
```

Three objections: the algorithm becomes unreadable, the instrumented version is no
longer the code you would actually write, and the counter slows the timing runs, so
timing and counting could not share one implementation.

Putting the instrumentation in the **data** instead solves all three — a `Probe`
that counts its own comparisons and a `TrackedList` that counts reads and writes,
both cooperating with the normal Python protocols. The sorts are written with plain
`a[i] < a[j]` and have no idea they are being measured. The same function is timed
on plain integers and counted on probes.

Then verifying my own documentation caught the design's limit. I had claimed the
tally measured comparisons, reads *and* writes for the sorts. It does not. Every
sort starts with `a = list(seq)` so that it never mutates the caller's data — and
that copy is a plain `list`, so the tracked list observes the copy and nothing
after it. Read and write counts describe the copy, not the sort.

Comparisons survive, because `Probe` elements are copied *by reference* into the
new list and keep reporting to the same tally wherever they end up. Since every
complexity claim here rests on comparison counts, the conclusions hold — but I had
been about to publish two columns of numbers that meant nothing. The sort
benchmarks now record comparisons only.

**What I took from it:** the observer-effect problem is real in performance work,
and the fix was to move the measurement to a layer the algorithm talks *through*
rather than one it has to know about. But "the instrumentation is transparent" cut
both ways: it was transparent enough that I stopped thinking about what it could
and could not see. Writing the worked example in the README, running it, and
finding `writes=0` is what surfaced it — documentation as a test.

Comparison counts also turn out to be far better than timings for complexity
fitting: exact, deterministic, and identical on any machine, whereas timings carry
cache effects and scheduler noise as apparent curvature.

---

## Summary of corrected expectations

| I assumed | Measurement showed |
|---|---|
| Every sort is faster on sorted input | Selection sort does 1,024× more work than insertion sort there |
| Insertion sort is for small arrays | It beats merge sort at n = 8192 when input is nearly sorted |
| A bad pivot means quicksort is slow | It means `RecursionError` at n = 1024 |
| A better complexity class means faster | Rabin-Karp is 2.6× slower than naive search on random text |
| O(1) hash lookup is a property of hashing | It is a property of bounding the load factor |
| Curve fitting identifies complexity | It cannot separate n from n log n over one decade |
| Benchmarks measure the algorithm | Mine measured my input generator |
