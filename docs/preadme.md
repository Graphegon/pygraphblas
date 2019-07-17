For the next example, lets take a problem that at first seems quite
different, but can be solved in a similarly with a different semiring.
In this case, we have a network of computer nodes, and the edge
weights are the reliablity of the links between them as a percentage
of success.

![Finding the Most Reliable Path](../docs/ReliablePath.svg)

How to we solve which path is the most reliable from A to C?  Like the
previous example, we operate along each path, but instead of adding
distances, probablity theory tells us that consecutive events must be
multiplied.  And instead of finding the minimum distance, we use the
max() function to find the path with maximum reliability.

In pygraphblas, the semiring that solves this problem is called
`pygraphblas.semiring.max_times_float`.


XXX Figure out UDTs for this

As a final example, let's look at a problem that again looks
completely different, but can be solved using yet another semiring.
In this graph, the nodes define a language structure, and the edges
are letters that can be used to build words.  How can we find every
possible word in this language?

![Finding All Words in a Language](../docs/AllWords.svg)

I'm sure you're starting to see the pattern here.  Instead of adding,
or multiplying, the path operator we use is concatenation.  An instead
of min() or max(), the final function is to take the union of all path
words.

In pygraphblas, this particular example does not yet have a semiring.
GraphBLAS does allow for the create of User Defined Types (UDTs).  So
it can solve these kinds of problems, but pygraphblas does not yet
implement UDTs.  But it is still a valid example of a graph problem
that can be solve with a semiring.  You can imagine the semring would
be called something like `union_concat_str`.

So what is the same, and what is different in these three examples?
The structure of the graph is the same, as is the *pattern* of
operation applied to get the solution.  The differences are that each
graph's edges are of different types, and the operations that are
applied are different.  These differences are encapsulated by the
semirings used, and the names of the semrings are clues as to what
they abstract:

- min_plus_int: The type is `int`, the inner operation is addition,
  the outer operation is min()

- max_times_float: The type is `float`, the inner operation is
  multiplication, the outer operation is max()

- union_concat_str (hypothetical): The type is `str`, the inner
  operation is concatenation, the outer operation is set union.

