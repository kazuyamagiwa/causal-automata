Glossary
========

Terms used throughout the docs and papers (Bayesian Automata research
report appendix).

Markov blanket
  The information bubble around a node: its parents, its children, and
  the parents of its children (spouses). Conditioning on the blanket
  renders the node independent of the rest of the network.

Conditional mutual information (CMI)
  How much unique information :math:`X` still carries about :math:`Y`
  after knowing :math:`Z`. Near-zero CMI means conditional independence
  given :math:`Z`.

Directed Acyclic Graph (DAG)
  A directed network with no directed cycles. Required for Bayesian
  networks so causal influence cannot feed back infinitely.

Adjacency matrix
  An :math:`N \times N` grid representing a graph. Entry :math:`(i, j)`
  nonzero means a directed edge :math:`i \to j`.

Precision matrix
  The inverse of a covariance matrix. Zeros (after conditioning) reveal
  partial correlations; a zero off-diagonal implies conditional
  independence given the other variables in the block.

Elevation
  Continuous node height :math:`E \in [0, 1]` used by the automaton to
  orient edges downhill and forbid cycles.

V-well / collider / sink
  A V-structure :math:`i \to k \leftarrow j`. Unshielded triples that
  show explaining-away (Berkson) — marginally weak :math:`|\rho_{ij}|`
  but stronger :math:`|\rho_{ij\mid k}|` — mark middle node :math:`k` as
  a collider. The automaton pushes :math:`E_k` below
  :math:`\min(E_i, E_j)` (and gently buoys the parents) so later birth
  prefers arrows into the sink.

Partial correlation (:math:`\rho`)
  Linear-Gaussian proxy for CMI used in
  :func:`causal_automata.bayesian.step`. Computed from the precision of
  a ridge-regularized sub-covariance of :math:`\{i, j\} \cup Z`.

Skeleton
  Undirected version of a DAG (edge present if either direction
  exists). Skeleton F1 scores recovery of unordered pairs.
