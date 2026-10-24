JAX static arguments (``ticks`` and ``k``)
==========================================

The JAX entry points :func:`causal_automata.jax_core.step` and
:func:`causal_automata.jax_core.run` are compiled with :func:`jax.jit`.
Two parameters **must be compile-time constants** (Python scalars known
when the function is traced), not traced JAX arrays:

============ ==============================================================
Parameter    Why it is static
============ ==============================================================
``ticks``    ``run`` uses ``jax.lax.scan(..., length=ticks)``. Scan length
             is part of the XLA program shape.
``k``        Death / birth invert batched blocks of shape
             ``(2+k) × (2+k)``. Changing ``k`` changes the compiled graph.
============ ==============================================================

If you pass a traced value (for example a JAX array scalar, or a value
that depends on another traced input) for either argument, JAX raises
``ConcretizationTypeError`` during tracing.

How the library marks them static
---------------------------------

Both APIs use :func:`functools.partial` to bind ``static_argnames`` before
``jax.jit``::

   import functools
   import jax

   @functools.partial(jax.jit, static_argnames=("k", "anneal", "fuzzy", "multihop"))
   def step(A, E, Cov, ..., k: int = 2, ...):
       ...

   @functools.partial(
       jax.jit,
       static_argnames=("ticks", "k", "anneal", "fuzzy", "multihop"),
   )
   def run(A0, E0, Cov, ticks: int = 100, k: int = 3, ...):
       ...

Call sites must pass ordinary Python ``int`` values (or leave the
defaults). Different ``ticks`` / ``k`` pairs each get their own XLA
specialization on first use; subsequent calls with the same static args
reuse the cache.

Correct usage
-------------

Prefer the high-level scan loop when you want a fixed horizon::

   from causal_automata import run, default_conditioning_k

   n = Cov.shape[0]
   k = default_conditioning_k(n)          # Python int, e.g. 3 or 5
   A_final, E_final = run(A0, E0, Cov, ticks=100, k=k)

Or bind static kwargs once with :func:`functools.partial` if you wrap
``step`` yourself (for example inside a larger jitted training loop)::

   import functools
   from causal_automata.jax_core import step

   # Freeze static shapes for this experiment; recompile only if k changes.
   step_k3 = functools.partial(step, k=3, multihop=True)

   A, E = A0, E0
   for t in range(100):                  # Python loop length is fine
       A, E = step_k3(A, E, Cov)

When you need a custom ``jax.lax.scan`` over ticks, keep ``ticks`` as a
Python int and mark it static on *your* wrapper the same way::

   import functools
   import jax
   import jax.numpy as jnp
   from causal_automata.jax_core import step

   @functools.partial(jax.jit, static_argnames=("ticks", "k"))
   def evolve(A0, E0, Cov, ticks: int, k: int):
       def body(carry, _):
           A, E = carry
           return step(A, E, Cov, k=k), None

       (A, E), _ = jax.lax.scan(body, (A0, E0), xs=None, length=ticks)
       return A, E

   Af, Ef = evolve(A0, E0, Cov, ticks=100, k=5)

What goes wrong
---------------

These patterns force ``ticks`` or ``k`` into the traced computation and
fail::

   # BAD — ticks becomes a traced array scalar
   ticks = jnp.array(100)
   run(A0, E0, Cov, ticks=ticks, k=3)

   # BAD — k depends on a traced value
   @jax.jit
   def bad(A, E, Cov, n_edges):
       k = jnp.minimum(5, n_edges)       # traced
       return step(A, E, Cov, k=k)

   # BAD — bare @jax.jit without static_argnames on a scan wrapper
   @jax.jit
   def also_bad(A0, E0, Cov, ticks):
       return jax.lax.scan(..., length=ticks)  # length must be concrete

Fix by keeping ``ticks`` / ``k`` as Python ``int`` (or capturing them with
``functools.partial`` / ``static_argnames``) and only tracing ``A``, ``E``,
and ``Cov``.

Practical tips
--------------

* Sweeping ``k`` (``--k-sweep``) intentionally recompiles once per ``k``.
  That cost is paid at the first call for each value.
* ``anneal``, ``fuzzy``, and ``multihop`` are also static booleans: they
  change control flow / shapes inside the tick, so they are listed in
  ``static_argnames`` alongside ``k`` / ``ticks``.
* The NumPy prototype (:func:`causal_automata.numpy_step`) has no JIT
  constraints; use it when debugging threshold logic without worrying
  about concretization.
