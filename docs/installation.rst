Installation
============

Requirements
------------

* Python 3.10 or newer
* Core stack: NumPy, NetworkX, scikit-learn
* Optional: JAX (CPU, CUDA, or Metal/MPS), gCastle + PyTorch for benchmarks

From PyPI (v0.3.1)
------------------

.. code-block:: bash

   pip install "causal-automata==0.3.1"
   pip install "causal-automata[jax]"      # portable CPU JAX
   pip install "causal-automata[cuda]"     # NVIDIA CUDA 13 JAX + Colab stack
   pip install "causal-automata[gcastle]"  # PC / GOLEM benchmarks

Or with ``uv``::

   uv pip install "causal-automata[jax]"
   uv add "causal-automata[jax]"   # inside a uv project

From a source checkout (contributors)
-------------------------------------

Editable install while developing the repo::

   pip install -e .
   pip install -e ".[jax]"
   pip install -e ".[docs]"   # Sphinx + RTD theme for local doc builds
   # or: uv pip install -e ".[jax,docs]"

CI and Read the Docs both resolve extras with **``uv``**
(``uv pip install ".[cpu]"`` in GitHub Actions;
``.readthedocs.yaml`` uses ``method: uv`` with extras ``docs`` + ``cpu``).

JAX accelerators (CPU / CUDA / MPS)
-----------------------------------

The ``jax_core`` engine is pure ``jax`` / ``jax.numpy``. Accelerator
selection is left to the installed wheel — there are no hardcoded
backend branches in the package.

**Portable CPU JAX** (default extra)::

   pip install "causal-automata[jax]"
   # or: uv pip install "causal-automata[jax]"

**NVIDIA CUDA** (Colab T4 / local GPU; see current JAX install docs)::

   # Runtime → Change runtime type → T4 GPU (Colab)
   pip install "causal-automata[cuda]"
   # Equivalent stack often used on Colab:
   #   pip install -U "jax[cuda13]" gcastle torch
   #   pip install causal-automata

**Apple Silicon Metal / MPS**::

   pip install "causal-automata[jax]"
   # Then install a Metal-enabled JAX wheel for your macOS / Python, e.g.:
   #   pip install jax-mps
   # (see upstream JAX / jax-mps docs for the current package name)

Smoke the compile after install::

   python -c "from causal_automata import warmup, run, __version__; print(__version__, warmup())"

You should see a device string such as ``jax backend=cpu ...``,
``gpu:...``, or an MPS/Metal device depending on the wheel.

When wrapping ``step`` / ``run`` in your own jitted code, keep ``ticks``
and ``k`` as Python integers (or bind them with ``functools.partial``).
See :doc:`jax_static`.

Build the docs locally
----------------------

Hosted docs: https://causal-automata.readthedocs.io/en/latest/

From a checkout (mirrors Read the Docs / CI)::

   uv venv
   uv pip install ".[docs,cpu]"
   cd docs && make html
   # open _build/html/index.html

Read the Docs builds from :file:`.readthedocs.yaml` at the repository root
using native ``uv`` (``python.install.method: uv``) with the ``docs`` and
``cpu`` extras — the same resolver as GitHub Actions CI.
