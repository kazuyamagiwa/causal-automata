Installation
============

Requirements
------------

* Python 3.10 or newer
* Core stack: NumPy, NetworkX, scikit-learn
* Optional: JAX (CPU or CUDA), gCastle + PyTorch for benchmarks

From PyPI
---------

.. code-block:: bash

   pip install causal-automata
   pip install "causal-automata[jax]"     # CPU JAX
   pip install "causal-automata[cuda]"    # NVIDIA CUDA JAX + Colab-style stack
   pip install "causal-automata[gcastle]" # PC / GOLEM benchmarks

Or with ``uv``::

   uv add "causal-automata[jax]"

From a source checkout (contributors)
-------------------------------------

Editable install while developing the repo::

   pip install -e .
   pip install -e ".[jax]"
   pip install -e ".[docs]"   # Sphinx + RTD theme for local doc builds

JAX accelerators
----------------

**Portable CPU JAX** (no NVIDIA wheel)::

   pip install "causal-automata[jax]"

**NVIDIA / Colab T4** (CUDA 13 wheels; see current JAX install docs)::

   # Runtime → Change runtime type → T4 GPU
   pip install "causal-automata[cuda]"

Smoke the JAX compile after install::

   python -c "from causal_automata import warmup; print(warmup())"

Build the docs locally
----------------------

Hosted docs: https://causal-automata.readthedocs.io/en/latest/

From a checkout::

   pip install -e ".[docs,jax]"
   cd docs && make html
   # open _build/html/index.html

Read the Docs builds from :file:`.readthedocs.yaml` at the repository root
using the ``docs`` and ``jax`` extras.
