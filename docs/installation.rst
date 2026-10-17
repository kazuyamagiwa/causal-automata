Installation
============

Requirements
------------

* Python 3.10 or newer
* Core stack: NumPy, NetworkX, scikit-learn
* Optional: JAX (CPU or CUDA), gCastle + PyTorch for benchmarks

Core (CPU)
----------

.. code-block:: bash

   pip install -e .

or, with an editable checkout and optional extras:

.. code-block:: bash

   pip install -e ".[jax]"
   pip install -e ".[gcastle]"
   pip install -e ".[docs]"   # Sphinx + RTD theme for local builds

JAX accelerators
----------------

**Portable CPU JAX** (no NVIDIA wheel)::

   pip install -e ".[jax]"

**NVIDIA / Colab T4** (CUDA 13 wheels; see current JAX install docs)::

   # Runtime → Change runtime type → T4 GPU
   pip install -U "jax[cuda13]" gcastle torch networkx scikit-learn numpy
   pip install -e .

**Full NVIDIA extra** (mirrors the Colab stack in project metadata)::

   pip install -e ".[cuda]"

Smoke the JAX compile after install::

   python -c "from causal_automata import warmup; print(warmup())"

Build the docs locally
----------------------

.. code-block:: bash

   pip install -e ".[docs,jax]"
   cd docs && make html
   # open _build/html/index.html

Read the Docs builds from :file:`.readthedocs.yaml` at the repository root
using the ``docs`` and ``jax`` extras.
