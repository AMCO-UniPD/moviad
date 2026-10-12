Getting started
=================

Choose an environment
-----------------------

The package currently declares **Python 3.13 or newer**. Its default dependency
set includes ``faiss-gpu-cu11`` and is intended for a compatible Linux/CUDA
environment. CPU and Apple Silicon installations are not validated with this
manifest. Confirm FAISS, PyTorch, and CUDA compatibility on your training machine
before running experiments. The minimal integration keeps these requirements
and the existing lockfile unchanged.

From the repository root, in your configured training environment:

.. code-block:: bash

   uv sync --locked
   uv run python examples/train_stfpm_mvtec.py --help

Alternatively, install the package into an existing compatible environment:

.. code-block:: bash

   python -m pip install -e .

The library uses a ``src`` layout; installation is required before importing it.
Dataset images and some pretrained backbone weights are external assets and are
not downloaded by installing MoViAD.

Run the standard example
--------------------------

Provide an extracted MVTec AD directory containing category directories such as
``bottle/train/good`` and ``bottle/test``:

.. code-block:: bash

   uv run python examples/train_stfpm_mvtec.py \
     --dataset-path /path/to/mvtec --category bottle \
     --epochs 2 --batch-size 8 --save-path disk/checkpoints/stfpm.pt

The frozen teacher may download ImageNet weights on first use. The example
trains with the current generic trainer, evaluates image/pixel ROC AUC, and saves
a final state dictionary. It does not enable experiment tracking. It uses the
test split for periodic reporting; use a separately held-out validation dataset
when tuning hyperparameters for a publishable experiment.

Build the documentation independently
---------------------------------------

Documentation does not import the ML package or require CUDA:

.. code-block:: bash

   python3 -m venv .venv-docs
   .venv-docs/bin/python -m pip install -r docs/requirements.txt
   .venv-docs/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
   python3 -m http.server 8000 --directory docs/_build/html

Open http://localhost:8000. Read the Docs uses the same requirements and fails
on Sphinx warnings. Hosting remains unchanged until a maintainer publishes.
