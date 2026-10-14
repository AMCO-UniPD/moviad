Migration from the old main branch
====================================

The integration starts with continual's code and merges main's remaining
changes. One duplicate keyword in ``SingleModel`` was removed to restore its
importability. Main's historical root-level package and scripts are not restored.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Historical main
     - Current continual baseline
   * - ``moviad/`` at repository root
     - ``src/moviad/``; install the package before importing.
   * - ``setup.py`` / frozen requirements
     - ``pyproject.toml`` and ``uv.lock``.
   * - ``main_scripts/main_stfpm.py``
     - ``examples/train_stfpm_mvtec.py`` for the current documented workflow.
   * - ``TrainerSTFPM``
     - Generic ``Trainer`` with ``STFPMTrainArgs``.
   * - Dataset constructor keyword bundles
     - ``DatasetArguments``, category, and split.
   * - Constructed ``Evaluator`` instance
     - Static ``Evaluator.evaluate(...)``.
   * - Old evaluator / metrics module paths
     - ``moviad.utilities.evaluation``.

Legacy modules and unresolved issues
--------------------------------------

Some entrypoints and specialized trainers still import deleted builders,
loggers, or trainers. Presence in the repository does not establish support for
the current API. Training tests also include absolute dataset paths, network
weight downloads, and external tracking calls.

Import checks cover the current generic and continual trainers and the documented
STFPM example. They do not certify historical entrypoints or specialized trainers.
PhiNet is supplied by ``micromind.networks.phinet``; the old local checkpoint
directory is not an importable backbone module. ``common.Args`` retains its field
names but annotates dataset configuration with ``DatasetArguments`` and dataset
type with a ``VADDataset`` class.

The default dependency manifest is GPU-specific. Generic Trainer saving,
constant-map normalization, and MVTec contamination have focused regressions.
Legacy workflows and backbone checkpoint paths still require migration. Consult
``journal.md`` for the integration decisions and current validation limits.

MIRAGE remains an optional, pinned Git submodule from main. Its contents are not
part of the core package, and the integration does not initialize it. On a
machine where you explicitly want that project:

.. code-block:: bash

   git submodule update --init MIRAGE

Release version
-----------------

The package metadata remains at continual's ``0.1.0``. Historical main and old
docs used ``0.4``; choosing a new release version is a maintainer decision.
The documentation now derives its version directly from ``pyproject.toml``.
