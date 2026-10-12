Development and release review
================================

Where code belongs
--------------------

- ``src/moviad/models``: model implementation and model-specific training arguments.
- ``src/moviad/datasets``: parsing and preprocessing using shared dataset arguments.
- ``src/moviad/scenarios/continual``: streams, orchestration, and strategies.
- ``src/moviad/trainers``: standard and chunked orchestration.
- ``src/moviad/utilities/evaluation``: evaluator and metrics.
- ``examples``: runnable, guarded scripts using current APIs.
- ``docs``: Sphinx source and local styles; generated HTML stays untracked.

Testing deliberately
----------------------

Run focused tests in a compatible ML environment. Review a test before running
it: existing training tests may require external data, CUDA, pretrained weights,
or tracking credentials. Import checks alone do not establish numerical parity.
For new regressions prefer small deterministic fixtures, no network downloads,
and no experiment-tracking login.

Documentation must pass:

.. code-block:: bash

   .venv-docs/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html

Local branch review
---------------------

``almost-main`` is the minimal integration branch. The separate
``refactoring-continual`` branch descends from it and contains consistency work.
Review their diffs separately:

.. code-block:: bash

   git diff origin/continual almost-main -- src pyproject.toml uv.lock
   git diff almost-main refactoring-continual

Before releasing, validate on the target Linux/CUDA environment with real
MVTec data, standard and continual training, checkpoint round trips, and
representative strategy/model combinations. Check the pinned MIRAGE reference
and resolve the version policy. All preparation in this session remains local;
publishing or pushing is a separate maintainer action.

Keep ``journal.md`` current with decisions, checks, known limitations, and next
steps so a new session can resume safely.
