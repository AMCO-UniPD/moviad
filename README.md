# MoViAD

A modular research library for visual anomaly detection, with standard training,
edge-oriented feature extraction, and continual learning over streams of tasks.

See the documentation in [link](https://amco-unipd.github.io/moviad/index.html).

The current library lives in `src/moviad`. This local integration preserves the
continual branch's implementations and incorporates main's MIRAGE submodule.

## Install and run

Use Python **3.13+** in a compatible Linux/CUDA environment. The current dependency
manifest includes GPU FAISS; CPU/macOS installation is not validated.

```bash
uv sync --locked
uv run python examples/train_stfpm_mvtec.py --help
uv run python examples/train_stfpm_mvtec.py \
  --dataset-path /path/to/mvtec --category bottle --epochs 2
```

The example uses a pretrained teacher, which may download weights. Datasets are
external. It does not enable experiment tracking. Standard training uses model
training arguments and the generic `Trainer`; continual training uses
`ContinualDataset`, a strategy, and `ContinualTrainer`.

## Documentation

Sphinx documentation includes setup, current APIs, standard and continual
workflows, migration notes, and development guidance. Build it independently
from the GPU stack:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
.venv-docs/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
python3 -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>. Read the Docs configuration is in
[.readthedocs.yaml](.readthedocs.yaml). No hosting or remote changes are made by
building locally.

## Components

- Models: STFPM, PaSTe, PaDiM/Lite, PatchCore, CFA, RD4AD, FastFlow, GANomaly,
  SimpleNet, SuperSimpleNet, and Dinomaly.
- Datasets: MVTec AD, MVTec AD 2, VisA, RealIAD, and MIIC.
- Continual strategies: fine tuning, replay, per-task models, and model-specific
  PatchCore, PaDiM, and CFA strategies.
- Feature extraction: torchvision backbones, MCUNet, MicroNet, and PhiNet.

Implementation maturity varies. Some older entrypoints and specialized trainers
still use historical APIs. Read [migration notes](docs/migration.rst) before
reusing old experiments.

## MIRAGE

Main's optional [MIRAGE](https://github.com/AMCO-UniPD/MIRAGE) submodule remains
pinned and separate from the core package. Initialize it only if needed:

```bash
git submodule update --init MIRAGE
```

## Local integration branches

`almost-main` contains the minimal merge and documentation.
`refactoring-continual` descends from it and contains consistency improvements.
[journal.md](journal.md) records decisions, validation, release gates, and how to
resume work. All integration work stays local.

## Contributing

Keep implementations under `src/moviad`, examples under `examples`, and
reference/guides under `docs`. Use deterministic offline fixtures for regressions;
existing training tests may need datasets, downloaded weights, and tracking
credentials. See [development guidance](docs/contributing.rst).

MoViAD is distributed under the [Apache 2.0 license](LICENSE). Individual vendored
components may carry their own license files. Preserve those notices.
