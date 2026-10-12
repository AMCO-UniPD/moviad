Standard workflows
====================

Data → features → model → metrics
-----------------------------------

``DatasetArguments`` holds the dataset root and preprocessing sizes. Dataset
classes receive that object, a category, and a split. MVTec loads its metadata
in the constructor; training items are image tensors and test items contain
``(image, label, mask, path)``.

.. code-block:: python

   from moviad.datasets.dataset_arguments import DatasetArguments
   from moviad.datasets.mvtec import MVTecDataset
   from moviad.utilities.configurations import Split

   args = DatasetArguments(
       dataset_path="/path/to/mvtec",
       img_size=(224, 224),
       gt_mask_size=(224, 224),
   )
   training = MVTecDataset(args, category="bottle", split=Split.TRAIN)
   testing = MVTecDataset(args, category="bottle", split=Split.TEST)

Model-specific training arguments
-----------------------------------

Models own their ``train_step`` / ``train_epoch`` implementation. For STFPM,
``STFPMTrainArgs`` initializes its optimizer; the generic ``Trainer`` receives
datasets and constructs data loaders internally.

.. code-block:: python

   from moviad.models.stfpm.stfpm import STFPMTrainArgs
   from moviad.trainers.trainer import Trainer
   from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc

   training_args = STFPMTrainArgs(batch_size=8, epochs=2)
   metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL)]
   trainer = Trainer(
       train_args=training_args, model=model,
       train_dataset=training, eval_dataset=testing,
       metrics=metrics, device=device, save_path=None,
   )
   trainer.train()

Here ``model`` and ``device`` must already be constructed. See the complete
:doc:`getting-started` example for backbone setup and checkpoint saving.
Use the training-argument subclass appropriate to your model; ``TrainingArgs``
alone does not initialize an optimizer.

Evaluation and checkpoints
----------------------------

``Evaluator.evaluate(model, dataloader, metrics, device)`` is a static method.
In evaluation mode the model must return ``(anomaly_maps, anomaly_scores)``.
Image metrics consume labels/scores; pixel metrics consume masks/maps. Evaluation
normalizes maps across the collected dataset by default. Constant maps require
care: the current normalization divides by their range.

For the generic trainer, leave ``save_path=None`` and explicitly save a state
dictionary after training:

.. code-block:: python

   from pathlib import Path
   import torch

   output = Path("disk/checkpoints/model.pt")
   output.parent.mkdir(parents=True, exist_ok=True)
   torch.save(model.state_dict(), output)

Generic Trainer's automatic final-save path currently calls an unavailable
``save_model`` method. This integration documents the workaround rather than
changing continual training behavior. Checkpoint reconstruction is model-specific;
instantiate matching feature extractors and model arguments before loading.
