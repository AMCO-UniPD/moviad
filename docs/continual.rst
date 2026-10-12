Continual learning
====================

A task stream with shared contracts
-------------------------------------

``ContinualDataset`` creates training and test datasets for an ordered list of
categories. If omitted, it uses ``dataset_class.get_categories()``.
``ContinualTrainer`` invokes ``start_task``, ``train_task``, and ``end_task`` on
its strategy, then evaluates all tasks seen so far.

.. code-block:: python

   from moviad.scenarios.continual.continual_dataset import ContinualDataset
   from moviad.scenarios.continual.continual_trainer import ContinualTrainer
   from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
   from moviad.datasets.dataset_arguments import DatasetArguments
   from moviad.datasets.mvtec import MVTecDataset
   from moviad.models.stfpm.stfpm import STFPMTrainArgs
   from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc

   stream = ContinualDataset(
       DatasetArguments("/path/to/mvtec", img_size=(224, 224),
                        gt_mask_size=(224, 224)),
       MVTecDataset, categories=["bottle", "cable"],
   )
   strategy = FineTuning(model)  # an already constructed STFPM on device
   matrices = ContinualTrainer(
       continual_dataset=stream, continual_model=strategy, device=device,
       metrics=[RocAuc(MetricLvl.IMAGE)],
       training_args=STFPMTrainArgs(batch_size=8, epochs=2),
       logger=None,
   ).train()

Performance history
---------------------

The return value maps metric names to ``T × T`` NumPy arrays. Row ``i`` records
performance after training task ``i``; column ``j`` identifies the evaluated
task. Future tasks are left as NaN. The trainer prints average performance and
final average forgetting; an optional logger receives these reports.

Available strategies
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module / class
     - Role
   * - ``fine_tuning.FineTuning``
     - Train the same model successively on each category.
   * - ``replay.replay_model.Replay``
     - Mix retained samples with training data from the current task.
   * - ``single_model.SingleModel``
     - Reset the shared model for each task; optionally save per-task checkpoints.
   * - ``patchcore_cl.PatchCoreCL``
     - Model-specific continual memory-bank management.
   * - ``patchcore_clpp.PatchCoreCLPP``
     - PatchCore continual variant with its own memory update policy.
   * - ``patchcore_ftpp.PatchCoreFineTuningPP``
     - PatchCore fine-tuning variant.
   * - ``padim_cl``
     - Unimodal and multimodal Gaussian-statistics strategies, including Lite variants.
   * - ``cfa_continual.CFAContinual``
     - CFA-specific replay and centroid behavior.

All module names above are relative to
``moviad.scenarios.continual.strategies``. Strategies differ in supported model,
checkpoint handling, and lifecycle signatures. Review the implementation before
substituting them in an experiment. The minimal branch preserves those behaviors.
