Component reference
=====================

Stable locations and contracts
--------------------------------

This reference documents the current source contracts without importing CUDA or
optional backbones during a documentation build.

.. py:class:: moviad.datasets.dataset_arguments.DatasetArguments(dataset_path, img_size=None, gt_mask_size=None, image_transform_list=None)

   Preprocessing configuration shared by dataset constructors. Sizes are height,
   width pairs. Provide both image and mask sizes for segmentation experiments.

.. py:class:: moviad.models.training_args.TrainingArgs(batch_size, epochs, evaluation_epoch_interval=1, optimizer=None, loss_function=None)

   Base training configuration. ``init_train(model)`` is the model-specific
   initialization hook. Existing ``__to_dict__()`` is used by experiment loggers.

.. py:class:: moviad.models.vad_model.VADModel

   PyTorch model base. Shared ``train_epoch`` averages ``train_step`` losses.
   Training and chunked fitting are model-specific. Reset/save/size hooks are
   placeholders unless a concrete model implements them.

.. py:class:: moviad.trainers.trainer.Trainer(train_args, model, train_dataset, eval_dataset, metrics, device, logger=None, logging_prefix="", save_path=None, saving_criteria=None)

   Builds loaders and runs the model's epoch method. ``train()`` takes no epoch
   argument. The current loaders use four workers. Evaluation expects a dataset
   at the configured interval. See :doc:`workflows` for saving limitations.

.. py:class:: moviad.scenarios.continual.continual_dataset.ContinualDataset(dataset_arguments, dataset_class, categories=None)

   Ordered category stream. ``get_task_data(i)`` returns a pair of datasets,
   not data loaders. ``get_task_data_evaluation(i)`` returns the test dataset.

.. py:class:: moviad.scenarios.continual.continual_trainer.ContinualTrainer(continual_dataset, continual_model, device, metrics, training_args, logger=None)

   Executes strategy lifecycle hooks and returns performance matrices from
   ``train()``. See :doc:`continual`.

.. py:method:: moviad.utilities.evaluation.evaluator.Evaluator.evaluate(model, dataloader, metrics, device, postprocess=min_max_norm)

   Static evaluation entrypoint. Returns metric names mapped to scalar values.

Model inventory
-----------------

Source implementations include STFPM, PaSTe, PaDiM/PaDiM Lite, PatchCore, CFA,
RD4AD, FastFlow, GANomaly, SimpleNet, SuperSimpleNet, and Dinomaly. These are not
interchangeable training APIs. Import concrete classes from their own modules
and inspect their training arguments before use.

Dataset inventory
-------------------

MVTec AD, MVTec AD 2, VisA, RealIAD, and MIIC have source implementations under
``moviad.datasets``. The current documented example targets MVTec AD.
Dataset layouts and split support are dataset-specific; not all implementations
have been migrated to identical contracts.

Feature extraction
--------------------

``moviad.utilities.custom_feature_extractor_trimmed.CustomFeatureExtractor``
accepts a backbone name, layer identifiers, device, and frozen/quantized flags.
Torchvision backbones use graph feature extraction. MCUNet, MicroNet, and PhiNet
have separate loading paths and may need external weights or dependencies.
