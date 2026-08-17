import gc
import torch
from dataclasses import dataclass
from moviad.common.args import Args
from moviad.entrypoints.common import load_datasets
from moviad.models.uniad.uniad import UniAD, UniADTrainArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.trainers.trainer import Trainer
from moviad.utilities.evaluation.metrics import RocAuc, MetricLvl


@dataclass
class UniADArgs(Args):
    category: str = None
    backbone: str = "resnet18"
    ad_layers: list = None
    img_input_size: tuple = (224, 224)
    batch_size: int = 4
    epochs: int = 50
    device: torch.device = None
    save_path: str = None


def train_uniad(args: UniADArgs, logger=None):

    train_dataset, test_dataset = load_datasets(
        args.dataset_config, args.dataset_type, args.category, image_size=args.img_input_size
    )

    feature_extractor = CustomFeatureExtractor(
        args.backbone, args.ad_layers, args.device, True, False, None
    )

    model = UniAD(feature_extractor=feature_extractor, input_size=args.img_input_size)
    model.to(args.device)

    train_args = UniADTrainArgs(batch_size=args.batch_size, epochs=args.epochs)
    metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL)]

    trainer = Trainer(
        train_args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metrics=metrics,
        device=args.device,
        logger=logger,
        save_path=args.save_path,
    )

    trainer.train()

    del model
    torch.cuda.empty_cache()
    gc.collect()
