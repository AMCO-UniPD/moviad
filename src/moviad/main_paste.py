from moviad.models.paste.paste import PaSTe, PasteTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.mvtec import MVTecDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
import torch
from torchvision import transforms
import wandb

import argparse


BACKBONES = [
    "mcunet-in3",
    "micronet-m0",
    "micronet-m1",
    "micronet-m2",
    "micronet-m3",
    #"phinet_2.3_0.75_5",
    "phinet_1.2_0.5_6_downsampling",
    #"phinet_0.8_0.75_8_downsampling",
    #"phinet_1.3_0.5_7_downsampling",
    #"phinet_0.9_0.5_4_downsampling_deep",
    #"phinet_0.9_0.5_4_downsampling",
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
    "mobilenet_v2",
]


# NOTA: WideResNet50 non è supportato da PaSTe (il paper lo esclude esplicitamente:
# ha solo 4 layer block e rimuovere il primo causerebbe un calo drastico di performance).
# Per resnet18, vgg19_bn, efficientnet_b5 il paper non fornisce valori PaSTe:
# usare questi backbone con PaSTe richiede una scelta manuale dei layer.
BACKBONE_LAYER_CONFIG = {
    # Dalla Tabella 1 del paper
    "mobilenet_v2":                  {"ad_layers": [7, 10, 14], "bootstrap": 6},
    "mcunet-in3":                    {"ad_layers": [6, 10, 14], "bootstrap": 5},
    "phinet_1.2_0.5_6_downsampling": {"ad_layers": [5, 6, 7],  "bootstrap": 4},
    "micronet-m1":                   {"ad_layers": [3, 4, 5],   "bootstrap": 2},
    # Non trattati nel paper per PaSTe - valori da verificare manualmente
    "wide_resnet50_2":  {"ad_layers": [2, 3, 4],   "bootstrap": 1},   # NON consigliato
    "resnet18":         {"ad_layers": [4, 5, 6],   "bootstrap": 3},   # da verificare
    "vgg19_bn":         {"ad_layers": [7, 10, 14], "bootstrap": 6},   # da verificare
    "efficientnet_b5":  {"ad_layers": [7, 10, 14], "bootstrap": 6},   # da verificare
}

DEFAULT_LAYER_CONFIG = {"ad_layers": [7, 10, 14], "bootstrap": 6}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        help="Es. FT_paste, MT_paste, replay_paste",
        choices=["FT_paste", "MT_paste", "replay_paste"]
    )
    parser.add_argument(
        "--backbone", type=str, default="wide_resnet50_2",
        help="Es. wide_resnet50_2", choices=BACKBONES
    )
    parser.add_argument("--epochs",     type=int,   required=True)
    parser.add_argument("--batch_size", type=int,   required=True)
    parser.add_argument("--lr",         type=float, required=True)
    parser.add_argument("--device",     type=int,   required=False)
    return parser.parse_args()


def build_model(backbone: str, device: str) -> PaSTe:
    cfg = BACKBONE_LAYER_CONFIG.get(backbone, DEFAULT_LAYER_CONFIG)
    model = PaSTe(
        backbone_model_name=backbone,
        ad_layers=cfg["ad_layers"],
        weights="IMAGENET1K_V2",
        student_bootstrap_layer=cfg["bootstrap"],
        input_size=(256, 256),
        output_size=(256, 256),
    ).to(device)
    return model


def train_paste_FT(dataset, backbone="wide_resnet50_2", epochs=20, batch_size=16, lr=0.4, device="cpu"):
    model = build_model(backbone, device)
    continual_model = FineTuning(model)

    training_args = PasteTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr)

    trainer = ContinualTrainer(
        dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        training_args=training_args,
        logger=wandb,
    )

    # check for parameter updates
    params_before = [p.detach().clone() for p in model.parameters()]
    trainer.train()
    params_after = [p.detach() for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def train_paste_multi_task(dataset, backbone="wide_resnet50_2", epochs=50, batch_size=16, lr=0.4, device="cpu"):
    model = build_model(backbone, device)

    train_dataset, test_dataset = dataset.get_all_tasks_data()

    training_args = PasteTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.detach().clone() for p in model.parameters()]
    trainer.train()
    params_after = [p.detach() for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def train_paste_replay(dataset, backbone="wide_resnet50_2", epochs=10, batch_size=8, lr=0.4, device="cpu"):
    model = build_model(backbone, device)
    continual_model = Replay(model, 100, 0.5)

    training_args = PasteTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr)

    trainer = ContinualTrainer(
        dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            #RocAuc(MetricLvl.PIXEL),
            #AvgPrec(MetricLvl.IMAGE),
            #AvgPrec(MetricLvl.PIXEL),
            #F1(MetricLvl.IMAGE),
            #F1(MetricLvl.PIXEL),
            #ProAuc(MetricLvl.PIXEL),
        ],
        training_args=training_args,
        logger=wandb,
    )

    # check for parameter updates
    params_before = [p.detach().clone() for p in model.parameters()]
    trainer.train()
    params_after = [p.detach() for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def main():
    args = get_args()
    backbone   = args.backbone
    model_name = args.model
    epochs     = args.epochs
    batch_size = args.batch_size
    lr         = args.lr
    device_idx = args.device if args.device is not None else 0
    device     = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="moviad_test",
        name=f"{model_name}_{backbone}_{epochs}_epochs_{batch_size}_minibatch_{lr}_lr"
    )

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    mvtec_transform = [
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]

    dataset_args = {
        "dataset_path": "/mnt/disk1/manuel_barusco/datasets/mvtec",
        "img_size":        (256, 256),
        "gt_mask_size":    (256, 256),
        "image_transform_list": mvtec_transform,
    }

    continual_dataset = ContinualDataset(
        dataset_arguments=DatasetArguments(**dataset_args),
        dataset_class=MVTecDataset,
        categories=[
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "transistor",
            "metal_nut",
            "pill",
            "screw",
            "zipper",
            "toothbrush",
        ],
    )

    if model_name == "FT_paste":
        train_paste_FT(
            dataset=continual_dataset, backbone=backbone,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device
        )
    elif model_name == "MT_paste":
        train_paste_multi_task(
            dataset=continual_dataset, backbone=backbone,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device
        )
    elif model_name == "replay_paste":
        train_paste_replay(
            dataset=continual_dataset, backbone=backbone,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")


if __name__ == "__main__":
    main()