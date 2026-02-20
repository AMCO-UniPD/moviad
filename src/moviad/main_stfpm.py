from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.mvtec import MVTecDataset
from torch.utils.data import Subset
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


def get_test(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Es. FT_stfpm, MT_stfpm, replay_stfpm", choices=["FT_stfpm", "MT_stfpm", "replay_stfpm"]) 
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Es. wide_resnet50_2", choices=BACKBONES)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", type=int, required=False)
    args = parser.parse_args()
    
    return args



def train_stfpm_FT(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=20, batch_size=16, lr=0.4, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)    
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    model = STFPM(teacher, student)
    model.to(device)
    continual_model = FineTuning(model)

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
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr),
        logger=wandb
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def train_stfpm_multi_task(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=50, batch_size=16, lr=0.4, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)    
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    train_dataset, test_dataset = dataset.get_all_tasks_data()

    model = STFPM(teacher, student)
    model.to(device)

    training_args = STFPMTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr)
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
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def train_stfpm_replay(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=10, batch_size=8, lr=0.4, device="cpu"):
    
    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)    
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)
    model = STFPM(teacher, student).to(device)

    continual_model = Replay(model, 100, 0.5)

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
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size, lr=lr),
        logger=wandb
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def main():
    args = get_test()
    backbone = args.backbone
    model = args.model
    numEpochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = args.device if args.device else 0

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    wandb.init(project="moviad_test", name=f"{model}_{backbone}_{numEpochs}_epochs_{batch_size}_minibatch_{lr}_lr")
    #wandb.define_metric("epoch")
    #wandb.define_metric("train_loss", step_metric="epoch")
    #wandb.define_metric("eval/*", step_metric="epoch")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    mvtec_transform = [
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]

    dataset_args = {
        "dataset_path" : "/mnt/disk1/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        #"image_transform_list" : None
        "image_transform_list": mvtec_transform
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
            "toothbrush"
        ],
    )

    if backbone == "mcunet-in3":
        layers = ["2", "6", "14"]
    else:
        layers = ["layer1", "layer2", "layer3"]

    if model == "FT_stfpm":
        train_stfpm_FT(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model == "MT_stfpm":
        train_stfpm_multi_task(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model == "replay_stfpm":
        train_stfpm_replay(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    else:
        raise NotImplementedError(f"Model {model} not implemented in tests.")



if __name__ == "__main__":
    main()