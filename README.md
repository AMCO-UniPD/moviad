<p align="center">
<img src="https://github.com/user-attachments/assets/93cf614e-2d18-4aa3-b670-80206c03bb7c" alt="drawing" style="width:300px;"/>
</p>

## Project Objective

MoViAD is an Open Source library for easy and modular Visual Anomaly Detection, built for industrial and research purposes. 
The library contains some State-of-the-Art models with their trainers, standard datasets, evaluator for calculating standard metrics, and a feature extractor.
The library structure is totally modular, allowing an easy isolation of the components needed for your project. 
The library will support different scenarios:
* Standard anomaly detection (model training and evaluation on a given category) ✅
* Contaminated anomaly detection (model training ed evaluation on a given category with a contaminate training set) ✅
* Continual anomaly detection (model training and evaluation considering a stream of tasks) 🚧 (Work in progress)
* Few shot anomaly detection 🚧 (Work in progress)


## How to Install

Inside the main repository directory, run the following command:

Editable mode (if you need to work on the code):

```bash
pip install -e ./
```

Fixed mode (if you just want to use the code):

```bash
pip install ./
```

## How to use the library

The library follows this structure:
- inside the <code>/models/model_name</code> directory is present the code of anomaly detection models
- inside the <code>/trainers</code> directory is present the code for training the anomaly detection models
- inside the <code>/datasets</code> directory is present the code for the anomaly detection datasets that must be used
- inside the <code>/utilies</code> directory is present the code for anomaly detection utilities

## Execution example

Inside the <code>/main_scripts</code> directory are present some execution scripts for training and testing the AD models. 

For example, for training patchcore: 

```bash
python main_scripts/main_patchcore.py --mode train --dataset_path /home/datasets/mvtec --category pill --backbone mobilenet_v2 --ad_layers features.4 features.7 features.10 --device cuda:0 --save_path ./patch.pt 
```

For every main script all its parameters are documented. 

**AD Models**

- PatchCore: [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html), [code](https://github.com/amazon-science/patchcore-inspection)
- CFA [paper](https://ieeexplore.ieee.org/abstract/document/9839549), [code](https://github.com/sungwool/CFA_for_anomaly_localization)
- Student-Teacher Feature Pyramid: [paper](https://arxiv.org/abs/2103.04257), [code](https://github.com/gdwang08/STFPM)
- PaSTe: [paper](https://arxiv.org/abs/2103.04257)
- PaDiM: [paper with code](https://paperswithcode.com/paper/padim-a-patch-distribution-modeling-framework)
- FastFlow: [paper](https://arxiv.org/abs/2111.07677)
- GANomaly: [paper](https://arxiv.org/abs/1805.06725)
- SuperSimpleNet: [paper](https://arxiv.org/abs/2408.03143)
- RD4AD: [paper](https://arxiv.org/abs/2201.10703)

**Feature Extraction Backbones**

- MobileNet: [link](https://paperswithcode.com/paper/mobilenets-efficient-convolutional-neural)
- PhiNet: [link](https://paperswithcode.com/paper/phinets-a-scalable-backbone-for-low-power-ai)
- MicroNet: [link](https://paperswithcode.com/paper/micronet-improving-image-recognition-with)
- MCUNet: [link](https://paperswithcode.com/paper/mcunet-tiny-deep-learning-on-iot-devices)
- WideResnet

**Datasets**

- MVTecAD: [link](https://paperswithcode.com/dataset/mvtecad)
- VisA: [link](https://paperswithcode.com/dataset/visa)
- RealIAD: [paper](https://arxiv.org/abs/2403.12580)
- MIIC Dataset: [link](https://github.com/wenbihan/MIIC-IAD)

**Submodules**

- MIRAGE: [repository](https://github.com/AMCO-UniPD/MIRAGE.git) - A comprehensive framework for anomaly detection integrated as a submodule. To clone with submodules, use:
  ```bash
  git clone --recurse-submodules https://github.com/your-repo/moviad.git
  ```
  Or if you already cloned without submodules, initialize them with:
  ```bash
  git submodule update --init --recursive
  ``` 

## Contribute

If you want to contribute to the repository, follow the present code structure: 
- inside the <code>/models/model_name</code> directory put the code for possible new anomaly detection models
- inside the <code>/trainers</code> directory put the code for training an anomaly detection model
- inside the <code>/datasets</code> directory put the code for possible new anomaly detection datasets that must be used

Every contribution must be open with a pull request. 

## Citations

If you use MoViAD in your work, please cite us! 🤗

Works that uses MoViAD: 
- "PaSTe: Improving the efficiency of visual anomaly detection at the edge", Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto. [paper](https://arxiv.org/abs/2103.04257)
- "From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithm", Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto. [paper](https://arxiv.org/pdf/2502.18328?)

