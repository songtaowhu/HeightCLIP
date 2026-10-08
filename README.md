# HeightCLIP
This repository provides the implementation for **HeightCLIP**，a vision-language network designed for height estimation from monocular remote sensing imagery. For more details , please refer to our [paper](https://www.sciencedirect.com/science/article/pii/S1569843226004152).
## Installation
The code was developed and tested with the following main environment:
- `pytorch` 2.0.1
- `pytorch3d` 0.7.4
- `fvcore` 0.1.5
- `wandb`  (for experiment tracking)

Please note that additional Python packages may be required depending on your environment. You may need to install other dependencies when running the code.
## Datasets
We evaluate HeightCLIP on three publicly available remote sensing datasets: ISPRS Vaihingen, ISPRS Potsdam, and DFC2019.

Please download the datasets from their official sources and modify the corresponding data paths in the code to match your local directory structure before running the experiments.

### Data Organization
The dataset should be organized with the following directory structure:
```text
📂 data_dir
├── 📂 image    # Optical remote sensing images
└── 📂 ndsm     # Ground-truth normalized digital surface models (nDSMs)

```
 
## Training
```text
python train.py
```
After training, several checkpoint files will be saved in the `checkpoint` directory, including the checkpoint from the final epoch and the best-performing checkpoint.
## Inference
The best checkpoint obtained during training is used for inference and evaluation.
```text
python inference_pretrain.py
```
## Acknowledgement
Our code is mainly built upon the publicly available implementations of **CaBins** and **AdaBins**. We sincerely thank the authors for making their excellent works publicly available to the research community. We also acknowledge the valuable contributions of **CLIP** and its open-source implementation, which provided an important foundation for our work.
