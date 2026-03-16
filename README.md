# DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation (DITR)

[[`Paper`](https://arxiv.org/abs/2503.18944)] [[`Project Page`](http://vision.rwth-aachen.de/ditr)] [[`Weights`](https://huggingface.co/rwthvision/ditr/tree/main)] [[`Logs`](https://huggingface.co/rwthvision/ditr/tensorboard)] [[`BibTeX`](#-Citation)]

## 📢 News

- 2026-03-16: Code release.
- 2025-11-05: Accepted to [3DV 2026](https://3dvconf.github.io/2026/).
- 2025-05-27: Winner of the [CVPR 2025 ScanNet++ challenge](https://kaldir.vc.in.tum.de/scannetpp/cvpr2025).
- 2024-05-27: Accepted to [CVPR 2025 T4V Workshop](https://sites.google.com/view/t4v-cvpr25/).

## Setup

This code repository is originally based on [Pointcept](https://github.com/Pointcept/Pointcept/blob/c4363c86e1e77325645d44bc8052b338031f1b30).

### Dependencies
Tested with CUDA 12.4, Python 3.11 and the dependencies in the lock file. Should probably also work with newer versions.
```bash
# make sure to load CUDA 12.4 beforehand
uv sync --python 3.11
```

### Data
Follow the instructions in the [original README](https://github.com/Pointcept/Pointcept/blob/c4363c86e1e77325645d44bc8052b338031f1b30/README.md) to setup the datasets. For ScanNet, the following command needs to be run afterwards as well.

```bash
python pointcept/datasets/preprocessing/scannet/prepare_2d_data/prepare_raw_data.py \
 --scannet_path $SCANNET_SCANS_PATH \
 --output_path data/processed/scannet_images
```

## Train

Refer to the [original README](https://github.com/Pointcept/Pointcept/blob/c4363c86e1e77325645d44bc8052b338031f1b30/README.md). The injection configs are generally named `semseg-pt-v3m1-0-image` and the distillation configs are named `distill-pt-v3m1-0-distill`.
They can be found in the respective `configs/$DATASET` folder.

For example, to train the injection model on the nuscenes dataset with 2 GPUs, run the following command:
```bash
sh scripts/train.sh -g 2 -d nuscenes -n $EXPERIMENT_NAME -c semseg-pt-v3m1-0-image
```

For distillation, the following command can be used:
```bash
sh scripts/train.sh -g 2 -d nuscenes -n $EXPERIMENT_NAME -c distill-pt-v3m1-0-distill
```

For fine-tuning a distilled model, the following command can be used:
```bash
sh scripts/train.sh -g 2 -d nuscenes -n $EXPERIMENT_NAME -c semseg-pt-v3m1-0-base -o "weight=exp/nuscenes/$DISTILL_EXPERIMENT_NAME/model/model_last.pth"
```

## Model Zoo

### DITR (injected)
| 3D Backbone | Image Backbone | Dataset | Val mIoU | Exp Dir |
| --- | --- | --- | --- | --- |
| PTv3 | DINOv2 ViT-L | ScanNet | 80.5 | uploading...
| PTv3 | DINOv2 ViT-L | ScanNet200 | 41.2 | [link](https://huggingface.co/rwthvision/ditr/tree/main/scannet200/semseg-ptv3_dino-L)
| PTv3 | DINOv3 ViT-L | ScanNet200 | 42.3 | [link](https://huggingface.co/rwthvision/ditr/tree/main/scannet200/semseg-ptv3_dinov3-L)
| PTv3 | DINOv2 ViT-L | S3DIS | 74.1 | uploading...
| PTv3 | DINOv2 ViT-S | nuScenes | 82.8 | [link](https://huggingface.co/rwthvision/ditr/tree/main/nuscenes/semseg-ptv3_dino-S)
| PTv3 | DINOv2 ViT-B | nuScenes | 83.0 | [link](https://huggingface.co/rwthvision/ditr/tree/main/nuscenes/semseg-ptv3_dino-B)
| PTv3 | DINOv2 ViT-L | nuScenes | 83.1 | [link](https://huggingface.co/rwthvision/ditr/tree/main/nuscenes/semseg-ptv3_dino-L)
| PTv3 | DINOv2 ViT-g | nuScenes | 84.2 | [link](https://huggingface.co/rwthvision/ditr/tree/main/nuscenes/semseg-ptv3_dino-g)
| PTv3 | DINOv3 ViT-L | nuScenes | 83.9 | uploading...
| PTv3 | DINOv2 ViT-L | SemanticKITTI | 69.0 | uploading...

### D-DITR (distilled)
| 3D Backbone | Datasets | Exp Dir |
| --- | --- | --- |
| PTv3 | ScanNet | uploading...
| PTv3 | ScanNet + Structured3D | uploading...

### D-DITR (distilled + fine-tuned)
| 3D Backbone | Dataset | Val mIoU | Exp Dir |
| --- | --- | --- | --- |
| PTv3 | ScanNet | 79.2 | uploading...
| PTv3 | ScanNet200 | 37.7 | uploading...
| PTv3 | S3DIS | 75.0 | uploading...
| MinkUNet | ScanNet | 76.2 | uploading...

## Disclaimer
 
This software is a research prototype only and suitable only for test purposes. It has been published solely for use in research applications; it is not permitted to use this software in any kind of improper, disrespectful, defamatory, obscene, military or otherwise harmful application. This software is not suitable for use in or for products and/or services and in particular not in or for safety-relevant areas. It was solely developed for and published as part of the publication "DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation" and will neither be maintained nor monitored in any way.
 
## Acknowledgment
 
The research and development of this software by RWTH Aachen has been supported by Robert Bosch GmbH under the project "Context Understanding for Autonomous Systems".

## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry.

```
@InProceedings{knaebel2026ditr,
  title     = {{DINO} in the Room: Leveraging {2D} Foundation Models for {3D} Segmentation},
  author    = {Knaebel, Karim and Yilmaz, Kadir and de Geus, Daan and Hermans, Alexander and Adrian, David and Linder, Timm and Leibe, Bastian},
  booktitle = {2026 International Conference on 3D Vision (3DV)},
  year      = {2026}
}
```
