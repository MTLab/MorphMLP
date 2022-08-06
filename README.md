# [ECCV2022] MorphMLP \[[arxiv](https://arxiv.org/abs/2111.12527)\]

We current release the code and models for:

- [x] Kintics-400
- [x] Something-Something V1
- [x] Something-Something V2
-  ImageNet-1K: For our models training/testing on ImageNet-1K, and how to transfer the pretrained weight for video usage, you can refer [IMAGE.md](mlp_images/IMAGE.md). 




## Update

***Aug,3rd 2022***

**\[Initial commits\]:** 

1. Pretrained models on Kinetics-400, Something-Something V1



## Model Zoo

> The ImageNet-1K pretrained models, followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/1VIJTQtc69l11MxDNiq-OzyPEAlUVNHIx?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (cjwu)](https://pan.baidu.com/s/11oYfumslhIf7pdG3M-jYlQ).

### Note

- All the models are pretrained on ImageNet-1K. You can find those pre-trained models in [pretrained](https://drive.google.com/drive/folders/105DRws977iNnjEv-Hjfix5Q3JtzqoDUm?usp=sharing).
-  \#Frame = \#input_frame x \#crop x \#clip
  - \#input_frame means how many frames are input for model per inference
  - \#crop means spatial crops (e.g., 3 for left/right/center)
  - \#clip means temporal clips (e.g., 4 means repeted sampling four clips with different start indices) 

### Kinetics-400

| Model       | #Frame | Sampling Stride | FLOPs | Top1 | Model                                                        | Log                                                          | config                                                        |
| ----------- | ------ | --------------- | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MorphMLP-S | 16x1x4 | 4            | 268G  | 78.7 | [google](https://drive.google.com/file/d/1UVMYES1yXO9ZzJOHcYKbMmyX30w3s9uo/view?usp=sharing) | [google](https://drive.google.com/file/d/1WEYSe1RK68EHRehBZgzLUEOKDX_JGBsZ/view?usp=sharing) |[config](configs/K400/K400_MLP_S16x4.yaml) |
| MorphMLP-S | 32x1x4 | 4            | 532G  | 79.7 | [google](https://drive.google.com/file/d/1u9jjka6Ea-o5Isy1xb4Y_PR6LfqLg3QV/view?usp=sharing)                                                            | [google](https://drive.google.com/file/d/1ik9_OnG85boYGqXwN3TIaf1nDnE-J50V/view?usp=sharing) | [config](configs/K400/K400_MLP_S32x4.yaml) |
| MorphMLP-B | 16x1x4 | 4            | 392G  | 79.5 | [google](https://drive.google.com/file/d/1bmJcpcln9fVEj_o8fzFYHjup7dRGbJeD/view?usp=sharing) | [google](https://drive.google.com/file/d/1uazG3dahCxms2V1MMuvntkTkcCp0A1oV/view?usp=sharing) | [config](configs/K400/K400_MLP_B16x4.yaml) |
| MorphMLP-B | 32x1x4 | 4            | 788G | 80.8 | [google](https://drive.google.com/file/d/17iu9L5lnQ0CucXV1vvlJZDAF2RfXqdYu/view?usp=sharing) | [google](https://drive.google.com/file/d/17sCcKYb5F2axvPFd_TWAk74fC2bBGdK1/view?usp=sharing) | [config](configs/K400/K400_MLP_B32x4.yaml) |

### Something-Something V1

| Model       | Pretrain | #Frame | FLOPs | Top1 | Model                                                        | Log                                                          | config                                                        |
| ----------- | -------- | ------ | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MorphMLP-S | IN-1K     | 16x1x1 | 67G  | 50.6 | [soon] | [soon] |[config](configs/SSV1/SSV1_MLP_S16.yaml) |
| MorphMLP-S | IN-1K     | 16x3x1 | 201G  | 53.9 | [soon] | [soon] |[config](configs/SSV1/SSV1_MLP_S32.yaml) |
| MorphMLP-B | IN-1K     | 16x3x1 | 294G  | 55.1 | [google](https://drive.google.com/file/d/1Cz4xQ4Uad9AiQbmTwDElml_Dxe1Nw2SN/view?usp=sharing) | [google](https://drive.google.com/file/d/1QJ7QgB1TTrlbJMfYyMm4hmt0AW9YxkJz/view?usp=sharing) |[config](configs/SSV1/SSV1_MLP_B16.yaml) |
| MorphMLP-B | IN-1K     | 32x3x1 | 591G  | 57.4 | [google](https://drive.google.com/file/d/1yxwoR53L0qRU44MRx9gZM2D6YPU4_eZw/view?usp=sharing) | [google](https://drive.google.com/file/d/1YVHPDKhtjFvcSrAwXZWswBhgoyRR6q8g/view?usp=sharing) |[config](configs/SSV1/SSV1_MLP_B32.yaml) |

### Something-Something V2

| Model       | Pretrain | #Frame | FLOPs | Top1 | Model                                                        | Log                                                          | config                                                        |
| ----------- | -------- | ------ | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MorphMLP-S | IN-1K    | 16x3x1 | 201G  | 67.1 | [soon] | [soon] |[config](configs/SSV2/SSV2_MLP_S16.yaml) |
| MorphMLP-S | IN-1K    | 32x3x1 | 405G  | 68.3 | [soon] | [soon] |[config](configs/SSV2/SSV2_MLP_S32.yaml) |
| MorphMLP-B | IN-1K     | 16x3x1 | 294G  | 67.6 | [soon]| [soon] |[config](configs/SSV2/SSV2_MLP_B16.yaml) |
| MorphMLP-B | IN-1K    | 32x3x1 | 591G  | 70.1 | [soon]| [soon] |[config](configs/SSV2/SSV2_MLP_B32.yaml) |

## Usage

### Installation

Please follow the installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.

### Training

1. Download the pretrained models into the pretrained folder.

2. Simply run the training code as followed:
  ```shell
  python3 tools/run_net.py --cfg configs/K400/K400_MLP_S16x4.yaml DATA.PATH_PREFIX path_to_data OUTPUT_DIR your_save_path
  ```


**[Note]:**

- You can change the configs files to determine which type of the experiments.

- For more config details, you can read the comments in `slowfast/config/defaults.py`.

- To avoid **out of memory**, you can use `torch.utils.checkpoint` (will be updated soon):



### Testing

We provide testing example as followed:
### Kinetics400
```shell
python3 tools/run_net.py --cfg configs/K400/K400_MLP_S16x4.yaml DATA.PATH_PREFIX path_to_data TRAIN.ENABLE False  TEST.NUM_ENSEMBLE_VIEWS 4 TEST.NUM_SPATIAL_CROPS 1 TEST.CHECKPOINT_FILE_PATH your_model_path OUTPUT_DIR your_output_dir
```
### SomethingV1&V2
```shell
python3 tools/run_net.py   --cfg configs/SSV1/SSV1_MLP_B32.yaml DATA.PATH_PREFIX your_data_path TEST.NUM_ENSEMBLE_VIEWS 1 TEST.NUM_SPATIAL_CROPS 3 TEST.CHECKPOINT_FILE_PATH your_model_path OUTPUT_DIR your_output_dir
```

Specifically, we need to set the number of crops&clips and your checkpoint path then run multi-crop/multi-clip test:


 Set the number of crops and clips:

   **Multi-clip testing for Kinetics**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 1
   ```

   **Multi-crop testing for Something-Something**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 1
   TEST.NUM_SPATIAL_CROPS 3
   ```

 You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

##  Cite MorphMLP

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@article{zhang2021morphmlp,
  title={Morphmlp: A self-attention free, mlp-like backbone for image and video},
  author={Zhang, David Junhao and Li, Kunchang and Chen, Yunpeng and Wang, Yali and Chandra, Shashwat and Qiao, Yu and Liu, Luoqi and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2111.12527},
  year={2021}
}
```

## Acknowledgement

This repository is built based on [SlowFast](https://github.com/facebookresearch/SlowFast) repository.

