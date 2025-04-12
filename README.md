<p align="center">
  <a href="">
    <img src='assets/logo.svg' alt='ICLR2025_REALLOD_LOGO' width="250px"/><br/>
  </a>
</p>

<h2> 
  <p align=center> 
    [ICLR2025] Re-Aligning Language to Visual Objects with an Agentic Workflow 
  </p> 
</h2>

<div align="center">

📄 [**Table of Contents**](#-table-of-contents) | ✨ [**Home Page**](https://iclr.cc/virtual/2025/poster/29934) | 📚 [**Paper**](https://arxiv.org/abs/2503.23508)  | 📺 [**Video**](https://recorder-v3.slideslive.com/#/?token=ICLR2025__29934__yuming-chen-jiangyan-feng-ha)  | 🛠️ [**Install**](#️-dependencies-and-installation-) | 📖 [**Citation**](#-citation-) | 📜 [**License**](#-license-) | ❓ [**FAQ**](https://github.com/FishAndWasabi/Real-LODissues?q=label%3AFAQ+)

</div>

<span style="font-size: 18px;"><b>Key Point:</b> Rather than becoming assistants to enhance human productivity, agents hold a deeper value paradigm that they can establish workflows that serve as a <b>flywheel</b>, sustaining <b>high-value data assets</b> across AI industries. Our paper is an application in multimodal domains to demonstrate this potential. If this repo helps you, please consider giving us a :star2:! </span>

<b>Note:</b> This repository is also a <span style="color: orange;"><a href="https://github.com/open-mmlab/mmdetection">MMDetection</a> style</span> codebase for Languaged-based Object Detection! Please feel free to use it for your own projects! 

<b>TL; DR:</b> An agentic workflow including planning, tool use, and reflection steps to improve the alignment quality of language expressions and visual objects for LOD model.

This repository contains the official implementation of the following paper:

> **Re-Aligning Language to Visual Objects with an Agentic Workflow** </br>
> [Yuming Chen](http://www.fishworld.site/), [Jiangyan Feng](https://github.com/jyFengGoGo)<sup>\*</sup>, [Haodong Zhang](https://openreview.net/profile?id=~Haodong_Zhang2)<sup>\*</sup>, [Lijun Gong](https://scholar.google.com.hk/citations?user=CvmpmS0AAAAJ&hl=en), [Feng Zhu](https://zhufengx.github.io/), [Rui Zhao](https://zhaorui.xyz/), [Qibin Hou](https://houqb.github.io/)<sup>\#</sup>, [Ming-Ming Cheng](https://mmcheng.net), [Yibing Song](https://ybsong00.github.io/)<sup>\#</sup> </br>
> (\* denotes equal contribution. \# denotes the corresponding author.) </br>
> ICLR 2025 Conference </br>


## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [✨ News 🔝](#-news-)
- [🛠️ Dependencies and Installation 🔝](#️-dependencies-and-installation-)
- [🤖 Real-Agent 🔝](#-real-agent-)
- [🏡 Real-Data 🔝](#-real-data-)
- [🏗️ Real-Model 🔝](#️-real-model-)
- [📖 Citation 🔝](#-citation-)
- [📜 License 🔝](#-license-)
- [📮 Contact 🔝](#-contact-)
- [🤝 Acknowledgement 🔝](#-acknowledgement-)


## ✨ News [🔝](#-table-of-contents)

> Future work can be found in [todo.md](docs/todo.md).

- **Apr, 2025**: The 🏡 **Real-Data** is publicly available!
- **Apr, 2025**: The code of 👼 **Real-Model** is publicly available!
- **Jan, 2025**: 🔥 Our paper is accepted by ICLR 2025!

## 🛠️ Dependencies and Installation [🔝](#-table-of-contents)

> We provide a simple scrpit `install.sh` for installation, or refer to [install.md](docs/install.md) for more details.

1. Clone and enter the repo.

   ```shell
   git@github.com:FishAndWasabi/Real-LOD.git
   cd Real-LOD
   ```

2. Run `install.sh`.

   ```shell
   bash install.sh
   ```

3. Activate your environment!

   ```shell
   conda activate Real-LOD
   ```


## 🤖 Real-Agent [🔝](#-table-of-contents)

### Workflow

Coming soon!


<!-- ### Finetuning -->

<!-- Coming soon! -->

### Examples

<p align="center">
  <a href="">
    <img src='assets/real-lod_examples.png' alt='ICLR2025_REALMODEL_EXAMPLES'/><br/>
  </a>
</p>

## 🏡 Real-Data [🔝](#-table-of-contents)

### Data Information

The dataset is uploaded on [Hugging Face](https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data) and Baidu Yun. Below is the detailed information and corresponding data paths:

| **Src**       | **Scale** | **Img Num** | **Ins Num** | **Exp Num**  | **File**                          | **Baidu Yun**                                                                 |
|---------------|-----------|-------------|-------------|--------------|-----------------------------------|-------------------------------------------------------------------------------|
| O365          | Small     | 8,513       | 64,528      | 1,974,504    | `real-data-o365-small.jsonl`      | [Link](https://pan.baidu.com/s/1ow5YaAM-YpvMCb4q9WM9QA?pwd=uubj)              |
| O365          | Base      | 68,104      | 416,537     | 13,628,900   | `real-data-o365-base.jsonl`       | [Link](https://pan.baidu.com/s/1N3dcFztAsZ77bdZgchwmBw?pwd=e27p)              |
| O365          | Large     | 574,883     | 3,390,718   | 112,061,648  | `real-data-o365-large.jsonl`      | [Link](https://pan.baidu.com/s/1AByxuSTLHCrqwsaODd0N4w?pwd=a9wr)              |
| OI           | Small     | 19,888      | 36,069      | 1,069,254    | `real-data-openimage-small.jsonl` | [Link](https://pan.baidu.com/s/1XwbuEYylWFNnOPLbMFCq7A?pwd=pimc)              |
| OI           | Base      | 24,663      | 48,783      | 1,435,416    | `real-data-openimage-base.jsonl`  | [Link](https://pan.baidu.com/s/1HjsV_J5cxTfOOKiMRiBwKQ?pwd=brpx)              |
| OI           | Large     | 828,314     | 1,776,100   | 81,420,000   | `real-data-openimage-large.jsonl` | [Link](https://pan.baidu.com/s/1mjRP70aIxLI_rYobDy0BMA?pwd=g34j)              |
| LVIS          | -         | 94,171      | 99,815      | 3,078,400    | `real-data-lvis.jsonl`            | [Link](https://pan.baidu.com/s/1qTs_SQKHPBj_X6hPkceJ-g?pwd=v52y)              |


You can access the dataset through huggingface using the following commands:

```shell
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data
cd Real-LOD-Data/real-data
```

**Note**: The annotation files are provided, and the images remain sourced from their original datasets.

### Data Format

The dataset is structured in the following format:

```shell
{
  "filename": "path/to/image",
  "height": <image_height>,
  "width": <image_width>,
  "pairs": {
    <source_model>: {
      "bboxes": [
        [<x1>, <y1>, <x2>, <y2>],
        ...
      ],
      "category": <object_category>,
      "relation": <single/multi>,
      "positive_expressions": [
        <positive_expression_1>,
        <positive_expression_2>,
        ...
      ],
      "negative_expressions": [
        <negative_expression_1>,
        <negative_expression_2>,
        ...
      ]
    },
    ...
  }
}
```

- `filename`: Path to the image file.
- `height` and `width`: The height and width of the image.
- `pairs`: Object/expression pairs in the image:
  - `<source_model>`: The source model used to generate expressions (e.g., `vlm_short`, `vlm_long`, or `llm`).
  - `bboxes`: A list of bounding boxes, each defined by `[x1, y1, x2, y2]`.
  - `category`: The category of the object within the bounding box.
  - `relation`: Specifies whether the object is associated with a single or multiple expressions.
  - `positive_expressions`: A list of expressions that positively describe the object.
  - `negative_expressions`: A list of expressions that do not describe the object.



## 👼 Real-Model [🔝](#-table-of-contents)

### Train

#### 1.1 Data Preparation

The tree of training data:

```shell
├── data
│   ├── real-data
│   │   ├── real-data-o365-small.jsonl
│   │   ├── real-data-o365-base.jsonl
│   │   ├── real-data-o365-large.jsonl
│   │   ├── real-data-openimage-small.jsonl
│   │   ├── real-data-openimage-base.jsonl
│   │   ├── real-data-openimage-large.jsonl
│   │   └── real-data-lvis.jsonl
│   └── object365
│       ├── images
│       │   ├── train
│       │   │   ├── xxx.jpg
│       │   │   ├── ...
│   └── openimage
│       ├── train
│       │   ├── xxx.jpg
│       │   ├──...
│   └── coco 
|       ├── train2017
│       │   ├── xxx.jpg
│       │   ├──...
```

To obtain the images for the datasets mentioned, please refer to the following tools and URLs:

- **Object365**: https://github.com/FrancescoSaverioZuppichini/Object365-download
- **OpenImage**: https://github.com/cvdfoundation/open-images-dataset
- **COCO**: http://images.cocodataset.org/zips/train2017.zip


#### 1.2 Training with single GPU

```shell
python tools/train_real_model.py ${CONFIG_FILE} [optional arguments]
```

####  1.3 Training with multi GPU

```shell
CUDA_VISIBLE_DEVICES=${GPU_IDs} bash tools/dist_train_real_model.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

You could run `python tools/train_real_model.py --help` to get detailed information of this scripts.

<details>
<summary> Detailed arguments </summary>

```
positional arguments:
  config                train config file path

optional arguments:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the dir to save logs and models
  --amp                 enable automatic-mixed-precision training
  --auto-scale-lr       enable automatically scaling LR.
  --resume [RESUME]     If specify checkpoint path, resume from it, while if not specify, try to auto resume from the latest checkpoint in the work directory.
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like
                        key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```

</details>

### Evaluation

#### 1.1 Data Preparation

The tree of evaluation data:

```shell
├── data
│   ├── d3
│   ├── OVDEval
│   ├── omnilabel_val_v0.1.3
│   └── coco
│   └── object365
│   └── openimagesv5
```

To obtain the evaluation datasets, please refer to the following tools and URLs:

- **OmniLabel**: https://www.omnilabel.org/dataset/download
- **DOD**: https://github.com/shikras/d-cube?tab=readme-ov-file#download
- **OVDEval**: https://huggingface.co/datasets/omlab/OVDEval

#### 1.2 Model Checkpoint

We provide the model checkpoint [Real-Model_base](https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data/blob/main/real-model-ckpts/real-model_b-357a96d2.pth) and [Real-Model_tiny](https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data/blob/main/real-model-ckpts/real-model_t-d5faf209.pth) on the [HuggingFace](https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data), you can access them through these code:

```shell
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data
cd Real-LOD-Data/real-model-ckpts
```

#### 1.3 Evaluation with single GPU

```shell
python tools/dist_test_real_model.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

#### 1.4 Evaluation with multi GPU

```shell
CUDA_VISIBLE_DEVICES=${GPU_IDs} bash tools/dist_dist_test_real_model.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

You could run `python tools/dist_test_real_model.py --help` to get detailed information of this scripts.

<details>
<summary> Detailed arguments </summary>

```
positional arguments:
  config                test config file path
  checkpoint            checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the directory to save the file containing evaluation metrics
  --out OUT             dump predictions to a pickle file for offline evaluation
  --show                show prediction results
  --show-dir SHOW_DIR   directory where painted images will be saved. If specified, it will be automatically saved to the work_dir/timestamp/show_dir
  --wait-time WAIT_TIME
                        the interval of show (s)
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like
                        key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --tta
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```

</details>


### Examples

<p align="center">
  <a href="">
    <img src='assets/real-model_examples.png' alt='ICLR2025_REALMODEL_EXAMPLES'/><br/>
  </a>
</p>

## 📖 Citation [🔝](#-table-of-contents)

If you find our repo useful for your research, please cite us:

```
@inproceedings{chen2025realigning,
  title={Re-Aligning Language to Visual Objects with an Agentic Workflow},
  author={Yuming Chen and Jiangyan Feng and Haodong Zhang and Lijun GONG and Feng Zhu and Rui Zhao and Qibin Hou and Ming-Ming Cheng and Yibing Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=MPJ4SMnScw}
}
```

There are also relvant citations for other outstanding works in this repo:

```
@inproceedings{dang2024instructdet,
  title={Instruct{DET}: Diversifying Referring Object Detection with Generalized Instructions},
  author={Ronghao Dang and Jiangyan Feng and Haodong Zhang and Chongjian GE and Lin Song and Lijun GONG and Chengju Liu and Qijun Chen and Feng Zhu and Rui Zhao and Yibing Song},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=hss35aoQ1Y}
}

@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## 📜 License [🔝](#-table-of-contents)

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## 📮 Contact [🔝](#-table-of-contents)

For technical questions, please contact `chenyuming[AT]mail.nankai.edu.cn`.

For commercial licensing, please contact `cmm[AT]nankai.edu.cn`.

## 🤝 Acknowledgement [🔝](#-table-of-contents)

This repository borrows heavily from [mmdetection](https://github.com/open-mmlab/mmdetection), [grounding-dino](https://github.com/IDEA-Research/GroundingDINO), [peft](https://github.com/huggingface/peft), [transformers](https://github.com/huggingface/transformers),and [chatglm](https://github.com/THUDM/ChatGLM-6B).

For images from COCO, Objects365 and OpenImage, please see and follow their terms of use: [MSCOCO](https://cocodataset.org/#download), [Objects365](https://www.objects365.org/overview.html), and [OpenImage](https://storage.googleapis.com/openimages/web/index.html).

The README file is referred to [LED](https://github.com/Srameo/LED) and [LE3D](https://github.com/Srameo/LE3D/blob/main/README.md?plain=1).

We also thank all of our contributors.

<div align="center">
<a href="https://github.com/FishAndWasabi/RealLOD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FishAndWasabi/RealLOD" />
</a>
</div>