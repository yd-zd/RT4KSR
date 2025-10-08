# Towards Real-Time 4K Image Super-Resolution

**[Eduard Zamfir](https://scholar.google.com/citations?hl=en&user=5-FIWKoAAAAJ), [Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)**

[Computer Vision Lab, CAIDAS, University of Würzburg](https://www.informatik.uni-wuerzburg.de/computervision/home/)

Work part of the [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) Challenge @ CVPR 2023 in Vancouver

----
<img src="assets/rt4ksr_teaser.png" width="1000" />

## Abstract
Over the past few years, high-definition videos and images in 720p (HD), 1080p (FHD), and 4K (UHD) resolution have become standard. While higher resolutions offer improved visual quality for users, they pose a significant chal- lenge for super-resolution networks to achieve real-time performance on commercial GPUs. This paper presents a comprehensive analysis of super-resolution model designs and techniques aimed at efficiently upscaling images from 720p and 1080p resolutions to 4K. We begin with a simple, effective baseline architecture and gradually modify its design by focusing on extracting important high-frequency details efficiently. This allows us to subsequently downscale the resolution of deep feature maps, reducing the overall computational footprint, while maintaining high reconstruction fidelity. We enhance our method by incorporating pixel-unshuffling, a simplified and speed-up reinterpretation of the basic block proposed by NAFNet, along with structural re-parameterization. We assess the performance of the fastest version of our method in the new [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) challenge and demonstrate its potential in comparison with state-of-the-art efficient super-resolution models when scaled up. Our method was tested successfully on high-quality content from photography, digital art, and gaming content.

----

<a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.html"><img src="assets/paper.png" width="200" border="0"></a>  <a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.html"><img src="assets/report.png" width="200" border="0"></a>

&ensp;&ensp;&emsp;&emsp;&emsp;&emsp;[Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.html)
&ensp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Challenge Report](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.html)

---
## Installation

- Create conda environment:
```
conda create --name rtsr python==3.10
source activate rt4ksr
```
- Install PyTorch (see [PyTorch instructions](https://pytorch.org/get-started/locally/)). For example,
```
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
- Install the dependencies:
```
pip install -r requirements.txt
````

----
## Usage

### Data Preparation
We generate bicubically downscaled LR images online and test our models on the standard benchmarks for Super-Resolution which can be found [here](https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets). You can test any SR benchmark using our scripts and model weights for `x2` and `x3` SR. Please follow the dataset directory structure below:
````
dataroot/
|---testsets/
|   |---set5
|   |   |---test/
|   |   |   |---HR/
|   |   |   |---f"LR_bicubic_x{scale}"/

...
````

### Test Model
You can find all the necessary details of testing the models in `test.py`. The argument `--is-train` is always needed, because the training architecture must be loaded first before reparameterization. Add `--rep` when you want to run the inference using the reparameterized version.

````
python code/test.py --dataroot [DATAROOT] --checkpoint-id rt4ksr_[x2|x3] --scale [x2|x3] --arch rt4ksr_rep --benchmark ntire23rtsr --is-train
````

### Quick Inference (Added for Convenience)
For quick single-image upscaling and performance testing, we provide an enhanced inference script:

````
python code/inference.py --input [IMAGE_PATH] --output [OUTPUT_PATH] --scale [2|3]
````

**Features:**
- Single image processing with automatic output naming (`input--2x.png`)
- Optional output path specification
- Built-in performance benchmarking capabilities

**Example:**
````
python code/inference.py --input image.png --scale 2
# Outputs: image--2x.png (auto-generated filename)
````

---
## [NTIRE 2023 4K RTSR Benchmark](https://github.com/eduardzamfir/NTIRE23-RTSR)


| Method  | Scale            |PSNR (RGB)| PSNR (Y)| SSIM (RGB) | SSIM (Y) |
|---------|------------------|----------|---------|------------|----------|
| Bicubic | x2 (1080p -> 4K) | 33.916   | 36.664  |  0.8829    | 0.9160   |
|         | x3 (720p -> 4K)  | 31.302   | 33.812  |  0.8246    | 0.8656   |
| RT4KSR  | x2 (1080p -> 4K) | 34.193   | 37.013  | 0.8848     | 0.9180   |              
|         | x3 (720p -> 4K)  | 31.721   | 34.349  | 0.8300     | 0.8715   |

---
## Citation
```
@InProceedings{Zamfir_2023_CVPR,
    author    = {Zamfir, Eduard and Conde, Marcos V. and Timofte, Radu},
    title     = {Towards Real-Time 4K Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1522-1532}
}
```
