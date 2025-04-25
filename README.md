# MCFNet

![Language](https://img.shields.io/badge/language-python-brightgreen) 

Our model was trained on an NVIDIA A800-SXM4-80GB GPU.

<div align="center">
    <img src="MCFNet.png" alt="framework" width="800"/>
</div>

## 👉 Data

We conducted 10 distinct data partitions based on [IF_CALC](https://github.com/Ding-Kexin/IF_CALC/blob/main/Model/index_2_data.py) implementation and adopted the average results across these iterations as the final reported outcomes in our study.

* [Houston](https://hyperspectral.ee.uh.edu/)

* [MUUFL](https://github.com/GatorSense/MUUFLGulfport/)

* [Trento](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)

## 🌈 Results

| Dataset  | OA (%) | AA (%) | Kappa (%) |
|----------|--------|--------|-----------|
| Houston    | 95.43 |  96.09 |    95.06  |
| MUUFL   | 85.97 |  82.11 |    81.80  |
| Trento  | 99.40 |  98.84 |    99.20  |

## 🌿 Getting Started

### Environment Setup

To get started, we recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment.
    
    conda create -n mcfnet python==3.11
    
    conda activate mcfnet
    
    pip install -r requirements.txt
    
    pip install PyWavelets


### Train and Test
    python demo.py

### Citation

We sincerely apologize for the partial code availability (MCFNet.py and FDFA.py) at this stage. The remaining code files will be fully released upon formal acceptance of the paper.

## 🌸 Acknowledgment

We are deeply grateful to repositories [IF_CALC](https://github.com/Ding-Kexin/IF_CALC), [GLT](https://github.com/Ding-Kexin/IEEE_TGRS_GLT-Net) and [FDNet](https://github.com/RSIP-NJUPT/FDNet.git), which served as the foundational basis for our code implementation.
