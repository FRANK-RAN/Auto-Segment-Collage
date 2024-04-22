# SoftCollage
The official PyTorch implementation of *SoftCollage: A Differentiable Probabilistic Tree Generator for Image Collage*.

## Abstract
Image collage task aims to create an informative and visual-aesthetic visual summarization for an image collection. While several recent works exploit tree-based algorithm to preserve image content better, all of them resort to hand-crafted adjustment rules to optimize the collage tree structure, leading to the failure of fully exploring the structure space of collage tree. Our key idea is to soften the discrete tree structure space into a continuous probability space. We propose *SoftCollage*, a novel method that employs a neural-based differentiable probabilistic tree generator to produce the probability distribution of correlation-preserving collage tree conditioned on deep image feature, aspect ratio and canvas size. The differentiable characteristic allows us to formulate the tree-based collage generation as a differentiable process and directly exploit gradient to optimize the collage layout in the level of probability space in an end-to-end manner. To facilitate image collage research, we propose AIC, a large-scale public-available annotated dataset for image collage evaluation. Extensive experiments on the introduced dataset demonstrate the superior performance of the proposed method. Data and codes are available at [here](https://github.com/ChineseYjh/SoftCollage).

## Citation
```
@InProceedings{Yu_2022_CVPR,
    author    = {Yu, Jiahao and Chen, Li and Zhang, Mingrui and Li, Mading},
    title     = {SoftCollage: A Differentiable Probabilistic Tree Generator for Image Collage},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3729-3738}
}
```

## Proposed dataset
The **AIC** and **ICSS** are available at [here](https://drive.google.com/file/d/1CsUNWLHCciq_0QlHmGoeGAYRcyzavB3z/view?usp=sharing).

## Run the codes
- Create an environment using ```conda create -n softcollage python=3.6 && conda activate softcollage && pip install -r requirements.txt```
- Download the AIC dataset and set *ICSS_DIR* in ```configs/SC.conf``` as **the path of the AIC dataset**.
- Run ```sh run.sh```

