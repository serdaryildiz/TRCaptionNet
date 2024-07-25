# !Paper! TRCaptionNet: A novel and accurate deep Turkish image captioning model with vision transformer based image encoders and deep linguistic text decoders

<font size='3'> <p align="center">
    <a href='https://scholar.google.com/citations?user=sl1KrkYAAAAJ&hl=tr'> Serdar Yıldız* </a> 
    <a href='https://scholar.google.com/citations?user=4_OxlcsAAAAJ&hl=tr'> Abbas Memiş </a>
    <a href='https://scholar.google.com/citations?user=DaCI6_YAAAAJ&hl=tr'> Songül Varlı </a>
</p></font>

<p align="center">
    <br />
    <br />
    <a href='https://doi.org/10.55730/1300-0632.4035'><img src='https://img.shields.io/badge/Paper-TUBITAK-red'></a>
    <a href='https://huggingface.co/spaces/serdaryildiz/TRCaptionNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> 
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


   




## Abstract

Image captioning is known as a fundamental computer vision task aiming to figure out and describe what is happening in an image or image region. Through an image captioning process, it is ensured to describe and define the actions and the relations of the objects within the images. In this manner, the contents of the images can be understood and interpreted automatically by visual computing systems. In this paper, we proposed the TRCaptionNet a novel deep learning-based Turkish image captioning (TIC) model for the automatic generation of Turkish captions. The model we propose essentially consists of a basic image encoder, a feature projection module based on vision transformers, and a text decoder. In the first stage, the system encodes the input images via the CLIP (contrastive language–image pretraining) image encoder. The CLIP image features are then passed through a vision transformer and the final image features to be linked with the textual features are obtained. In the last stage, a deep text decoder exploiting a BERT (bidirectional encoder representations from transformers) based model is used to generate the image cations. Furthermore, unlike the related works, a natural language-based linguistic model called NLLB (No Language Left Behind) was employed to produce Turkish captions from the original English captions. Extensive performance evaluation studies were carried out and widely known image captioning quantification metrics such as BLEU, METEOR, ROUGE-L, and CIDEr were measured for the proposed model. Within the scope of the experiments, quite successful results were observed on MS COCO and Flickr30K datasets, two known and prominent datasets in this field. As a result of the comparative performance analysis by taking the existing reports in the current literature on TIC into consideration, it was witnessed that the proposed model has superior performance and outperforms the related works on TIC so far.

![fig_model.png](fig%2Ffig_model.png)

## Installation

This project was developed on `torch 2.0.0 CUDA 11.8` and `Python 3.10`.


    git clone https://github.com/serdaryildiz/TRCaptionNet.git
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


## Dataset

For the COCO dataset, please visit the [TurkishCaptionSet-COCO](https://github.com/serdaryildiz/TurkishCaptionSet-COCO) repository.

For the Flickr30k dataset : [Flickr30k-Turkish](https://drive.google.com/drive/folders/1Exi-iL7I7JWevnncQU77uRnu851ow4br?usp=sharing)

## Checkpoint

### COCO-Test

| Model                                                                                                                                                                                                                   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-5 | METEOR | ROUGE-L | CIDEr  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|---------|--------|
| **CLIP ViT-B/16 + (no pretrain)**                                                                                                                                                                                       | 0.5069 | 0.3438 | 0.2190 | 0.1416 | 0.2221 | 0.4127  | 0.4934 |
| **CLIP ViT-B/32 + (no pretrain)**                                                                                                                                                                                       | 0.4795 | 0.3220 | 0.2056 | 0.1328 | 0.2157 | 0.4065  | 0.4512 |
| **CLIP ViT-L/14 + (no pretrain)**                                                                                                                                                                                       | 0.5262 | 0.3643 | 0.2367 | 0.1534 | 0.2290 | 0.4296  | 0.5209 |
| **CLIP ViT-L/14@336px + (no pretrain)**                                                                                                                                                                                 | 0.5325 | 0.3693 | 0.2376 | 0.1528 | 0.2338 | 0.4387  | 0.5288 |
| **ViT-B/16 + BERTurk**                                                                                                                                                                                                  | 0.5572 | 0.3945 | 0.2670 | 0.1814 | 0.2459 | 0.4499  | 0.6146 |
| **CLIP ViT-B/16 + (BERTurk)**                                                                                                                                                                                           | 0.5412 | 0.3802 | 0.2555 | 0.1715 | 0.2387 | 0.4419  | 0.5848 |
| [**CLIP ViT-L/14 + (BERTurk)**](https://drive.google.com/u/0/uc?id=14Ll1PIQhsMSypHT34Rt9voz_zaAf4Xh9&export=download&confirm=t&uuid=9b4bf589-d438-4b4f-a37c-fc34b0a63a5d&at=AB6BwCAY8xK0EZiPGv2YT7isL8pG:1697575816291) | 0.5761 | 0.4124 | 0.2803 | 0.1905 | 0.2523 | 0.4609  | 0.6437 |
| **CLIP ViT-L/14@336px + (BERTurk)**                                                                                                                                                                                     | 0.4639 | 0.3198 | 0.2077 | 0.1346 | 0.2276 | 0.4190  | 0.4971 |


### Flickr-Test

| Model                                                                                                                                                                                                                   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-5 | METEOR | ROUGE-L | CIDEr  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|---------|--------|
| **CLIP ViT-B/16 + (no pretrain)**                                                                                                                                                                                       | 0.4754 | 0.2980 | 0.1801 | 0.1046 | 0.1902 | 0.3732  | 0.2907 |
| **CLIP ViT-B/32 + (no pretrain)**                                                                                                                                                                                       | 0.4581 | 0.2866 | 0.1742 | 0.1014 | 0.1855 | 0.3754  | 0.2659 |
| **CLIP ViT-L/14 + (no pretrain)**                                                                                                                                                                                       | 0.5186 | 0.3407 | 0.2184 | 0.1346 | 0.2045 | 0.4058  | 0.3507 |
| **CLIP ViT-L/14@336px + (no pretrain)**                                                                                                                                                                                 | 0.5259 | 0.3525 | 0.2249 | 0.1334 | 0.2157 | 0.4237  | 0.3808 |
| **ViT-B/16 + BERTurk**                                                                                                                                                                                                  | 0.5400 | 0.3742 | 0.2533 | 0.1677 | 0.2232 | 0.4324  | 0.4636 |
| **CLIP ViT-B/16 + (BERTurk)**                                                                                                                                                                                           | 0.5182 | 0.3523 | 0.2348 | 0.1532 | 0.2105 | 0.4079  | 0.4010 |
| [**CLIP ViT-L/14 + (BERTurk)**](https://drive.google.com/u/0/uc?id=14Ll1PIQhsMSypHT34Rt9voz_zaAf4Xh9&export=download&confirm=t&uuid=9b4bf589-d438-4b4f-a37c-fc34b0a63a5d&at=AB6BwCAY8xK0EZiPGv2YT7isL8pG:1697575816291) | 0.5713 | 0.4056 | 0.2789 | 0.1843 | 0.2330 | 0.4491  | 0.5154 |
| **CLIP ViT-L/14@336px + (BERTurk)**                                                                                                                                                                                     | 0.4548 | 0.3039 | 0.1937 | 0.1179 | 0.2056 | 0.3966  | 0.3550 |

## Demo
to run demo for images:

    python demo.py --model-ckpt ./checkpoints/TRCaptionNet_L14_berturk.pth --input-dir ./images/ --device cuda:0

## Train

    python train.py --config $YOUR_CONFIG_PATH$

## Test

    python eval.py


    --config  : config path to just create model
    --device  : model device
    --weights  : model checkpoint çpth path
    --test-json  : dataset test .json file path
    --test-data  : test dataset input image dir path
    --dataset  : dataset name [coco-karphaty, flickr30k, tasviret]
    --batch-size  : batch size
    --num-worker  : number of worker for dataloader



## Citation

If you find our work helpful, please cite the following paper:

```
@article{yildiz2023trcaptionnet,
  author    = {Serdar Yildiz and Abbas Memis and Songül Varli},
  title     = {TRCaptionNet: A novel and accurate deep Turkish image captioning model with vision transformer based image encoders and deep linguistic text decoders},
  journal   = {Turkish Journal of Electrical Engineering and Computer Sciences},
  volume    = {31},
  number    = {6},
  year      = {2023},
  pages     = {11},
  doi       = {10.55730/1300-0632.4035},
  url       = {https://doi.org/10.55730/1300-0632.4035},
}
```

### Thanks to awesome works

- [BLIP](https://github.com/salesforce/BLIP)
- [ClipCap](https://github.com/rmokady/CLIP_prefix_caption)
