# [TIP2024] Convolution Enhanced Bi-Branch Adaptive Transformer with Cross-Task Interaction
PyTorch implementation of Convolution-Enhanced Bi-Branch Adaptive Transformer with Cross-Task Interaction for Food  Category and Ingredient Recognition [[Paper](https://ieeexplore.ieee.org/document/10471331)]

If you use the code in this repo for your work, please cite the following bib entries:

## Abstract
Recently, visual food analysis has received more and more attention in the computer vision community due to its wide application scenarios, e.g., diet nutrition management, smart restaurant, and personalized diet recommendation. Considering that food images are unstructured images with complex and unfixed visual patterns, mining food-related semantic-aware regions is crucial. Furthermore, the ingredients contained in food images are semantically related to each other due to the cooking habits and have significant semantic relationships with food categories under the hierarchical food classification ontology. Therefore, modeling the long-range semantic relationships between ingredients and the categories-ingredients semantic interactions is beneficial for ingredient recognition and food analysis. Taking these factors into consideration, we propose a multi-task learning framework for food category and ingredient recognition. This framework mainly consists of a food-orient Transformer named Convolution-Enhanced Bi-Branch Adaptive Transformer (CBiAFormer) and a multi-task category-ingredient recognition network called Structural Learning and Cross-Task Interaction (SLCI). In order to capture the complex and unfixed fine-grained patterns of food images, we propose a query-aware data-adaptive attention mechanism called Bi-Branch Adaptive Attention (BiA-Attention) in CBiAFormer, which consists of a local fine-grained branch and a global coarse-grained branch to mine local and global semantic-aware regions for different input images through an adaptive candidate key/value sets assignment for each query. Additionally, a convolutional patch embedding module is proposed to extract the fine-grained features which are neglected by Transformers. To fully utilize the ingredient information, we propose SLCI, which consists of cross-layer attention to model the semantic relationships between ingredients and two cross-task interaction modules to mine the semantic interactions between categories and ingredients. Extensive exper
iments show that our method achieves competitive performance on three mainstream food datasets (ETH Food-101, Vireo Food-172, and ISIA Food-200). Visualization analyses of CBiAFormer and SLCI on two tasks prove the effectiveness of our method.

![framework](Figures/figure1.png)
<p align="center">The illustration for our framework.</p>

![BiA-Attention](Figures/figure2.png)
<p align="center">The illustration for our proposed Bi-branch Adaptive Attention.</p>

## Requirements
Please, install the following packages:
- [PyTorch >= version 1.7](https://pytorch.org)
- tqdm == 0.46
- apex == 0.1
- einops==0.4.0
- importlib
- numpy
- pandas
- tensorboard
- timm == 0.5.4
- yacs==0.1.8

You may also install the environment through the requirement.txt file, please run:

```sh
$ conda install --file requirement.txt
```

## Datasets and pretrained models
We follow [IG-CMAN](https://github.com/minweiqing/Ingredient-Guided-Cascaded-Multi-Attention-Network-for-Food-Recognition) setting to use the same data index_list for training.  

For ETH Food-101, the dataset will be download from [here]()(Code: ).

For Vireo Food-172, the dataset will be download from [here](Code: ).

For ISIA Food-200, the dataset will be download from [here](https://github.com/minweiqing/Ingredient-Guided-Cascaded-Multi-Attention-Network-for-Food-Recognition).

You can find pretrained models from [here]()

## Training scripts
ETH Food-101
```
$ python -m torch.distributed.launch --nproc_per_node=4 train.py  --fp16 --name CBiAFormer --model_type CBiAFormer-B --dataset food101 --pretrained_dir pretrained_path --data_root data_path --img_size 384 --train_batch_size 1024  --learning_rate 0.008
```

Vireo Food-172
```
$ python -m torch.distributed.launch --nproc_per_node=4 train.py  --fp16 --name CBiAFormer --model_type CBiAFormer-B --dataset food172 --pretrained_dir pretrained_path --data_root data_path --img_size 384 --train_batch_size 1024  --learning_rate 0.008
```

ISIA Food-200
```
$ python -m torch.distributed.launch --nproc_per_node=4 train.py  --fp16 --name CBiAFormer --model_type CBiAFormer-B --dataset food200 --pretrained_dir pretrained_path --data_root data_path --img_size 384 --train_batch_size 1024  --learning_rate 0.008
```


## Reference
If you are interested in our work and want to cite it, please acknowledge the following paper:
```
@article{liu2024convolution,
  title={Convolution-Enhanced Bi-Branch Adaptive Transformer with Cross-Task Interaction for Food Category and Ingredient Recognition},
  author={Liu, Yuxin and Min, Weiqing and Jiang, Shuqiang and Rui, Yong},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
