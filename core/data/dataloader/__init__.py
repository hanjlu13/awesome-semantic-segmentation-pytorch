"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .ade import ADE20KSegmentation
from .cityscapes import CitySegmentation
from .mscoco import COCOSegmentation
from .pascal_aug import VOCAugSegmentation
from .pascal_voc import VOCSegmentation
from .sbu_shadow import SBUSegmentation

datasets = {
    "ade20k": ADE20KSegmentation,
    "pascal_voc": VOCSegmentation,
    "pascal_aug": VOCAugSegmentation,
    "coco": COCOSegmentation,
    "citys": CitySegmentation,
    "sbu": SBUSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
