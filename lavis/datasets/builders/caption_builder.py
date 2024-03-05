"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)
from lavis.datasets.datasets.vlm_datasets import TimeCaptionDataset, TimeCaptionEvalDataset
from lavis.datasets.datasets.vlm_aug_datasets import TemporalDataset, TemporalEvalDataset
from lavis.datasets.datasets.ht_dataset import HowTo100MDataset
from lavis.datasets.datasets.dvc_dataset import DVCDataset


@registry.register_builder("ht_pretrain_dvc")
class HowTo100MDVCBuilder(BaseDatasetBuilder):
    train_dataset_cls = HowTo100MDataset
    eval_dataset_cls = DVCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ht100m/dvc.yaml",
    }

@registry.register_builder("dvc_charades")
class DVCCharadesBuilder(BaseDatasetBuilder):
    train_dataset_cls = DVCDataset
    eval_dataset_cls = DVCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dvc/defaults.yaml",
    }

@registry.register_builder("dvc_activity")
class DVCActivityBuilder(BaseDatasetBuilder):
    train_dataset_cls = DVCDataset
    eval_dataset_cls = DVCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dvc/defaults.yaml",
    }


@registry.register_builder("dvc")
class DVCBuilder(BaseDatasetBuilder):
    train_dataset_cls = DVCDataset
    eval_dataset_cls = DVCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dvc/defaults.yaml",
    }


@registry.register_builder("ht_pretrain")
class HowTo100MBuilder(BaseDatasetBuilder):
    train_dataset_cls = HowTo100MDataset
    eval_dataset_cls = TemporalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ht100m/defaults.yaml",
    }


@registry.register_builder("temporal")
class TemporalBuilder(BaseDatasetBuilder):
    train_dataset_cls = TemporalDataset
    eval_dataset_cls = TemporalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/temporal/defaults.yaml",
    }


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("qvhighlights")
class QVHLBuilder(BaseDatasetBuilder):
    train_dataset_cls = TimeCaptionDataset
    eval_dataset_cls = TimeCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvhighlights/defaults.yaml",
    }

@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
