"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import random

from lavis.datasets.datasets.base_dataset import BaseDataset
import torch


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["video"],
                "text": ann["text"],
                "video": sample["video"],
            }
        )


class TimeCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompts = [
        "Please predict start and end time of the following event: {}.",
        "Can you tell me the time window of this event: {}?"
    ]
        self.reverse_prompts = [
        "Describe the what is happening during the time window {} of the video concisely.",
        "Provide a brief description of the given video clip located in {}.",
        "Offer a succinct explanation of the video presented in {}.",
        "Summarize the visual content of the video with the range {}.",
        "Give a short and clear explanation of the video in {}.",
        "Share a concise interpretation of the video in {}.",
        "Relay a brief, clear account of the video in {}.",
        "Render a clear and concise summary of the video clip in {}.",
        "Write a terse but informative summary of the video in {}.",
    ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        # start = ann["start"]
        # end = ann["end"]
        start = 0
        end = -1
        video_path = os.path.join(self.vis_root, vname)
        video = self.vis_processor(video_path)
        # video = self.vis_processor(video_path, start=start, end=end)
        for _ in range(5):
            if video is None:
                index = random.randint(0, len(self.annotation) - 1)
                ann = self.annotation[index]
                vname = ann["video"]
                # start = ann["start"]
                # end = ann["end"]
                video_path = os.path.join(self.vis_root, vname)
                video = self.vis_processor(video_path)
                # video = self.vis_processor(video_path, start=start, end=end)
            else:
                break
        
        context = self.text_processor(ann["context"])
        if context == "":
            context = ""
        else:
            if context[-1] == '.':
                context += " "
            else:
                context += ". "
        # if not ann["reverse"]:
        prompt = random.choice(self.prompts)
        caption = self.text_processor(ann["caption"])
        context += prompt.format(caption)
        context += " The output format should be <start><end>."
        text_output = self.text_processor(ann["text_output"])
        # else:
        #     prompt = random.choice(self.reverse_prompts)
        #     context += "<start><end> indicates start and end time of a clip extracted from the whole video."
        #     context += " " + prompt.format(ann["caption"])
        #     text_output = self.text_processor(ann["text_output"])
        # print(video.shape)
        return {
            "video": video,
            "text_input": context,
            "text_output": text_output,
            "qid": ann["qid"] if "qid" in ann else index,
            "split": ann["split"] if "split" in ann else 1
        }


class TimeCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompts = [
        'Please predict start and end time of the following event: {}.',
        "Can you tell me the time window of this event: {}?"
    ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        start = 0
        end = -1
        video_path = os.path.join(self.vis_root, vname)
        video = self.vis_processor(video_path)
        for _ in range(5):
            if video is None:
                index = random.randint(0, len(self.annotation) - 1)
                ann = self.annotation[index]
                vname = ann["video"]
                video_path = os.path.join(self.vis_root, vname)
                video = self.vis_processor(video_path)
            else:
                break
    
        context = self.text_processor(ann["context"])
        # context = ""
        if context == "":
            context += ""
        else:
            if context[-1] == '.':
                context += " "
            else:
                context += ". "
        # context = ""
        prompt = random.choice(self.prompts)
        caption = self.text_processor(ann["caption"])
        context += " " + prompt.format(caption)
        context += " The output format should be <start><end>."
        text_output = self.text_processor(ann["text_output"])

        return {
            "video": video,
            "text_input": context,
            "text_output": text_output,
            "qid": ann["qid"] if "qid" in ann else index,
            "split": ann["split"] if "split" in ann else 1
        }