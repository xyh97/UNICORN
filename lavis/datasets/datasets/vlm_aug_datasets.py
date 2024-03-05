"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
from collections import OrderedDict
import random
import math

from lavis.datasets.datasets.base_dataset import BaseDataset
import torch

number_to_word = {
    1: 'one', 2: 'two', 3: 'three'
}

number_to_word_2 = {
    1: 'first', 2: 'second', 3: 'third'
}

wrong_videos = [
    "activitynet/video_1fps_new/v_3TwqeiVbpS8.mp4",
    "activitynet/video_1fps_new/v_Si6LZFiQT3k.mp4",
    "activitynet/video_1fps_new/v_6DXH6kwMe-Q.mp4",
    "activitynet/video_1fps_new/v_vdYFwqfqgJA.mp4",
    "activitynet/video_1fps_new/v_0dkIbKXXFzI.mp4",
    "activitynet/video_1fps_new/v_90vop6PS2Y0.mp4",
    "activitynet/video_1fps_new/v_pA8QJ2ZoeBM.mp4",
    "activitynet/video_1fps_new/v_2YSsqivrvR4.mp4",
    "activitynet/video_1fps_new/v_iBEUNOMTr8M.mp4",
    "activitynet/video_1fps_new/v_D0RDF1ez-8Y.mp4",
    "youcook2/video_1fps_new/training/216/ffoRmenLSLs.mp4",
    "youcook2/video_1fps_new/training/221/UB1_MNpdvgs.mp4",
    # "youcook2/video_1fps_new/training/416/wKHC2gbRdA0.mp4",
    "youcook2/video_1fps_new/training/201/5cwg5mURihI.mp4",
    "youcook2/video_1fps_new/training/204/e-LKNWrv6TQ.mp4",
    # "youcook2/video_1fps_new/training/112/CgbEMrfzmQQ.mp4",
    "youcook2/video_1fps_new/validation/304/wii9jNiNl9Y.mp4",
    "youcook2/video_1fps_new/validation/121/hs2h7nb5PHQ.mp4",
    ]

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


class TemporalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format)
        self.prompts = [
        "Please predict start and end time of the following event: {}.",
        "Can you tell me the time window of this event: {}?"
    ]
        self.reverse_prompts = [
        "Describe the what is happening during the time window {} of the video concisely.",
        "Provide a brief description of the given video clip located in {}.",
        # "Offer a succinct explanation of the video presented in {}.",
        "Summarize the visual content of the video within the time window {}.",
        "Give a short and clear explanation of the video in {}.",
        # "Share a concise interpretation of the video in {}.",
        # "Relay a brief, clear account of the video in {}.",
        # "Render a clear and concise summary of the video clip in {}.",
        # "Write a terse but informative summary of the video in {}.",
    ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        start = ann["start"]
        end = ann["end"]
        duration = ann["duration"]
        # video_path = os.path.join(self.vis_root, vname)
        if '.mp4' not in vname:
            video_path = os.path.join(self.vis_root, 'howto100m/video_1fps', vname)
            if os.path.exists(video_path+'.mp4'):
                video_path += '.mp4'
            else:
                video_path += '.webm'
        else:
            video_path = os.path.join(self.vis_root, vname)
        # prob = random.random()
        video, start_offset, end_offset, crop = self.vis_processor(video_path, start=start, end=end)
        # print(vname, prob, start, end, start_offset, end_offset, self.num_bins)
        for _ in range(10):
            if video is None or vname in wrong_videos:
                index = random.randint(0, len(self.annotation) - 1)
                ann = self.annotation[index]
                vname = ann["video"]
                start = ann["start"]
                end = ann["end"]
                duration = ann["duration"]
                # video_path = os.path.join(self.vis_root, vname)
                if '.mp4' not in vname:
                    video_path = os.path.join(self.vis_root, 'howto100m/video_1fps', vname)
                    if os.path.exists(video_path+'.mp4'):
                        video_path += '.mp4'
                    else:
                        video_path += '.webm'
                else:
                    video_path = os.path.join(self.vis_root, vname)
                video, start_offset, end_offset, crop = self.vis_processor(video_path, start=start, end=end)
            else:
                break
        if crop:
            duration = min(end_offset - start_offset, duration)
        start = max(0, start - start_offset)
        end = min(end - start_offset, duration)
        
        rel_start = math.floor(start * self.num_bins / duration)
        rel_end = math.floor(end * self.num_bins / duration)
        if rel_end == self.num_bins:
            rel_end -= 1
        # print(vname, start, end, start_offset, end_offset)
        # context = self.text_processor(ann["context"])
        context = ""
        if context == "":
            context = ""
        else:
            if context[-1] == '.':
                context += " "
            else:
                context += ". "
        # context = ""
        prompt = random.choice(self.prompts)
        caption = self.text_processor(ann["query"].lower())
        context += prompt.format(caption)
        if self.output_format == "()":
            context += " The output format should be (start,end), where start and end represent start and end time respectively."
            text_output = "({},{})".format(rel_start, rel_end)
        else:
            context += " The output format should be <start><end>."
            text_output = "<{}><{}>".format(rel_start, rel_end)
            # text_output = self.text_processor(ann["text_output"])
        c, t, h, w = video.size()
        num_frames = self.vis_processor.n_frms
        if t < num_frames:
            padding = torch.zeros(c, num_frames-t, h, w)
            mask = torch.arange(num_frames).less(t).to(torch.long)
            video = torch.cat([video, padding], dim=1)
        else:
            mask = torch.ones(num_frames, dtype=torch.long)

        if self.text_processor.reverse:
            reverse_input = "The time window is represented in the format <start><end>."
            prompt = random.choice(self.reverse_prompts)
            reverse_input += " " + prompt.format(text_output)
            reverse_output = self.text_processor(ann["query"])
            return {
                "vid": vname,
                "video": video,
                "text_input": context,
                "text_output": text_output,
                "reverse_input": context,
                "reverse_output": text_output,
                "video_mask": mask,
                "qid": ann["qid"] if "qid" in ann else index,
                "split": ann["split"] if "split" in ann else 1
            }
        else:
            return {
                "vid": vname,
                "video": video,
                "text_input": context,
                "text_output": text_output,
                "video_mask": mask,
                "qid": ann["qid"] if "qid" in ann else index,
                "split": ann["split"] if "split" in ann else 1
            }


class TemporalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format)
        self.prompts = [
        'Please predict start and end time of the following event: {}.',
        "Can you tell me the time window of this event: {}?"
    ]

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        start = ann["start"]
        end = ann["end"]
        duration = ann["duration"]
        # video_path = os.path.join(self.vis_root, vname)
        if '.mp4' not in vname:
            video_path = os.path.join(self.vis_root, 'howto100m/video_1fps', vname)
            if os.path.exists(video_path+'.mp4'):
                video_path += '.mp4'
            else:
                video_path += '.webm'
        else:
            video_path = os.path.join(self.vis_root, vname)
        video, start_offset, end_offset, crop = self.vis_processor(video_path, start=start, end=end)
        for _ in range(5):
            if video is None:
                index = random.randint(0, len(self.annotation) - 1)
                ann = self.annotation[index]
                vname = ann["video"]
                start = ann["start"]
                end = ann["end"]
                duration = ann["duration"]
                if '.mp4' not in vname:
                    video_path = os.path.join(self.vis_root, 'howto100m/video_1fps', vname)
                    if os.path.exists(video_path+'.mp4'):
                        video_path += '.mp4'
                    else:
                        video_path += '.webm'
                else:
                    video_path = os.path.join(self.vis_root, vname)
                # video_path = os.path.join(self.vis_root, vname)
                video, start_offset, end_offset, crop = self.vis_processor(video_path, start=start, end=end)
            else:
                break
        duration = min(duration, end_offset-start_offset)
        # context = self.text_processor(ann["context"])
        # context = ""
        # if context == "":
        #     context += ""
        # else:
        #     if context[-1] == '.':
        #         context += " "
        #     else:
        #         context += ". "
        # context = ""
        # prompt = random.choice(self.prompts)
        prompt = self.prompts[0]
        caption = self.text_processor(ann["query"].lower())
        # context += " " + prompt.format(caption)
        context = prompt.format(caption)
        # context += " Use ; to separate multiple windows."
        # context += " The video is quantized into 75 equally spaced timestamps from <0> to <74>."
        context += " The output format should be <start><end>."
        # text_output = self.text_processor(ann["text_output"])
        c, t, h, w = video.size()
        num_frames = self.vis_processor.n_frms
        if t < num_frames:
            padding = torch.zeros(c, num_frames-t, h, w)
            mask = torch.arange(num_frames).less(t).to(torch.long)
            video = torch.cat([video, padding], dim=1)
        else:
            mask = torch.ones(num_frames, dtype=torch.long)
        return {
            "vid": vname,
            "video": video,
            "text_input": context,
            "video_mask": mask,
            # "text_output": text_output,
            "qid": ann["qid"] if "qid" in ann else index,
            "split": ann["split"] if "split" in ann else 1,
            "clip_start": start_offset,
            "clip_end": end_offset,
            "duration": duration,
            # "start": start,
            # "end": end
        }