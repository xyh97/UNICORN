"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
import json
import pandas as pd
import numpy as np
from collections import OrderedDict
import random
import math
from decord import VideoReader
from lavis.datasets.datasets.base_dataset import BaseDataset
import torch


class HowTo100MDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format)
        self.vis_root = vis_root + '/howto100m/video_1fps'
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.num_bins = num_bins
        self.output_format = output_format
        self.titles = json.load(open("/nfs/data/data/howto100m/annotations/howto100m_vid_to_title.json"))

        self.prompts = [
            "Please predict start and end time of the following event in the format <start><end>: {}.",
            "Can you tell me the time window of this event in the format <start><end>: {}."
            ]
        self.duration = 64
        with open(os.path.join(os.path.dirname(self.vis_root), 'annotations/holdout.txt')) as f:
            holdout_vids = f.readlines()
            holdout_vids_set = set([i.strip() for i in holdout_vids])

        self.vid_to_asr_dict = json.load(open(os.path.join(os.path.dirname(self.vis_root), ann_paths[0][1:]), "r"))
        # all_vids = list(keep_vids_set)
        all_vids = list(self.vid_to_asr_dict.keys())
        all_vids = [i for i in all_vids if i not in holdout_vids_set]
    
        self.htm_vlen_df = pd.read_csv(os.path.join(os.path.dirname(self.vis_root), 'annotations/htm_vlen.csv'), names=['vid','vlen'])
        proper_vlen_vids = set(self.htm_vlen_df['vid'][(self.htm_vlen_df['vlen'] < 200) \
            & (self.htm_vlen_df['vlen'] > 32)].tolist())
        all_vids = [i for i in all_vids if i in proper_vlen_vids]
        all_vids = sorted(all_vids)
        self.video_info = all_vids
    
    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, index):
    
        num_retries = 10
        for _ in range(num_retries):
            try:
                vid = self.video_info[index]
                video_path = os.path.join(self.vis_root, vid)
                if os.path.exists(video_path+'.mp4'):
                    video_path += '.mp4'
                else:
                    video_path += '.webm'
                vr = VideoReader(video_path)
                vlen = len(vr)
                if vlen < 32:
                    index = random.randint(0, len(self.video_info) - 1)
                    continue
                duration = random.randint(32, min(vlen, 180))
                # duration = 64
                caps, (start_timestamp, end_timestamp) = self._get_text(vid, vlen, duration)
                end_timestamp = min(vlen, end_timestamp)
                text_output = ""
                # print(len(caps["text"]))
                query_idx = np.random.choice(range(len(caps["text"])))
                if len(caps["text"]) > 10:
                    selected_idx = np.random.choice(len(caps["text"]), 10, replace=False)
                    # query_idx = np.random.choice(selected_idx)
                else:
                    selected_idx = None
                    # query_idx = np.random.choice(range(len(caps["text"])))
                for i, (cap, start, end) in enumerate(zip(caps["text"], caps["start"], caps["end"])):
                    rel_start = math.floor(start * self.num_bins / duration)
                    rel_end = math.floor(end * self.num_bins / duration)
                    if rel_end == self.num_bins:
                        rel_end -= 1
                    if rel_start >= self.num_bins or rel_end >= self.num_bins:
                        print(vid, caps, duration, rel_start, rel_end)
                    if selected_idx is None or (selected_idx is not None and i in selected_idx):
                        # text_output += "{}({},{}).".format(self.text_processor(cap), rel_start, rel_end)
                        text_output += "{}<{}><{}>.".format(cap, rel_start, rel_end)
                    if i == query_idx:
                        query = cap
                        prompt = random.choice(self.prompts)
                        reverse_input = prompt.format(query)
                        # reverse_input += " The output format should be <start><end>."
                        # reverse_output = "({},{})".format(rel_start, rel_end)
                        reverse_output = "<{}><{}>".format(rel_start, rel_end)

                context = "Write a dense description for this video to include key events with their timestamps. Time window <start><end> should be inserted at the beginning of each sentence."
                video, _, _, _ = self.vis_processor(video_path, start=start_timestamp, end=end_timestamp)

            except:
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self.video_info) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # print(vid, text_output)
        return {
            "video": video,
            "vid": vid,
            "text_input": context,
            "text_output": text_output,
            "reverse_input": reverse_input,
            "reverse_output": reverse_output,
        }

    def _get_text(self, vid, vlen, duration):
        cap_df = pd.DataFrame.from_dict(self.vid_to_asr_dict[vid])
        cap_df = cap_df[cap_df['end'] < vlen]

        no_caption_flag = False

        if len(cap_df['end'].tolist()):
            last_timestamp = cap_df['end'].tolist()[-1]
        else:
            no_caption_flag = True
        
        if not no_caption_flag:
            if (cap_df['start'] < last_timestamp - duration - 1).sum() == 0:
                no_caption_flag = True
            else:
                start_idx = np.random.choice(
                    cap_df.index[cap_df['start'] < last_timestamp - duration])
                start_timestamp = int(round(cap_df.iloc[start_idx]['start']))
                end_timestamp = start_timestamp + duration

        sentences = []
        starts = []
        ends = []

        if not no_caption_flag:
            for idx in range(start_idx, len(cap_df)):
                cap_entry = cap_df.iloc[idx]
                text, s, e = cap_entry['text'], cap_entry['start'], cap_entry['end']
                # s, e = round(s), round(e)
                text = str(text).replace('\n',' ').strip()
                if len(text.split()) > 32:
                    text = ' '.join(text.split()[0:32])
                if len(text.split()) <= 6:
                    continue
                if s > end_timestamp or e-s < 1:
                    break
                elif e > end_timestamp:
                    e = end_timestamp

                trim_start = max(s - start_timestamp, 0)
                trim_end = min(e - start_timestamp, duration)
                if trim_end == trim_start:
                    break


                sentences.append(text)
                starts.append(trim_start)
                ends.append(trim_end)

        if len(sentences) == 0 or no_caption_flag:  # handle unlucky sampling
            text = '</s>'
            sentences.append(text)
            starts.append(0)
            ends.append(duration)
            if no_caption_flag:
                start_timestamp = 0
                end_timestamp = duration

        return {'text': sentences, 'start': starts, 'end': ends}, \
                (start_timestamp, end_timestamp)