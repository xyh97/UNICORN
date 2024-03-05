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


class DVCDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_bins, output_format)
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.num_bins = num_bins
        self.output_format = output_format

        self.vid_to_asr_dict = json.load(open(ann_paths[0], "r"))

        all_vids = list(self.vid_to_asr_dict.keys())
        all_vids = sorted(all_vids)
        self.video_info = all_vids


        self.prompts = [
            # "Please predict start and end time of the following event in the format <start><end>: {}.",
            # "Can you tell me the time window of this event in the format <start><end>: {}."
            # 'Analyze the video in a comprehensive and detailed manner, discussing its themes and elements.',
            # 'Characterize the video using a well-detailed description, capturing its essence and events.',
            # 'Clarify the contents of the displayed video with great detail, focusing on its progression.',
            # 'Describe the following video in detail, including the actions and scenes.',
            # 'Examine the video closely and share its details, including the actions, characters, and setting.',
            # 'Explain the various aspects of the video before you, including the setting and actions.',
            # 'Give an elaborate explanation of the video you see, including the events and characters.',
            # 'Offer a thorough analysis of the video, discussing its various elements and storyline.',
            'Provide a detailed description of the given video, capturing its key moments.',
            # 'Share a comprehensive rundown of the presented video, highlighting its main sequences.',
            ]

    
    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, index):
    
        num_retries = 10
        for _ in range(num_retries):
            try:
                vid = self.video_info[index]
                if "qvhighlights" in vid:
                    vid = vid.split("+")[0]
                video_path = os.path.join(self.vis_root, vid)
                vr = VideoReader(video_path)
                vlen = int(len(vr) / vr.get_avg_fps())
                duration = random.randint(32, 150)
                duration = min(vlen, duration)
                # print(duration)
                if "charades" in vid or "webvid" in vid:
                    duration = vlen
                if "qvhighlights" in vid:
                    video, _, _, _ = self.vis_processor(video_path, start=0, end=vlen)
                    vid_ori = self.video_info[index]
                    query = self.text_processor(self.vid_to_asr_dict[vid_ori]['text'].lower())
                    context = 'Please predict start and end time of the following event: {}.'.format(query)
                    context += " The output format should be <start><end>."
                    num_frames = self.vis_processor.n_frms
                    mask = torch.ones(num_frames, dtype=torch.long)
                    return {
                        "vid": vid_ori,
                        "video": video,
                        "text_input": context,
                        "text_output": "",
                        # "reverse_input": reverse_input,
                        # "reverse_output": reverse_output,
                        "video_mask": mask,
                        "duration": vlen
                    }
                caps, (start_timestamp, end_timestamp) = self._get_text(vid, vlen, duration)
                # text_output = ""
                text_output = []
                query_idx = np.random.choice(range(len(caps["text"])))
                # if "qvhighlights" in vid:
                #     cap_idx = np.random.choice(range(len(caps["text"])))
                duration = end_timestamp - start_timestamp
                # if "charades/Charades_v1_480/WMR4G" in vid:
                #     print(caps, duration,start_timestamp, end_timestamp)
                reverse_input = ""
                reverse_output = ""
                for i, (cap, start, end) in enumerate(zip(caps["text"], caps["start"], caps["end"])):
                    # cap = self.text_processor(cap)
                    rel_start = math.floor(start * self.num_bins / duration)
                    rel_end = math.floor(end * self.num_bins / duration)
                    if rel_end == self.num_bins:
                        rel_end -= 1
                    if rel_start == 0 and rel_end == 0:
                        # print(vid, i, query_idx, caps)
                        continue
                    if rel_start == rel_end:
                        rel_start -= 1
                        # print("=", vid)
                        # continue
                    # if rel_start > 74 or rel_end > 74:
                    #     print(vid, caps, duration, start_timestamp, end_timestamp)
                    # text_output += "<{}><{}>;".format(rel_start, rel_end)
                    if "charades" not in vid:
                        cap = self.text_processor(cap)
                        cap = cap.capitalize()
                        cap += '.'
                    text_output.append(cap)
                    # text_output.append(f"<{rel_start}><{rel_end}>")
                    # text_output.append(f"<{rel_start}><{rel_end}>{cap}")
                    # if "qvhighlights" not in vid:
                        # text_output += "<{}><{}>{}.".format(rel_start, rel_end, cap)
                    # else:
                    #     if cap_idx == i:
                    #         text_output += "<{}><{}>{}.".format(rel_start, rel_end, cap)
                    if i == query_idx:
                        query = cap
                        prompt = random.choice(self.prompts)
                        reverse_input = prompt.format(query)
                        # reverse_input += " The output format should be <start><end>."
                        reverse_output = "<{}><{}>".format(rel_start, rel_end)
                # np.random.shuffle(text_output)
                # if reverse_input == "":
                # if len(text_output) > 5:
                #     text_output = text_output[:5]
                # np.random.shuffle(text_output)
                    # idx = random.randint(0, len(text_output)-5)
                    # text_output = text_output[idx:idx+5]
                # text_output = " ".join(text_output)
                text_output = " ".join(text_output)
                # text_output += "."
                # context = "[i,j] represents a continous time window and all i and j are between 0 to 74. Identify at most ten such windows for key events in this video. "
                # title = self.text_processor(self.titles[vid]['title'])
                # context = "The video is about {}. ".format(title)
                # context = f"Identify at most five time windows for key events in this video."
                # context = "Identify at most five time windows for key events in this video. The output format should be <start><end> and all values are between 0 to 74."
                # context += "Describe this video step by step. Each caption begins with its time window <start><end>."
                # context = "Locate one of potential time window for key events in this video. The output format should be <start><end>."
                # context = "Write a dense description for this video to include as many events with their timestamps as possible. Each sentence should follow the time window <start><end>."
                # context = "Predict a set of time windows for key events in this video. Use ; to separate each window in the format <start><end>."
                # context = "There are 50 frames in this video. A range [i,j] means frames from index i to index j describe the same event. Output at most five such ranges and the index is between 0 to 49."
                context = ""
                # context += random.choice(self.prompts)
                if "charades" not in vid:
                    title = self.vid_to_asr_dict[vid]["title"]
                    if title != "":
                        if "activitynet" in vid:
                            context += "This video is about {}. ".format(self.text_processor(title.lower()))
                        elif "youcook" in vid:
                            context += "This instructional video is about {}. ".format(self.text_processor(title.lower()))

                if "youcook" in vid:
                    context += "Describe the video in detail with key steps to complete the task."
                    # context += "Describe the video in detail with key steps to cook this food."
                else:
                    context += random.choice(self.prompts)

                video, _, _, _ = self.vis_processor(video_path, start=start_timestamp, end=end_timestamp)

            except Exception as e:
                # print(e)
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                # print()
                index = random.randint(0, len(self.video_info) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # print(vid)
        c, t, h, w = video.size()
        num_frames = self.vis_processor.n_frms
        if t < num_frames:
            padding = torch.zeros(c, num_frames-t, h, w)
            mask = torch.arange(num_frames).less(t).to(torch.long)
            video = torch.cat([video, padding], dim=1)
        else:
            mask = torch.ones(num_frames, dtype=torch.long)

        return {
            "vid": vid,
            "video": video,
            "text_input": context,
            "text_output": text_output,
            # "reverse_input": reverse_input,
            # "reverse_output": reverse_output,
            "video_mask": mask,
            "duration": vlen
        }

    def _get_text(self, vid, vlen, duration):
        cap_df = pd.DataFrame.from_dict(self.vid_to_asr_dict[vid])
        if "webvid" in vid:
            cap_df = cap_df
        else:
            cap_df = cap_df[cap_df['end'] < vlen+3]
        sentences = []
        starts = []
        ends = []

        if self.vis_processor.pretrain and duration != vlen:
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

            if not no_caption_flag:
                for idx in range(start_idx, len(cap_df)):
                    cap_entry = cap_df.iloc[idx]
                    text, s, e = cap_entry['text'], cap_entry['start'], cap_entry['end']
                    if type(text) is list:
                        text = np.random.choice(text)
                    # s, e = round(s), round(e)
                    text = str(text).replace('\n',' ').strip()
                    # if len(text.split()) > 32:
                    #     text = ' '.join(text.split()[0:32])
                    # if len(text.split()) <= 5:
                    #     continue
                    if math.floor(e / duration * self.num_bins) == math.floor(s / duration* self.num_bins):
                        continue
                    if s > end_timestamp:
                        break
                    if e > end_timestamp:
                        e = end_timestamp

                    trim_start = max(s - start_timestamp, 0)
                    trim_end = min(e - start_timestamp, duration)
                    if trim_end == trim_start:
                        break

                    sentences.append(text)
                    starts.append(trim_start)
                    ends.append(trim_end)

            # if len(sentences) == 0 or no_caption_flag:  # handle unlucky sampling
            #     text = '</s>'
            #     sentences.append(text)
            #     starts.append(0)
            #     ends.append(duration)
            #     if no_caption_flag:
            #         start_timestamp = 0
            #         end_timestamp = duration
        if len(sentences) == 0 and "webvid" not in vid:
            start_timestamp = 0
            end_timestamp = vlen
            for idx in range(len(cap_df)):
                cap_entry = cap_df.iloc[idx]
                text, s, e = cap_entry['text'], cap_entry['start'], cap_entry['end']
                if type(text) is list:
                    text = np.random.choice(text)
                # s, e = round(s), round(e)
                text = str(text).replace('\n',' ').strip()
                if e > end_timestamp:
                    e = end_timestamp
                if s > end_timestamp:
                    s = end_timestamp
                if e - s < 1:
                    continue
                sentences.append(text)
                starts.append(s)
                ends.append(e)
        else:
            # print(cap_df)
            start_timestamp = 0
            end_timestamp = vlen
            for idx in range(len(cap_df)):
                cap_entry = cap_df.iloc[idx]
                text, s, e = cap_entry['text'], cap_entry['start'], cap_entry['end']
                if type(text) is list:
                    text = np.random.choice(text)
                text = str(text).replace('\n',' ').strip()
                sentences.append(text)
                starts.append(s)
                ends.append(e)


        return {'text': sentences, 'start': starts, 'end': ends}, \
                (start_timestamp, end_timestamp)