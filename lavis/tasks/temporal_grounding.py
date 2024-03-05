"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import re
import math

from lavis.common.dist_utils import main_process
from lavis.common.utils import get_abs_path
from lavis.common.registry import registry
from lavis.common.tg_eval import eval_submission
from lavis.common.tg_utils import load_jsonl
from lavis.tasks.base_task import BaseTask


@registry.register_task("temporal_grounding")
class TemporalGroundingTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, eval_file, num_bins, abs_time=False, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.abs_time = abs_time
        self.num_bins = num_bins

        self.report_metric = report_metric

        self.gt = None
        if not os.path.isabs(eval_file):
            self.eval_file = get_abs_path(eval_file)
        else:
            self.eval_file = eval_file

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        eval_file = run_cfg.eval_file
        num_bins = run_cfg.num_bins
        abs_time = run_cfg.get("abs_time", False)

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            eval_file=eval_file,
            num_bins=num_bins,
            abs_time=abs_time,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        query_ids = samples["qid"]
        splits = samples["split"]
        starts = samples["clip_start"]
        ends = samples["clip_end"]
        for caps, qid, split, start, end in zip(captions, query_ids, splits, starts, ends):
            windows = []
            start, end = float(start), float(end)
            for caption in caps:
                if "<" in caption:
                    pred_window = re.findall(r'<(.*?)>', caption)
                else:
                    pred_window = re.findall(r'\((.*?)\)', caption)
                    if pred_window:
                        pred_window = pred_window[0].split(",")
                if self.abs_time:
                    interval = (end - start) / self.num_bins
                    s = start
                else:
                    interval = 1
                    s = 0
                try:
                    if len(pred_window)==2:
                        pred_window = [int(pred_window[0])*interval+s, int(pred_window[1])*interval+s, 1.0]
                        windows.append(pred_window)
                    elif len(pred_window) == 1:
                        pred_window = [int(pred_window[0])*interval+s, int(pred_window[0])*interval+s+5, 1.0]
                        windows.append(pred_window)
                    elif len(pred_window) > 2:
                        pred_window = [int(pred_window[0])*interval+s, int(pred_window[1])*interval+s, 1.0]
                        windows.append(pred_window)
                except:
                    windows.append([30, 40, 1.0])
            if len(windows) == 0:
                windows.append([30, 40, 1.0])

            results.append({"qid": int(qid), "pred_relevant_windows": windows, "split": int(split), "clip_start": float(start), "clip_end": float(end)})
        
        return results


    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="qid",
        )
        # metrics = {"agg_metrics": 0.0}
        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name, epoch=epoch
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name, epoch):
        with open(eval_result_file) as f:
            predictions = json.load(f)

        if self.gt is None:
            if self.abs_time:
                self.gt = load_jsonl(self.eval_file)
            else:
                # self.gt = load_jsonl(self.eval_file)
                qid_to_video = {}
                for pred in predictions:
                    qid_to_video[pred["qid"]] = [pred["clip_start"], pred["clip_end"]]
                self.gt = load_jsonl(self.eval_file)
                for query in self.gt:
                    relevant_windows = query["relevant_windows"]
                    qid = query["qid"]
                    clip_start, clip_end = qid_to_video[qid]
                    duration = min(clip_end - clip_start, query["duration"])
                    # duration = query["duration"]
                    new_windows = []
                    for window in relevant_windows:
                        start, end = window
                        start = max(0, start - clip_start)
                        end = min(end - clip_start, duration)
                        rel_start = math.floor(start * self.num_bins / duration)
                        rel_end = math.floor(end * self.num_bins / duration)
                        if rel_end == self.num_bins:
                            rel_end -= 1
                        new_windows.append([rel_start, rel_end])
                    query["relevant_windows"] = new_windows

        results = eval_submission(predictions, self.gt, verbose=False)
        results["agg_metrics"] = results["brief"]["MR-full-R1@0.5"]
        final_result_file = os.path.join(registry.get_path("result_dir"), "metric_epoch{}.json".format(epoch))
        json.dump(results, open(final_result_file, "w"), indent=4)
        return results
