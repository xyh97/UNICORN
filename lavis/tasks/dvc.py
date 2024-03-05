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
from lavis.common.dvc_eval import evaluate_para
from lavis.common.tg_eval import eval_submission
from lavis.common.tg_utils import load_jsonl
from lavis.tasks.base_task import BaseTask


@registry.register_task("dvc")
class DVCTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, eval_file, num_bins, abs_time=False, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.abs_time = abs_time
        self.num_bins = num_bins

        self.report_metric = report_metric
        self.eval_file = []
        for f in eval_file:
            if not os.path.isabs(f):
                self.eval_file.append(get_abs_path(f))
            else:
                self.eval_file.append(f)

        self.gt = None

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        eval_file = run_cfg.eval_file
        num_bins = run_cfg.num_bins
        abs_time = run_cfg.get("abs_time", True)

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

        vids = samples["vid"]
        durations = samples["duration"]
        for (caption, vid, d) in zip(captions, vids, durations):
            results.append({"vid": vid, "caption": caption, "duration": float(d)})
        
        return results


    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="vid",
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
        
        # youcook2
        predicted_captions = []
        gt_captions = []
        keys = []
        new_keys = []

        val_youcook = json.load(open(self.eval_file[0]))

        for k, v in val_youcook.items():
            keys.append(k)

        for pred in predictions:
            if 'youcook2' not in pred['vid']:
                continue
            if type(pred['caption']) is list:
                dc = pred['caption'][0]
            else:
                dc = pred['caption']
            key = pred['vid']
            caps = []
            for c in dc.split('.'):
                q = c.strip()
                if q == "":
                    continue
                if q not in caps:
                    caps.append(q)
            new_dc = ' '.join(caps)
            if key in keys:
                predicted_captions.append(new_dc)
                new_keys.append(key)

        for k in new_keys:
            v = val_youcook[k]
            gt_caption = ' '.join(v['text'])
            gt_captions.append([gt_caption])

        results_youcook = evaluate_para(predicted_captions, gt_captions)

        final_result_file = os.path.join(registry.get_path("result_dir"), "metric_epoch{}_youcook.json".format(epoch))
        json.dump(results_youcook, open(final_result_file, "w"), indent=4)

        # activitynet
        predicted_captions = []
        gt_captions = []
        keys = []
        new_keys = []
        val1_activity = json.load(open(self.eval_file[1]))
        val2_activity = json.load(open(self.eval_file[2]))

        for k, v in val1_activity.items():
            keys.append(k)

        for k, v in val2_activity.items():
            keys.append(k)

        keys = list(set(keys))

        for pred in predictions:
            if 'activitynet' not in pred['vid']:
                continue
            if type(pred['caption']) is list:
                dc = pred['caption'][0]
            else:
                dc = pred['caption']
            key = pred['vid'].split("/")[-1].split(".")[0]
            caps = []
            for c in dc.split('.'):
                q = c.strip()
                if q == "":
                    continue
                if q not in caps:
                    caps.append(q)
            new_dc = ' '.join(caps)
            if key in keys:
                predicted_captions.append(new_dc)
                new_keys.append(key)
        
        for k in new_keys:
            x = []
            if k in val1_activity:
                x.append(val1_activity[k])
            if k in val2_activity:
                x.append(val2_activity[k])
            gt_captions.append(x)
        
        results_activitynet = evaluate_para(predicted_captions, gt_captions)

        results_activitynet["agg_metrics"] = results_activitynet["Para_CIDER"]
        final_result_file = os.path.join(registry.get_path("result_dir"), "metric_epoch{}_activitynet.json".format(epoch))
        json.dump(results_activitynet, open(final_result_file, "w"), indent=4)

        # qvhighlights
        if len(self.eval_file) == 4:
            preds_qv = []
            for pred in predictions:
                if 'qvhighlights' not in pred['vid']:
                    continue
                windows = []
                qid = pred['vid'].split("+")[1]
                caps = pred['caption']
                duration = pred['duration']
                for caption in caps:
                    pred_window = re.findall(r'<(.*?)>', caption)
                    interval = duration / self.num_bins
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

                preds_qv.append({"qid": int(qid), "pred_relevant_windows": windows})
            val_qv = load_jsonl(self.eval_file[-1])
            results_qv = eval_submission(preds_qv, val_qv, verbose=False)
            final_result_file = os.path.join(registry.get_path("result_dir"), "metric_epoch{}_qvhighlights.json".format(epoch))
            json.dump(results_qv, open(final_result_file, "w"), indent=4)

        return results_activitynet
