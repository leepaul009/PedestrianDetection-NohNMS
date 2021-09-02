


import itertools, os
import torch
import pandas as pd

from detectron2.utils import comm
from .evaluator import DatasetEvaluator


class PedEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self.cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if "instances" in output:
                instances= output["instances"].to(self._cpu_device)
                record = {}
                record["image_id"]  = input['image_id']
                record["file_name"] = input['file_name']
                record["instances"] = {"pred_boxes":    instances.pred_boxes, 
                                       "scores":        instances.scores, 
                                       "pred_classes":  instances.pred_classes,
                                       "overlap_boxes": instances.overlap_boxes,
                                       "overlap_probs": instances.overlap_probs, }
                # print("input={}".format(input))
                # print("instances={}".format(instances))
                self._predictions.append(record)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            # self._predictions = list(itertools.chain(*self._predictions))


            if not comm.is_main_process():
                return {}

        output_file = os.path.join(self._output_dir, "output.pth")
        torch.save(self._predictions, output_file)
        
        results = None
        return results