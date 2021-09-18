


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
        self.data_dict = {
            "image_id": [],
            "score": [],
            "x": [],
            "y": [],
            "width": [],
            "height": [],
        }

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if "instances" in output:
                instances= output["instances"].to(self._cpu_device)
                record = {}
                record["image_id"]  = input['image_id']
                record["file_name"] = input['file_name']
                # record["instances"] = {"pred_boxes":    instances.pred_boxes, 
                #                        "scores":        instances.scores, 
                #                        "pred_classes":  instances.pred_classes,
                #                        "overlap_boxes": instances.overlap_boxes,
                #                        "overlap_probs": instances.overlap_probs, }
                record["instances"] = {"pred_boxes":    instances.pred_boxes, 
                                       "scores":        instances.scores, 
                                       "pred_classes":  instances.pred_classes}
                # print("input={}".format(input))
                # print("instances={}".format(instances))
                self._predictions.append(record)

                for box_i, score_i in zip(instances.pred_boxes, instances.scores):
                    score_i = score_i.cpu().item()
                    if score_i < .5:
                        continue
                    x1, y1, x2, y2 = list(map(int, box_i.cpu().tolist()))
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    self.data_dict["image_id"].append( record["image_id"] - 1 )
                    self.data_dict["score"].append( score_i )
                    self.data_dict["x"].append(x)
                    self.data_dict["y"].append(y)
                    self.data_dict["width"].append(w)
                    self.data_dict["height"].append(h)


    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            # self._predictions = list(itertools.chain(*self._predictions))


            if not comm.is_main_process():
                return {}

        output_file = os.path.join(self._output_dir, "output.pth")
        torch.save(self._predictions, output_file)

        
        df = pd.DataFrame(self.data_dict)

        df_path = os.path.join(self._output_dir, "submit_50.csv")
        df.to_csv(df_path, index=False)

        df75 = df[ df['score'] > .75 ]
        df_path = os.path.join(self._output_dir, "submit_75.csv")
        df75.to_csv(df_path, index=False)

        results = None
        return results
    
    def coco_evaluate(self):
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