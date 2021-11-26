# Update from NOH-NMS: Improving Pedestrian Detection by Nearby Objects Hallucination

## 1. Prepare Datasets
annotations should be the format of COCO.
Note: Images in 'images/test' are validation set(sorry for bad naming).
```
./datasets/ped
├── annotations
│   └── dhd_traffic_train.json
│   └── dhd_traffic_val.json
├── images
│   └── train
│   └── test
```

## 2. Installation
### 2.1 setup environment
```
conda create -n torch16-py38 python=3.8
conda activate torch16-py38

# install necessay deps:
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
pip install pycocotools 
or pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
conda install tqdm scipy pandas  -y
conda install -c conda-forge opencv -y
```
### 2.2 go to the path of this code repo, and install detectron2
```
cd {Path to PedestrianDetection-NohNMS}
pip install -e . 
```
### 2.3 (optial) install jupyter
```
conda install -c conda-forge notebook
or pip install notebook
```

## 3 Train and inference
### 3.1 Download pre-trained model:
you can download model from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md. or directlt download model from: https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl
And the loader of model will ignore the un-existed weight/bias or dimension-not-matched weight/bias for fast-rcnn cascade.

### 3.2 Train:
```
python tools/train_net.py    --num-gpus 4   --resume   --config-file configs/Ped/cascade.yaml   SOLVER.IMS_PER_BATCH 4    SOLVER.BASE_LR 0.02    SOLVER.CHECKPOINT_PERIOD 3464   TEST.EVAL_START 3464   TEST.EVAL_PERIOD 3464   MODEL.WEIGHTS  model_final_480dd8.pkl  OUTPUT_DIR "Experiments/noh_nms_cascade"
```
Under the directory 'Experiments/noh_nms_cascade/inference/', you can see the evaluation result of each evaluation(its duration is defined by TEST.EVAL_PERIOD).
For example, 'crowdhuman_evaluate_results_1.json' is the result first evaluation with COCO format. Besides, I have update the code to make sure all of the images in the evaluation set will be evaluated, otherwise its performace is unbelieveable!
### 3.3 Inference:

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

