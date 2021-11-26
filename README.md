# Update from NOH-NMS: Improving Pedestrian Detection by Nearby Objects Hallucination

## Prepare Datasets
Images in 'images/test' are validation set.
```
./datasets/ped
├── annotations
│   └── dhd_traffic_train.json
│   └── dhd_traffic_val.json
├── images
│   └── train
│   └── test
```

## Installation
```
  cd detectron2
  pip install -e . 
  #or rebuild
  sh build.sh
```

## Quick Start
See [GETTING_STARTED.md](GETTING_STARTED.md) in detectron2

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)

