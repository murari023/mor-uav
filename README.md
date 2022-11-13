# MOR-UAV
This repository contains a Keras implementation of the paper 'MOR-UAV: A Benchmark Dataset and Baselines for Moving Object Recognition in UAV Videos' published in ACM MM 2020. A large-scale video dataset is presented for moving object recognition (MOR) in aerial videos. A baseline model along with a framework to perfrom end-to-end MOR is also provided. The source code for training, testing and evaluation methods is provided.


We used the code base of [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) for our work. We forked off from [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) tag 0.5.1.

# Description
Given a real-time UAV video stream, how can we both localize and classify the moving objects, i.e. perform moving object recognition (MOR)? The MOR is one of the essential tasks to support various UAV vision-based applications including aerial surveillance, search and rescue, event recognition, urban and rural scene. We introduce MOR-UAV, a large-scale video dataset for MOR in aerial videos. We achieve this by labeling axis-aligned bounding boxes for moving objects which requires less computational resources than producing pixel-level estimates. We annotate 89,783 moving object instances collected from 30 UAV videos, consisting of 10,948 frames in various scenarios such as weather conditions, occlusion, changing flying altitude and multiple camera views. We assigned the labels for two categories of vehicles (car and heavy vehicle). Furthermore, we propose a deep unified framework MOR-UAVNet for MOR in UAV videos. Since, this is a first attempt for MOR in UAV videos, we present 16 baseline results based on the proposed framework over the MOR-UAV dataset through quantitative and qualitative experiments. We also analyze the motion-salient regions in the network through multiple layer visualizations. The MOR-UAVNet works online at inference as it requires only few past frames. Moreover, it doesn't require predefined target initialization from user. Experiments also demonstrate that the MOR-UAV dataset is quite challenging.

# Paper
[MOR-UAV: A Benchmark Dataset and Baselines for Moving Object Recognition in UAV Videos](https://dl.acm.org/doi/10.1145/3394171.3413934)

# BibTex
@inproceedings{mandal2020mor,title={Mor-uav: A benchmark dataset and baselines for moving object recognition in uav videos},author={Mandal, Murari and Kumar, Lav Kush and Vipparthi, Santosh Kumar},booktitle={Proceedings of the 28th ACM International Conference on Multimedia},pages={2626--2635},year={2020}}

# Dataset-Link
[MOR-UAV](https://drive.google.com/file/d/1z6kvIpoRTGTYXe3AG8z2Sik2ut-ApDbM/view?usp=sharing)

# Installation
1. Clone this repository.
2. Ensure numpy is installed using pip install numpy --user
3. In the repository, execute pip install . --user. Note that due to inconsistencies with how tensorflow should be installed, this package does not define a dependency on tensorflow as it will try to install that (which at least on Arch Linux results in an incorrect installation). Please make sure tensorflow is installed as per your systems requirements.
4. Run python setup.py build_ext --inplace to compile Cython code first.

# Training
MOR_UAVNet can be trained using this train.py script. Note that the train script uses relative imports since it is inside the keras_retinanet package.

Model can be trained on all csv files in --csv_path CSV_PATH folder. Randomly select a video, read continuous frames (1-10)10 (if depth of history frame is 10) frames, calculate temporal median and TDR block features and perform estimation using resnet 50 and retinanet pyramid. Next choose 11th frame with 10 previous history frames ([1:11] ) and perform estimation. At end of each video, randomly select new video, start with depth frames.

## CSV datasets
The CSV file with annotations should contain one annotation per line. Images with multiple bounding boxes should use one row per bounding box. Note that indexing for pixel values starts at 0. The expected format of each line is:
path/to/image.jpg,x1,y1,x2,y2,class_name

```shell
path/to/image.jpg,x1,y1,x2,y2,class_name
```

By default the CSV generator will look for images in csv_folder/train/ directory of the annotations (csv) file.

## Class mapping format
The class name to ID mapping file should contain one mapping per line. Each line should use the following format:
```
class_name,id
```
Indexing for classes starts at 0. Do not include a background class as it is implicit.

## Pretrained models
Model summary can be found in model_summary.md.
**Snapshot and model of trained MOR-UAVNet model is attached in snapshots folder.**

## Usage
```shell

keras_retinanet/bin/train.py csv classes.csv

Positional arguments:
  {csv}                 Arguments for csv dataset types.

Optional arguments:     
    --backbone BACKBONE   Backbone model used by retinanet. (default : resnet50) 
                one of the backbones in resnet models (resnet50, resnet101, resnet152)    
    --epochs EPOCHS       Number of epochs to train.     
    --steps STEPS         Number of steps per epoch.        (default: 10000)         
    --lr LR               Learning rate.                    (default: 1e-5)               
    --snapshot-path SNAPSHOT_PATH      
                          Path to store snapshots of models during training    
                          (defaults to './snapshots')    
    --depth DEPTH         Image frame depth.                (default: 10)    
    --initial_epoch INITIAL_EPOCH                           (default: 1)    
                          initial epochs to train.    
    --csv_path CSV_PATH                                     (default: './csv_folder/train/')    
                Path to store csv files for training

```

## Converting a training model to inference model
The training procedure of MOR-UAVNet works with training models. These are stripped down versions compared to the inference model and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model.

## Usage
```shell

keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_60.h5 snapshots/model.h5 

positional arguments:
  model_in              The model to convert.
  model_out             Path to save the converted model to.
optional arguments:
  --backbone BACKBONE   The backbone of the model to convert.
```

# Testing
Use the trained model to test the unseen videos.
Model can be tested on all csv files in --csv_path CSV_PATH. And output is stored in  `--save-path SAVE_PATH` folder.

## Usage

```shell

keras_retinanet/bin/evaluate.py csv classes.csv   snapshots/model.h5

positional arguments:
  {csv}                 Arguments for specific dataset types.
  model                 Path to RetinaNet model.

optional arguments:
  --depth DEPTH         Image frame depth.
  --csv_path CSV_PATH   Path to store csv files for training
              (default: './csv_folder/test/')
  --iou-threshold IOU_THRESHOLD
                        IoU Threshold to count for a positive detection
                        (defaults to 0.5).
  --save-path SAVE_PATH
                        Path for saving images with detections


```

By default the CSV generator will look for images in `visualize/test/` directory of the annotations (csv) file.
Evaluation output is stored in `outputs/FOLDERNAME/`.

# Layer Visualization
MotionRec layer can be visualized using `keras_retinanet/bin/visualize.py`

## Usage

```shell

keras_retinanet/bin/visualize.py csv classes.csv   snapshots/model.h5

positional arguments:
  {csv}                 Arguments for specific dataset types.
  model                 Path to RetinaNet model.

Optional arguments:
  --depth DEPTH           Image frame depth.
  --csv_path CSV_PATH     Path to store csv files for layer visualization
  --save-path SAVE_PATH
                          Path for saving images with visualization
  --layer LAYER           Name of the CNN layer to visualize.
  --layer_size LAYER_SIZE  CNN layer size
   --score-threshold SCORE_THRESHOLD
                        Threshold on score to filter detections with (defaults
                        to 0.05).
  --iou-threshold IOU_THRESHOLD
                        IoU Threshold to count for a positive detection
                        (defaults to 0.5).

```

By default the CSV generator will look for images in `visualize/backdoor/csv/` directory of the annotations (csv) file.
Visualization is stored in `visualize/backdoor/LAYER_NAME/output/`.


