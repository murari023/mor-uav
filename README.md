# MOR-UAV
This repository contains a Keras implementation of the paper 'MOR-UAV: A Benchmark Dataset and Baselines for Moving Object Recognition in UAV Videos' published in ACM MM 2020. A large-scale video dataset is presented for moving object recognition (MOR) in aerial videos. A baseline model along with a framework to perfrom end-to-end MOR is also provided. The source code for training, testing and evaluation methods is provided.


We used the code base of [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) for our work. We forked off from [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) tag 0.5.1.

# Description
Given a real-time UAV video stream, how can we both localize and classify the moving objects, i.e. perform moving object recognition (MOR)? The MOR is one of the essential tasks to support various UAV vision-based applications including aerial surveillance, search and rescue, event recognition, urban and rural scene. We introduce MOR-UAV, a large-scale video dataset for MOR in aerial videos. We achieve this by labeling axis-aligned bounding boxes for moving objects which requires less computational resources than producing pixel-level estimates. We annotate 89,783 moving object instances collected from 30 UAV videos, consisting of 10,948 frames in various scenarios such as weather conditions, occlusion, changing flying altitude and multiple camera views. We assigned the labels for two categories of vehicles (car and heavy vehicle). Furthermore, we propose a deep unified framework MOR-UAVNet for MOR in UAV videos. Since, this is a first attempt for MOR in UAV videos, we present 16 baseline results based on the proposed framework over the MOR-UAV dataset through quantitative and qualitative experiments. We also analyze the motion-salient regions in the network through multiple layer visualizations. The MOR-UAVNet works online at inference as it requires only few past frames. Moreover, it doesn't require predefined target initialization from user. Experiments also demonstrate that the MOR-UAV dataset is quite challenging.

# Paper
[MOR-UAV: A Benchmark Dataset and Baselines for Moving Object Recognition in UAV Videos](https://dl.acm.org/doi/10.1145/3394171.3413934)

#BibTex
@inproceedings{mandal2020mor,title={Mor-uav: A benchmark dataset and baselines for moving object recognition in uav videos},author={Mandal, Murari and Kumar, Lav Kush and Vipparthi, Santosh Kumar},booktitle={Proceedings of the 28th ACM International Conference on Multimedia},pages={2626--2635},year={2020}}

#Dataset-Link

#Installation
1. Clone this repository.
2. Ensure numpy is installed using pip install numpy --user
3. In the repository, execute pip install . --user. Note that due to inconsistencies with how tensorflow should be installed, this package does not define a dependency on tensorflow as it will try to install that (which at least on Arch Linux results in an incorrect installation). Please make sure tensorflow is installed as per your systems requirements.
4. Run python setup.py build_ext --inplace to compile Cython code first.

#Training
MOR_UAVNet can be trained using this train.py script. Note that the train script uses relative imports since it is inside the keras_retinanet package.

Model can be trained on all csv files in --csv_path CSV_PATH folder. Randomly select a video, read continuous frames (1-10)10 (if depth of history frame is 10) frames, calculate temporal median and TDR block features and perform estimation using resnet 50 and retinanet pyramid. Next choose 11th frame with 10 previous history frames ([1:11] ) and perform estimation. At end of each video, randomly select new video, start with depth frames.

