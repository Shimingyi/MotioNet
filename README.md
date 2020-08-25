# MotioNet

This library provides the source code of [*MotioNet*](http://rubbly.cn/publications/motioNet), [ToG 2020], a method to reconstruct human motion from monocular video. Benefited from a kinematic-based network, designed based on the common motion representation, the reconstructed human motion can be converted to bvh file directly without any post-processing.

**MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency** : [Project](https://rubbly.cn/publications/motioNet/) | [Paper](https://arxiv.org/abs/2006.12075) |[Video](https://www.youtube.com/watch?v=8YubchlzvFA)

<img src="https://rubbly.cn/publications/motioNet/cover.gif" align="center">



## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- 2D Pose Detection Tool (for evaluation on wild videos, now support: [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose))

Run this command to install the python packages:

```shell
pip install -r requirements.txt
```

## Quick Start
This project is still under building, now we provide pre-trained model and a script to generate bvh file from h36m dataset with one-step script. The model for wild videos will be released soon.

### Data preparation

#### Training Data

There are two datasets in full pipeline, one is [Human3.6m](http://vision.imar.ro/human3.6m/description.php) and another one is [CMU Motion Capture](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion) data. The h36m data is necessary for both traning and testing, so prepare it firstly.

For h36m, you can follow [h36m-fetch](https://github.com/anibali/h36m-fetch) to download original files, once you get extracted files, you can run our python script to convert it to a numpy matrix:

```shell
python ./data/prepare_h36m.py
```

For CMU data we just selected 310 clips for training, you can find the file list in *data/cmu_file_list* and collect them to one folder and run our script to take out useful rotations data.

```shell
python ./data/prepare_cmu.py
```

All procedures produce the same result, but we recommend you process data by yourself. We also provide link [[Google drive](https://drive.google.com/drive/folders/1mvRPqtsNp46grBQ9feYish8evhEkm_9O?usp=sharing)] to download them directly. And then place them into *./data* folder. 

#### Pre-trained Model

Now there are one pre-trained model which is training on h36m with ground-truth 2d detection. 

After downloading with link  [[Google drive](https://drive.google.com/drive/folders/19hO4eVV8cE61aVg3dA-hClVjrtiJhq8d?usp=sharing)], place them into ./checkpoints folder. 

More pre-trained model will be released next days.

### Test on h36m

Once you have h36m data and pre-trained model, you can run the script to evaluate our model on h36m.

```shell
# Evaluation only, without contact, translation
python evaluate.py -r ./checkpoints/h36m_gt.pth -i h36m -o [output_folder]

# With translation
python evaluate.py -r ./checkpoints/h36m_gt_t.pth -i h36m -o [output_folder]
```

The results will be placed in a subfolder in *output_folder* with a special name which is a combination of model parameters, it will be helpful to understant the training details. And two kinds of results will be generated: error.txt and BVH files. 

In error.txt, you can find same performance like what we show - around 53mm with gt detected(52mm when the channel parameter is 2048, but it causes waste of model size and training time). 4 randomly selected bvh file will also be stored in the folder, you can use Blender to see the visual performance. 

The translation will be observed clearly in action like Walking or WalkingDog. 

(NOTICE: The tranining data we use is in camera space, so the orentation of predicted pose will be strange when you observe it in Blender World Space, you need to rotate the user view or skeleton object in Blender)

## Train from scratch

We provide instructions for retraining our models

There are three kinds paramters for training: Runtime parameters, Network definition and Training parameters, you can find more details in *train.py*. We also provide a default configration of network definition and trainer in *./config_zoo/default.json*. Before your training, you should change the *save_dir* parameter to your path.

If you want to reproduce the results of our pretrained models, run the following commands:

```shell
# h36m evaluation, without translation and contact
python train.py -n h36m --kernel_size 5,3,1 --stride 1,1,1 --dilation 1,1,1 --channel 1024 --lr 0.001
```

## Citation

If you use this code for your research, please cite our papers:

```bittext
@article{shi2020motionet,
  title={MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency},
  author={Shi, Mingyi and Aberman, Kfir and Aristidou, Andreas and Komura, Taku and Lischinski, Dani and Cohen-Or, Daniel and Chen, Baoquan},
  journal={arXiv preprint arXiv:2006.12075},
  year={2020}
}
```

