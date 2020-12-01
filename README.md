# MotioNet

This library provides the source code of [*MotioNet*](http://rubbly.cn/publications/motioNet), [Transaction on Graphics (ToG) 2020], a kinematic deep neural network that reconstructs 3D skeletal human motion from monocular video. The network designed based on the common motion representation, and its direct output can be converted to bvh file without any post-processing step.

**MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency** : [Project](https://rubbly.cn/publications/motioNet/) | [Paper](https://arxiv.org/abs/2006.12075) |[Video](https://www.youtube.com/watch?v=8YubchlzvFA)

<img src="https://rubbly.cn/publications/motioNet/cover.gif" align="center">



## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- 2D Pose Detection Tool (for evaluation on wild videos, now support: [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose))

Based on your Cuda version, choose a suitable version of [PyTorch](https://pytorch.org/get-started/locally/) and install it, then run this command to install other packages:

```shell
pip install -r requirements.txt
```

## Quick Start
We provide a pretrained model and a few demo examples, that demonstrate how our framework works. In order to run the demo, download the 1. Pretrained model from [here](https://drive.google.com/drive/folders/19hO4eVV8cE61aVg3dA-hClVjrtiJhq8d) and place it in the **./checkpoints** folder 2. Training data from [here](https://drive.google.com/drive/folders/1mvRPqtsNp46grBQ9feYish8evhEkm_9O) and place it in the **./data** folder. Then simply run

```shell
mkdir output
python evaluate.py -r ./checkpoints/wild_gt_tcc.pth -i demo
```

The resulting bvh files will be found at **./output**. Note that for these examples we already extracted the 2D key-points from the videos using [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) , but if you want to use your own videos this piror step should be performed by yourself. All the original video can be found in our [video](https://www.youtube.com/watch?v=8YubchlzvFA).

### Data preparation

#### Training Data

There are two datasets in full pipeline, one is [Human3.6m](http://vision.imar.ro/human3.6m/description.php) and another one is [CMU Motion Capture](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion) data. The h36m data is necessary for both training and testing, so prepare it firstly.

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

Now there are two available pre-trained models, one is for evaluation on h36m with ground-truth 2d detection and another one is for wild videos using [confidence map, foot contact signal and adversarial training].

After downloading with link [[Google drive(updated!)](https://drive.google.com/drive/folders/19hO4eVV8cE61aVg3dA-hClVjrtiJhq8d?usp=sharing)], then place them into ./checkpoints folder.

More pre-trained model will be released next days.

### Test on h36m

Once you have h36m data and pre-trained model, you can run the script to evaluate our model on h36m.

```shell
# Evaluation only, without contact, translation
python evaluate.py -r ./checkpoints/h36m_gt.pth -i h36m -o [output_folder]

# With translation
python evaluate.py -r ./checkpoints/h36m_gt_t.pth -i h36m -o [output_folder]
```

The results will be placed in a subfolder in *./output* with a special name which is a combination of model parameters, it will be helpful to understand the training details. And two kinds of results will be generated: error.txt and BVH files.

In error.txt, you can find same performance like what we show - around 53mm with gt detected(52mm when the channel parameter is 2048, but it causes waste of model size and training time). And 4 randomly selected bvh file will also be stored in here, you can use Blender to see the visual performance.

The translation will be observed clearly in action like Walking or WalkingDog. 

(NOTICE: The training data we use is in camera space, so the orientation of predicted pose will be strange when you observe it in Blender World Space, you need to rotate the user view or skeleton object in Blender)

### Test on wild videos

Once you have h36m data, 2d detected data and pre-trained model, you can run the script to evaluate our model on wild videos. Now we provide the interface between Openpose and our network, the supporting of CPN and Detectron will coming soon.

After running Openpose, all 2d pose files will be stored in one folder, then you can use these folder path as input parameter to run the python script, the results will also be in a subfolder in *./output* with bvh format.

Now we **haven't apply any smoothing process** on the 2d **input** and **output**, you can do it by yourself to get better results like we show in the video, but here we want to show the original production.

```shell
python evaluate.py -r ./checkpoints/wild_gt_tcc.pth -i [openpose_results_folder]
```

## Train from scratch

We provide instructions for retraining our models

There are three kinds parameters for training: Runtime parameters, Network definition and Training parameters, you can find more details in *train.py*. We also provide a default configuration of network definition and trainer, you can find it in *./config_zoo/default.json*. Before your training, you should change the ***save_dir*** in this file to a suitable path.

If you want to reproduce the results of our pretrained models, run the following commands:

```shell
# h36m evaluation, without translation and contact
python train.py -n h36m --kernel_size 5,3,1 --stride 1,1,1 --dilation 1,1,1 --channel 1024 

# h36m, with translation
python train.py -n h36m --kernel_size 5,3,1 --stride 1,1,1 --dilation 1,1,1 --channel 1024 --translation 1

# wild videos
python train.py -n wild -d 1 --kernel_size 5,3,1 --stride 3,1,1 --dilation 1,1,1 --channel 1024 --confidence 1 --translation 1 --contact 1 --loss_term 1101
```
### Training visualization:

We use [tensorboardX](https://github.com/lanpa/tensorboardX) to visualize the losses when training. In the training, you can go into checkpoint folder and run this command:

```
tensorboard --logdir=./

Print:
TensorBoard 1.15.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then you can visit this link in your machine [http://localhost:6006/](http://localhost:6006/) to check the visualzation.

![vis_example](https://user-images.githubusercontent.com/7709951/100384574-6e8fb100-305b-11eb-9900-aa7b50be6ac0.png)

### Training tips:

There are few things you can do to get different model preferences:

- We haven't use any foot contact dataset to train it, we just extract it by average foot height and velocity in h36m dataset. And in this [contact detection function](https://github.com/Shimingyi/MotioNet/blob/master/data/h36m_dataset.py#L148), there is a magic number 20, and it can be defined by you self. If the number is bigger, more frames can be thought as ‘contacting'. It will enhence the attraction from floor, so the results will be worse if there are so many foot action.
- We trained our model on clips with variance frame number, so you can put all the frames of test video to the network without cutting.

## Limitations

- Moving camera: The model is trained on h36m dataset which includes 4 cameras, and what the network predict is in camera space. The reason is that, if we predict the pose in world space: (different 3d pose ⊕ different camera view) = same 2d projection, then if given a same 2d pose, these will be an ambiguity for predicted 3d pose. When the camera is moving, the situation is same like it, sometimes looks make sense, but not correct.
- The dependence on 2d detection: The input is 2d detection from other implement, although the confidence map is used as a adapter, the results will be influenced when 2d detection crushed, it's mostly appeared in fast motion or occlusion cases.
- Animation: Explained in paper, the loss is not applied on rotation directly so the absolutely value of rotation is wired although look fine in positional view. This problem will be solved in next version.

## Acknowledgments

The code of forward kinematics layer is from [cvpr2018nkn](https://github.com/rubenvillegas/cvpr2018nkn) in tensorflow, we re-write a pytorch version in our code. 

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

