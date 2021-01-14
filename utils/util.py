import os
import math
import numpy as np
from scipy.interpolate import splrep, splev, InterpolatedUnivariateSpline

np.seterr(divide='ignore', invalid='ignore')

ROTATION_NUMBERS = {'q': 4, '6d': 6, 'eular': 3}

def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def make_dataset(dir_list, phase, data_split=3, sort=False, sort_index=1):
    images = []
    for dataroot in dir_list:
        _images = []
        image_filter = []

        assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
        for root, _, fnames in sorted(os.walk(dataroot)):
            for fname in fnames:
                if phase in fname:
                    path = os.path.join(root, fname)
                    _images.append(path)
        if sort:
            _images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[sort_index]))
        if data_split is not None:
            for i in range(int(len(_images)/data_split - 1)):
                image_filter.append(_images[data_split*i])
            images += image_filter
            return images
        else:
            return _images

def mkdir(folder):
    if os.path.exists(folder):
        return 1
    else:
        os.makedirs(folder)


def normalize_data(orig_data):
    data_mean = np.mean(orig_data, axis=0)
    data_std = np.std(orig_data, axis=0)
    normalized_data = np.divide((orig_data - data_mean), data_std)
    normalized_data[normalized_data != normalized_data] = 0
    return normalized_data, data_mean, data_std


def umnormalize_data(normalized_data, data_mean, data_std):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # Dimensionality

    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(normalized_data, stdMat) + meanMat
    return orig_data


def interp_pose(pose_array, confidences, k=2):
    frame_number, joint_number = confidences.shape[0], confidences.shape[1]
    pose_completed = np.zeros_like(pose_array)
    confi_completed = np.zeros_like(confidences)
    for joint_index in range(joint_number):
        x_loc = pose_array[:, joint_index*2]
        y_loc = pose_array[:, joint_index*2 + 1]
        joint_confi = confidences[:, joint_index]
        select_indics = (joint_confi >= 0.4)
        x_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], x_loc[select_indics], k=2)
        y_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], y_loc[select_indics], k=2)
        c_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], joint_confi[select_indics], k=2)
        pose_completed[:, joint_index*2] = x_spl(np.arange(0, frame_number))
        pose_completed[:, joint_index*2 + 1] = y_spl(np.arange(0, frame_number))
        confi_completed[:, joint_index] = c_spl(np.arange(0, frame_number))
    return pose_completed, confi_completed