# -*- coding:utf-8 -*-

import os
import cv2
import math
import json
import numpy as np

from scipy.ndimage.filters import gaussian_filter


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    f.close()
    return json_file


def heatmap2location(heatmap):
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap[0].cpu().float().numpy()
        heatmap = np.transpose(heatmap, (1, 2, 0))
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap[:, :, part]
        map = gaussian_filter(map_ori, sigma=5)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.01))

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        if len(peaks) > 0:
            max = 0
            for index, peak in enumerate(peaks):
                score = map_ori[peak[1], peak[0]]
                current_max_score = map_ori[peaks[max][1], peaks[max][0]]
                if score > current_max_score:
                    max = index
            peaks_with_score = [(peaks[max][0], peaks[max][1], map_ori[peaks[max][1], peaks[max][0]], peak_counter)]
            all_peaks.append(peaks_with_score)
            peak_counter += len(peaks_with_score)
        else:
            all_peaks.append([])
    return all_peaks


def load_joints(folder_path, key=None):
    points_path = []
    for file_path in os.listdir(folder_path):
        if 'keypoints.json' in file_path:
            points_path.append(file_path)
    points_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
    person_number = len(load_json('%s/%s' % (folder_path, points_path[0]))['people'])

    joints_location_frames_persons = []

    print('Loading the poses in the folder....')
    for person_id in range(person_number):
        joints_location_frames = []
        for path in points_path:
            joints_path = '%s/%s' % (folder_path, path)
            joints_location = load_json(joints_path)['people'][person_id]['pose_keypoints_2d']
            joints_location_frames.append(joints_location)
        joints_location_frames_persons.append(joints_location_frames)
    joints_location_frames_persons = np.array(joints_location_frames_persons)[:, :, :51]
    return joints_location_frames_persons


def draw_jointsd(joints, images=None, width=None, height=None, _circle=True, _limb=True, save_folder=None, colors=None, alpha=0.4):
    # Joints: person_number * frame * joints_location
    if colors is None:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    else:
        colors = [[0, 0, 0]]*25
    limbSeq = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [0, 16]]
    stickwidth = 4

    if images is not None:
        assert joints.shape[1] == len(images)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for frame_id in range(joints.shape[1]):
        if images is None:
            canvas = np.zeros(shape=(height, width, 3))
            canvas.fill(255)
        else:
            canvas = cv2.imread(images[frame_id])
        for person_id in range(joints.shape[0]):
            person_joints = joints[person_id, frame_id, :].reshape(-1, 3)
            if _circle:
                for i in range(17):
                    if int(person_joints[i][0]) != 0 and int(person_joints[i][1]) != 0:
                        cv2.circle(canvas, (int(person_joints[i][0]), int(person_joints[i][1])), 4, colors[i], thickness=-1)
            if _limb:
                for i, limb in enumerate(limbSeq):
                    cur_canvas = canvas.copy()
                    point1_index = limb[0]
                    point2_index = limb[1]

                    point1 = (int(person_joints[point1_index][0]), int(person_joints[point1_index][1]))
                    point2 = (int(person_joints[point2_index][0]), int(person_joints[point2_index][1]))

                    if not (point1[0] == 0 and point1[1] == 0) or (point2[0] == 0 and point2[1] == 0):

                        X = [point1[1], point2[1]]
                        Y = [point1[0], point2[0]]
                        mX = np.mean(X)
                        mY = np.mean(Y)

                        # cv2.line()
                        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                        canvas = cv2.addWeighted(canvas, alpha, cur_canvas, 1 - alpha, 0)
        cv2.imwrite('%s/%s.png' % (save_folder, frame_id), canvas)
