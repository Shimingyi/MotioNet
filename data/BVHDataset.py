#!/usr/bin/python
# -*- coding:utf-8 -*-

import h5py
import torch
import numpy as np
import utils.BVH as BVH
import utils.Animation as Animation

from utils.Quaternions import Quaternions
from utils import h36m_utils
from utils import util
from torch.utils.data import Dataset

ROTATION_NUMBERS = {'q': 4, '6d': 6, 'euler': 3}

class BVHDataset(Dataset):
    def __init__(self, config, is_train=True):

        poses_3d_root, rotations, bones, alphas, contacts, projections = [], [], [], [], [], []
        self.frames = []
        self.config = config
        self.rotation_number = ROTATION_NUMBERS.get(config.arch.rotation_type)

        datasets = ['bvh']#, 'bvh']
        if 'h36m' in datasets:
            dim_to_use_3d = h36m_utils.dimension_reducer(3, config.arch.predict_joints)
            subjects = h36m_utils.TRAIN_SUBJECTS if is_train else h36m_utils.TEST_SUBJECTS
            actions = h36m_utils.define_actions('All')
            self.cameras = h36m_utils.load_cameras(config.trainer.data_path)
            for subject in subjects:
                for action in actions:
                    for subaction in range(1, 3):
                        data_file = h5py.File('%s/S%s/%s-%s/annot.h5' % (config.trainer.data_path, subject, action, subaction), 'r')
                        data_size = data_file['frame'].size / 4
                        data_set = np.array(data_file['pose/3d']).reshape((-1, 96))[:, dim_to_use_3d]
                        for i in range(4):
                            camera_name = data_file['camera'][int(data_size * i)]
                            R, T, f, c, k, p, res_w, res_h = self.cameras[(subject, str(camera_name))]
                            set_3d = data_set[int(data_size * i):int(data_size * (i + 1))].copy()
                            set_3d_world = h36m_utils.camera_to_world_frame(set_3d.reshape((-1, 3)), R, T)
                            # set_3d_world[:, [1, 2]] = set_3d_world[:, [2, 1]]
                            # set_3d_world[:, [2]] *= -1
                            # set_3d_world = set_3d_world.reshape((-1, config.arch.predict_joints * 3))
                            set_3d_root = set_3d_world - np.tile(set_3d_world[:, :3], [1, int(set_3d_world.shape[-1] / 3)])

                            set_bones = self.get_bones(set_3d_root, config.arch.predict_joints)
                            set_alphas = np.mean(set_bones, axis=1)

                            self.frames.append(set_3d_root.shape[0])
                            poses_3d_root.append(set_3d_root/np.expand_dims(set_alphas, axis=-1))
                            rotations.append(np.zeros((set_3d_root.shape[0], int(set_3d_root.shape[1]/3*self.rotation_number))))
                            bones.append(set_bones/np.expand_dims(set_alphas, axis=-1))
                            alphas.append(set_alphas)
                            contacts.append(self.get_contact(set_3d_world, config.arch.predict_joints))
                            projections.append((set_3d_world.copy() / np.expand_dims(set_alphas, axis=-1)).reshape((set_3d_world.shape[0], -1, 3))[:, 0, 2])

        if 'bvh' in datasets:
            to_keep = [0, 7, 8, 9, 2, 3, 4, 12, 15, 18, 19, 20, 25, 26, 27] if config.arch.predict_joints == 15 else [0, 7, 8, 9, 2, 3, 4, 12, 13, 15, 16, 18, 19, 20, 25, 26, 27]
            parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 9, 10, 7, 12, 13] if config.arch.predict_joints == 15 else [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

            bvh_files = util.make_dataset(['/mnt/dataset/test_bvh'], phase='bvh', data_split=1)
            bvh_files = bvh_files[:int(len(bvh_files)*0.8)] if is_train else bvh_files[int(len(bvh_files)*0.8):]
            for bvh_file in bvh_files:
                original_anim, joint_names, frame_rate = BVH.load(bvh_file)
                set_skel_in = original_anim.positions[:, to_keep, :]
                set_rotations = original_anim.rotations.qs[:, to_keep, :]
                anim = Animation.Animation(Quaternions(set_rotations), set_skel_in, original_anim.orients.qs[to_keep, :], set_skel_in, np.array(parents))
                set_3d_world = Animation.positions_global(anim).reshape(set_rotations.shape[0], -1)
                set_3d_world[:, 0:3] = (set_3d_world[:, 3:6] + set_3d_world[:, 12:15])/2
                set_3d_root = set_3d_world - np.tile(set_3d_world[:, :3], [1, int(set_3d_world.shape[-1] / 3)])

                set_bones = self.get_bones(set_3d_root, config.arch.predict_joints)
                set_alphas = np.mean(set_bones, axis=1)

                self.frames.append(set_3d_root.shape[0])
                poses_3d_root.append(set_3d_root / np.expand_dims(set_alphas, axis=-1))
                rotations.append(np.zeros((set_3d_root.shape[0], int(set_3d_root.shape[1] / 3 * self.rotation_number))))
                bones.append(set_bones / np.expand_dims(set_alphas, axis=-1))
                alphas.append(set_alphas)
                contacts.append(self.get_contact(set_3d_world, config.arch.predict_joints))
                projections.append((set_3d_world.copy() / np.expand_dims(set_alphas, axis=-1)).reshape((set_3d_world.shape[0], -1, 3))[:, 0, 2])

        self.poses_3d = np.concatenate(poses_3d_root, axis=0)
        self.rotations = np.concatenate(rotations, axis=0)
        self.bones = np.concatenate(bones, axis=0)
        self.alphas = np.concatenate(alphas, axis=0)
        self.contacts = np.concatenate(contacts, axis=0)
        self.projections = np.concatenate(projections, axis=0)

        posed_3d_flip = self.get_flipping(self.poses_3d, 3, config.arch.predict_joints)
        if config.trainer.data_aug_flip and is_train:
            self.poses_3d = np.concatenate([self.poses_3d, posed_3d_flip], axis=0)

        self.poses_2d = self.get_projection(self.poses_3d)
        self.poses_2d_root = (self.poses_2d - self.poses_2d[:, 0, None]).reshape((self.poses_3d.shape[0], -1))

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from utils import visualization
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2)
        for i in range(1):
            ax1 = plt.subplot(gs[0], projection='3d')
            visualization.show3Dpose(self.poses_3d[i], ax1, radius=5)

            ax2 = plt.subplot(gs[1])
            visualization.show2Dpose(self.poses_2d_root[i]*1000+500, ax2, radius=1000)

            fig.savefig('./images/2d_3d/_%d.png' % i)
            fig.clear()

        self.update_sequence_index()

    def __getitem__(self, index):
        items_index = self.sequence_index[index]
        file_path = self.images_path[items_index].tolist()
        pose_pixel = self.poses_2d_pixelwise[items_index]
        pose_3d_root = self.poses_3d_root[items_index]
        alpha = self.alphas[items_index]
        if self.standardization:
            length = self.length_normalized[items_index]
            quaternions = self.quaternions_normalized[items_index]
            projection = self.projection_factors_normalized[items_index]
            pose_2d_root = self.poses_2d_root_normalized[items_index]
            pose_2d_root_c = self.poses_2d_noised_normalized_with_confidence[items_index]
        else:
            length = self.lengths[items_index]
            quaternions = self.quaternions[items_index]
            projection = self.projection_factors[items_index]
            pose_2d_root = self.poses_2d_root[items_index]
            pose_2d_root_c = self.poses_2d_noised_with_confidence[items_index]
        return file_path, torch.from_numpy(pose_pixel).float(), \
               torch.from_numpy(pose_2d_root).float(), torch.from_numpy(pose_2d_root_c).float(), \
               torch.from_numpy(pose_3d_root).float(), torch.from_numpy(quaternions).float(), \
               torch.from_numpy(projection).float(), torch.from_numpy(length).float(), torch.from_numpy(alpha).float()

    def __len__(self):
        return len(self.sequence_index)

    def get_bones(self, position_3d, predict_joints):
        def distance(position1, position2):
            return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))
        if predict_joints == 15:
            length = np.zeros((position_3d.shape[0], 8))
        else:
            length = np.zeros((position_3d.shape[0], 10))
        length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
        length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
        length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)

        if predict_joints == 15:
            length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
            length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
            length[:, 5] = ((distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 12:3 * 12 + 3]) + distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 9:3 * 9 + 3])) / 2)
            length[:, 6] = ((distance(position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3]) + distance(position_3d[:, 3 * 10:3 * 10 + 3], position_3d[:, 3 * 9:3 * 9 + 3])) / 2)
            length[:, 7] = ((distance(position_3d[:, 3 * 13:3 * 13 + 3], position_3d[:, 3 * 14:3 * 14 + 3]) + distance(position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 10:3 * 10 + 3])) / 2)
        else:
            length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
            length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
            length[:, 5] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
            length[:, 6] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
            length[:, 7] = ((distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 11:3 * 11 + 3]) + distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
            length[:, 8] = ((distance(position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3]) + distance(position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 12:3 * 12 + 3])) / 2)
            length[:, 9] = ((distance(position_3d[:, 3 * 15:3 * 15 + 3], position_3d[:, 3 * 16:3 * 16 + 3]) + distance(position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3])) / 2)
        return length

    def get_contact(self, poses, predict_joints):
        contact_signal = np.zeros((poses.shape[0], 2))
        poses_reshape = poses.reshape((-1, predict_joints, 3))
        left_z = poses_reshape[:, 3, 2]
        right_z = poses_reshape[:, 6, 2]

        contact_signal[left_z <= (np.mean(np.sort(left_z)[:left_z.shape[0]//5])+20), 0] = 1
        contact_signal[right_z <= (np.mean(np.sort(right_z)[:right_z.shape[0]//5])+20), 1] = 1
        left_velocity = np.sqrt(np.sum((poses_reshape[2:, 3] - poses_reshape[:-2, 3])**2, axis=-1))
        right_velocity = np.sqrt(np.sum((poses_reshape[2:, 6] - poses_reshape[:-2, 6])**2, axis=-1))
        contact_signal[1:-1][left_velocity >= 5, 0] = 0
        contact_signal[1:-1][right_velocity >= 5, 1] = 0
        return contact_signal

    def get_flipping(self, poses, dim, predict_joints):
        key_left = [4, 5, 6, 9, 10, 11] if predict_joints == 15 else [4, 5, 6, 11, 12, 13]
        key_right = [1, 2, 3, 12, 13, 14] if predict_joints == 15 else [1, 2, 3, 14, 15, 16]
        poses_reshape = poses.reshape((poses.shape[0], -1, dim))
        poses_reshape[:, :, 0] *= -1
        poses_reshape[:, key_left + key_right] = poses_reshape[:, key_right + key_left]
        poses_reshape = poses_reshape.reshape(poses.shape[0], -1)
        return poses_reshape

    def get_projection(self, poses, cam_height=2, cam_dist=8, canvas_dist=1):
        """
        Given a rooted 3d pose, return the 2d pose in pixelwise with perspective projection
        :param poses: 3d pose input, with shape(N, J*3)
        :param cam_height: camera height
        :param cam_dist: camera distance from the center of character
        :param canvas_dist: canvas distance from the camera
        :return: the projected 2d pose
        """
        positions = poses.reshape((-1, 3))
        cam_pos = np.array([0, cam_height, cam_dist])
        ray = cam_pos - positions
        ray_proj = (canvas_dist / ray[:, [2]]) * ray
        positions_proj = cam_pos + ray_proj
        positions_proj *= -1
        return positions_proj.reshape((poses.shape[0], -1, 3))[..., [0, 1]]

        # positions = poses.reshape((-1, 3))
        # cam_pos = np.array([0, cam_dist, cam_height])
        # ray = cam_pos - positions
        # ray_proj = (canvas_dist / ray[:, [1]])*ray
        # positions_proj = cam_pos + ray_proj
        # positions_proj *= -1
        # return positions_proj.reshape((poses.shape[0], -1, 3))[..., [0, 2]]

    def update_sequence_index(self):
        self.sequence_index = []
        start_index = 0
        import random
        multi_frame = random.randint(15, 50) * 4 if self.multi_frame == 0 else self.multi_frame
        for frames in self.video_frames:
            if frames > self.multi_frame:
                if self.is_train:
                    offset = 5
                    factor = int((frames - multi_frame) // offset)
                    for i in range(factor):
                        start = int(i*offset + start_index)
                        end = int(i*offset + multi_frame + start_index)
                        self.sequence_index.append(list(range(start, end)))
                    self.sequence_index.append(list(range(frames-multi_frame, frames)))
                else:
                    self.sequence_index.append(list(range(start_index, start_index + frames)))
            start_index += frames

    def add_noise(self, pose_array, length_mean, training):
        pose_array = pose_array.copy()
        confidence_map_clone = np.ones(pose_array.shape)
        if training:
            noise_radius = length_mean.mean() / 20
            noise_array = np.zeros(pose_array.shape)
            confidence_map = np.zeros(pose_array.shape)
            for i in range(confidence_map.shape[-1]):
                noises = np.random.normal(-noise_radius/9, noise_radius/20, size=pose_array.shape[0])
                noise_array[:, i] += noises
                confidences = 1 - np.abs(noises) / noise_radius
                confidences[confidences < 0] = 0
                confidence_map[:, i] = confidences
            confidence_map_clone = confidence_map.copy()
            frames_number = noise_array.shape[0]
            joints_number = noise_array.shape[1]
            noise_frames = np.random.choice(frames_number, np.random.randint(low=frames_number/2, high=frames_number, size=1), replace=False)
            for i in list(noise_frames):
                noise_joints = np.random.choice(joints_number, np.random.randint(low=0, high=joints_number, size=1), replace=False)
                pose_array[i, noise_joints] += noise_array[i, noise_joints]
                confidence_map_clone[i, noise_joints] = confidence_map[i, noise_joints]
            delete_frames = np.random.choice(frames_number, np.random.randint(low=frames_number/16, high=frames_number/4, size=1), replace=False)
            for i in list(delete_frames):
                delete_joints = np.random.choice(int(joints_number/2), np.random.randint(low=0, high=joints_number/8, size=1), replace=False)
                delete_joints_index = []
                for joint_index in delete_joints:
                    delete_joints_index.extend([2 * joint_index, 2 * joint_index + 1])
                pose_array[i, delete_joints_index] = 0
                confidence_map_clone[i, delete_joints_index] = 0
            return pose_array, confidence_map_clone
        else:
            return pose_array, confidence_map_clone
