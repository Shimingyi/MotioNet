# -*- coding:utf-8 -*-
import copy
import random
import numpy as np

from utils.Quaternions import Quaternions
from torch.utils.data import Dataset
from utils import h36m_utils, util


class h36m_dataset(Dataset):
    def __init__(self, config, is_train=True):
        poses_3d, poses_2d, poses_2d_pixel, bones, alphas, contacts, proj_facters = [], [], [], [], [], [], []
        self.cameras = h36m_utils.load_cameras('./data/cameras.h5')

        self.frame_numbers = []
        self.video_name = []
        self.config = config
        self.is_train = is_train
        subjects = h36m_utils.TRAIN_SUBJECTS if is_train else h36m_utils.TEST_SUBJECTS

        positions_set = np.load('./data/data_h36m.npz', allow_pickle=True)['positions_3d'].item()
        if config.trainer.data == 'cpn':
            positions_set_2d = np.load('./data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['positions_2d'].item()
        elif config.trainer.data == 'detectron':
            positions_set_2d = np.load('./data/data_2d_h36m_detectron_ft_h36m.npz', allow_pickle=True)['positions_2d'].item()
        
        # Load human3.6m position data
        for subject in subjects:
            for action in positions_set['S%s' % subject].keys():
                action_sequences = positions_set['S%s' % subject][action]
                sequence_length = action_sequences[0].shape[0]
                for c_idx, set_3d in enumerate(action_sequences):
                    set_3d = set_3d.copy().reshape((set_3d.shape[0], -1))
                    R, T, f, c, k, p, res_w, res_h = self.cameras[(subject, c_idx)]
                    set_3d_world = h36m_utils.camera_to_world_frame(set_3d.reshape((-1, 3)), R, T).reshape(set_3d.shape)
                    augment_depth = random.randint(-5, 20) if config.trainer.data_aug_depth else 0
                    if config.trainer.data == 'gt':
                        set_2d = h36m_utils.project_2d(set_3d.reshape((-1, 3)), R, T, f, c, k, p, augment_depth=augment_depth, from_world=False)[0].reshape((set_3d.shape[0], int(set_3d.shape[-1] / 3 * 2)))
                    else:
                        set_2d = positions_set_2d['S%s' % subject][action][c_idx]
                        set_2d = set_2d.reshape((set_2d.shape[0], -1))[:min(set_3d.shape[0], set_2d.shape[0])]
                        set_3d = set_3d[:min(set_3d.shape[0], set_2d.shape[0])]
                    set_2d_pixel = set_2d
                    set_3d_root = set_3d - np.tile(set_3d[:, :3], [1, int(set_3d.shape[-1]/3)])
                    set_2d_root = set_2d - np.tile(set_2d[:, :2], [1, int(set_2d.shape[-1]/2)])

                    set_2d_root[:, list(range(0, set_2d.shape[-1], 2))] /= res_w
                    set_2d_root[:, list(range(1, set_2d.shape[-1], 2))] /= res_h

                    set_bones = self.get_bones(set_3d_root)
                    set_alphas = np.mean(set_bones, axis=1)

                    self.frame_numbers.append(set_3d_root.shape[0])
                    self.video_name.append('S%s_%s_%s' % (subject, action, c_idx))
                    poses_3d.append(set_3d_root/np.expand_dims(set_alphas, axis=-1))
                    poses_2d.append(set_2d_root)
                    poses_2d_pixel.append(set_2d_pixel)
                    bones.append(set_bones/np.expand_dims(set_alphas, axis=-1))
                    alphas.append(set_alphas)
                    contacts.append(self.get_contacts(set_3d_world))
                    proj_facters.append((set_3d / np.expand_dims(set_alphas, axis=-1)).reshape((set_3d.shape[0], -1, 3))[:, 0, 2])

        self.poses_3d = np.concatenate(poses_3d, axis=0)
        self.poses_2d = np.concatenate(poses_2d, axis=0)
        self.poses_2d_pixel = np.concatenate(poses_2d_pixel, axis=0)
        self.proj_facters = np.concatenate(proj_facters, axis=0)
        self.contacts = np.concatenate(contacts, axis=0)
        self.alphas = np.concatenate(alphas, axis=0)
        self.bones = np.concatenate(bones, axis=0)

        if is_train:
            if config.trainer.data_aug_flip:
                posed_3d_flip = self.get_flipping(self.poses_3d, dim=3)
                posed_2d_flip = self.get_flipping(self.poses_2d, dim=2)
                poses_2d_pixel_flip = self.get_flipping(self.poses_2d_pixel, dim=2)
                self.poses_3d = np.concatenate([self.poses_3d, posed_3d_flip], axis=0)
                self.poses_2d = np.concatenate([self.poses_2d, posed_2d_flip], axis=0)
                self.poses_2d_pixel = np.concatenate([self.poses_2d_pixel, poses_2d_pixel_flip], axis=0)
            if config.trainer.use_loss_D:
                rotations_set = np.load('./data/data_cmu.npz', allow_pickle=True)['rotations']
                self.r_frame_numbers = [r_array.shape[0] for r_array in rotations_set]
                self.rotations = np.concatenate(rotations_set, axis=0)
                self.rotations = self.rotations.reshape((self.rotations.shape[0], -1))
        if config.arch.confidence:
            self.poses_2d_noised, confidence_maps = self.add_noise(self.poses_2d, training=is_train)
            self.poses_2d_noised_with_confidence = np.zeros((self.poses_2d_noised.shape[0], int(self.poses_2d_noised.shape[-1] / 2 * 3)))
            for joint_index in range(int(self.poses_2d_noised.shape[-1] / 2)):
                self.poses_2d_noised_with_confidence[:, 3 * joint_index] = self.poses_2d_noised[:, 2 * joint_index]
                self.poses_2d_noised_with_confidence[:, 3 * joint_index + 1] = self.poses_2d_noised[:, 2 * joint_index + 1]
                self.poses_2d_noised_with_confidence[:, 3 * joint_index + 2] = (confidence_maps[:, 2 * joint_index] + confidence_maps[:, 2 * joint_index]) / 2

        self.set_sequences()
        
        self.poses_2d, self.poses_2d_mean, self.poses_2d_std = util.normalize_data(self.poses_2d_noised_with_confidence if config.arch.confidence else self.poses_2d) 
        self.bones, self.bones_mean, self.bones_std = util.normalize_data(self.bones) 
        self.proj_facters, self.proj_mean, self.proj_std = util.normalize_data(self.proj_facters)

    def __getitem__(self, index):
        items_index = self.sequence_index[index]
        random_flip = (self.config.trainer.data_aug_flip and self.is_train and random.randint(0, 10) > 5)
        poses_2d = self.poses_2d[items_index + int(self.poses_3d.shape[0]/2)] if random_flip else self.poses_2d[items_index]
        posed_3d = self.poses_3d[items_index + int(self.poses_3d.shape[0]/2)] if random_flip else self.poses_3d[items_index]
        poses_2d_pixel = self.poses_2d_pixel[items_index + int(self.poses_3d.shape[0]/2)] if random_flip else self.poses_2d_pixel[items_index]
        bones = self.bones[items_index]
        contacts = self.contacts[items_index]
        alphas = self.alphas[items_index]
        proj_facters = -self.proj_facters[items_index] if random_flip else self.proj_facters[items_index]
        
        if self.is_train:
            if self.config.trainer.use_loss_D:            
                rotations = self.rotations[self.r_sequence_index[np.array(index) % self.r_sequence_index.shape[0]]]
                return poses_2d, posed_3d, bones, contacts, alphas, proj_facters, rotations
            else:
                return poses_2d, posed_3d, bones, contacts, alphas, proj_facters
        else:
            return poses_2d_pixel, poses_2d, posed_3d, bones, contacts, alphas, proj_facters, self.video_name[index]

    def __len__(self):
        return self.sequence_index.shape[0]

    def set_sequences(self):
        def slice_set(offset, frame_number, frame_numbers):
            sequence_index = []
            start_index = 0
            for frames in frame_numbers:
                if frames > train_frames:
                    if self.is_train:
                        clips_number = int((frames - train_frames) // offset)
                        for i in range(clips_number):
                            start = int(i*offset + start_index)
                            end = int(i*offset + train_frames + start_index)
                            sequence_index.append(list(range(start, end)))
                        sequence_index.append(list(range(start_index + frames - train_frames, start_index + frames)))
                    else:
                        sequence_index.append(list(range(start_index, start_index + frames)))
                start_index += frames
            return sequence_index
        offset = 10
        train_frames = random.randint(26, 50) * 4 if self.config.trainer.train_frames == 0 else self.config.trainer.train_frames
        self.sequence_index = np.array(slice_set(offset, train_frames, self.frame_numbers))
        self.r_sequence_index = np.array(slice_set(offset, train_frames, self.r_frame_numbers)) if self.is_train and self.config.trainer.use_loss_D else 0

    def get_flipping(self, poses, dim):
        key_left = [4, 5, 6, 11, 12, 13]
        key_right = [1, 2, 3, 14, 15, 16]
        poses_reshape = poses.reshape((poses.shape[0], -1, dim))
        poses_reshape[:, :, 0] *= -1
        poses_reshape[:, key_left + key_right] = poses_reshape[:, key_right + key_left]
        poses_reshape = poses_reshape.reshape(poses.shape[0], -1)
        return poses_reshape

    def get_contacts(self, poses):
        poses_reshape = poses.reshape((-1, 17, 3))
        contact_signal = np.zeros((poses_reshape.shape[0], 2))
        left_z = poses_reshape[:, 3, 2]
        right_z = poses_reshape[:, 6, 2]

        contact_signal[left_z<=(np.mean(np.sort(left_z)[:left_z.shape[0]//5])+20), 0] = 1
        contact_signal[right_z<=(np.mean(np.sort(right_z)[:right_z.shape[0]//5])+20), 1] = 1
        left_velocity = np.sqrt(np.sum((poses_reshape[2:, 3] - poses_reshape[:-2, 3])**2, axis=-1))
        right_velocity = np.sqrt(np.sum((poses_reshape[2:, 6] - poses_reshape[:-2, 6])**2, axis=-1))
        contact_signal[1:-1][left_velocity>=5, 0] = 0
        contact_signal[1:-1][right_velocity>=5, 1] = 0
        return contact_signal

    def get_bones(self, position_3d):
        def distance(position1, position2):
            return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))
        length = np.zeros((position_3d.shape[0], 10))
        length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
        length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
        length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)
        length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
        length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
        length[:, 5] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
        length[:, 6] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
        length[:, 7] = ((distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 11:3 * 11 + 3]) + distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
        length[:, 8] = ((distance(position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3]) + distance(position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 12:3 * 12 + 3])) / 2)
        length[:, 9] = ((distance(position_3d[:, 3 * 15:3 * 15 + 3], position_3d[:, 3 * 16:3 * 16 + 3]) + distance(position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3])) / 2)
        return length

    def get_parameters(self):
        return self.poses_2d_mean, self.poses_2d_std, self.bones_mean, self.bones_std, self.proj_mean, self.proj_std

    def add_noise(self, pose_array, training):
        # The noise distrobution from lspet_dateset
        # Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle, Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow, Left wrist, Neck, Head top

        noise_distribution = [
            [302, 0.646721, 546, 0.558989, 520, 0.496221, 552, 0.458462, 653, 0.427399, 801, 0.337738, 1072, 0.304372, 1545, 0.305002, 954, 0.470892, 50, 0.640630, 2562],
            [367, 0.466607, 625, 0.407013, 625, 0.378210, 528, 0.394563, 502, 0.372536, 633, 0.351681, 959, 0.325538, 1809, 0.290272, 1697, 0.323600, 160, 0.344122, 1652],
            [201, 0.430108, 634, 0.375215, 677, 0.396447, 797, 0.394465, 982, 0.389279, 1299, 0.390620, 1657, 0.374467, 1641, 0.479758, 742, 0.664335, 46, 1.037032, 881],
            [150, 0.483025, 615, 0.381962, 675, 0.381421, 865, 0.390615, 1006, 0.386777, 1203, 0.387778, 1662, 0.409945, 1674, 0.512305, 756, 0.740501, 40, 0.999209, 911],
            [376, 0.459109, 628, 0.434061, 607, 0.404691, 529, 0.416635, 558, 0.398455, 642, 0.396559, 936, 0.297302, 1876, 0.277350, 1617, 0.358038, 141, 0.347932, 1647],
            [330, 0.614457, 592, 0.550512, 559, 0.499256, 585, 0.414218, 610, 0.414178, 741, 0.342583, 1061, 0.301762, 1552, 0.287223, 939, 0.460175, 48, 0.854059, 2540],
            [216, 0.637702, 371, 0.714801, 359, 0.531207, 386, 0.522511, 390, 0.463886, 474, 0.412121, 882, 0.334252, 1918, 0.271830, 2583, 0.239836, 251, 0.308276, 1727],
            [306, 0.691049, 440, 0.627822, 395, 0.536646, 433, 0.473726, 459, 0.372950, 572, 0.387745, 1013, 0.250803, 2090, 0.235394, 2448, 0.288418, 291, 0.420154, 1110],
            [113, 0.436746, 303, 0.397536, 407, 0.440972, 512, 0.442467, 592, 0.401959, 821, 0.359513, 1290, 0.282377, 2415, 0.242606, 2412, 0.406148, 256, 0.682228, 436],
            [104, 0.503449, 323, 0.455145, 412, 0.371643, 494, 0.477277, 567, 0.459041, 769, 0.356577, 1336, 0.286144, 2344, 0.272577, 2434, 0.453515, 303, 0.635327, 471],
            [274, 0.710043, 446, 0.655773, 369, 0.568227, 406, 0.509342, 443, 0.457261, 623, 0.348271, 988, 0.309336, 2090, 0.271880, 2499, 0.302444, 246, 0.401762, 1173],
            [210, 0.684107, 375, 0.689473, 371, 0.622541, 378, 0.564753, 430, 0.439869, 586, 0.417866, 809, 0.382077, 2032, 0.288338, 2435, 0.259397, 193, 0.292788, 1738],
            [91, 0.562958, 234, 0.466570, 298, 0.445163, 379, 0.403958, 510, 0.419463, 705, 0.403979, 1051, 0.426369, 1941, 0.401862, 3289, 0.484181, 811, 0.731046, 248]
        ]

        noise_distribution = np.array(noise_distribution)
        range_distribution = noise_distribution[:, :20].reshape((-1, 10, 2))
        missing_distribution = noise_distribution[:, -1]

        lsp_h36m_mapping = [12, 2, 1, 0, 3, 4, 5, 12, 12, 12, 12, 9, 10, 11, 8, 7, 6]
        length_mean = np.mean(np.sqrt(np.sum(np.square(pose_array[:, 2:4] - pose_array[:, 4:6]), axis=-1)))
        pose_array = pose_array.copy()
        confidence_map_clone = np.ones(pose_array.shape)
        if training:
            for h36m_index, lsp_index in enumerate(lsp_h36m_mapping):
                if h36m_index > 0:
                    deleted_index = np.random.randint(0, 10000, size=pose_array.shape[0]) < missing_distribution[lsp_index]/3
                    noise_radius = np.mean(range_distribution[lsp_index, :, 1]) * length_mean
                    noises_x = np.random.normal(-noise_radius / 6, noise_radius / 6, size=pose_array.shape[0])
                    noises_y = np.random.normal(-noise_radius / 6, noise_radius / 6, size=pose_array.shape[0])
                    confidences_x = 1 - np.abs(noises_x) / noise_radius
                    confidences_y = 1 - np.abs(noises_y) / noise_radius
                    confidences = (confidences_x + confidences_y) / 2
                    confidences[confidences<0] = 0
                    pose_array[:, h36m_index*2] += noises_x
                    pose_array[:, h36m_index*2+1] += noises_y
                    pose_array[deleted_index, h36m_index*2] = 0
                    pose_array[deleted_index, h36m_index*2+1] = 0
                    
                    confidence_map_clone[:, h36m_index*2] = confidences_x
                    confidence_map_clone[:, h36m_index*2+1] = confidences_y
                    confidence_map_clone[deleted_index, h36m_index*2] = 0
                    confidence_map_clone[deleted_index, h36m_index*2+1] = 0
            return pose_array, confidence_map_clone
        else:
            return pose_array, confidence_map_clone

