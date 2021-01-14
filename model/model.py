# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from model import model_zoo
from base.base_model import base_model
from utils import util


class fk_model(base_model):
    def __init__(self, config):
        super(fk_model, self).__init__()
        self.config = config
        assert len(config.arch.kernel_size) == len(config.arch.stride) == len(config.arch.dilation)
        self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.rotation_type = config.arch.rotation_type
        self.rotation_number = util.ROTATION_NUMBERS.get(config.arch.rotation_type)
        self.input_feature = 3 if config.arch.confidence else 2
        self.input_joint = 17 if config.trainer.data == 'gt' else 17
        self.output_feature = (17-5)*self.rotation_number # Don't predict the rotation of end-effector joint
        self.output_feature += 1 if config.arch.translation else 0
        self.output_feature += 2 if config.arch.contact else 0

        self.branch_S = model_zoo.pooling_shrink_net(self.input_joint*self.input_feature, 10, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage)
        self.branch_Q = model_zoo.pooling_net(self.input_joint * self.input_feature, self.output_feature, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage)
        
        if config.trainer.use_loss_D:
            self.rotation_D = model_zoo.rotation_D(12*self.rotation_number, 1, 16, 12)
            self.optimizer_D = torch.optim.Adam(list(self.rotation_D.parameters()), lr=0.0001, amsgrad=True)

        self.fk_layer = model_zoo.fk_layer(config.arch.rotation_type)
        self.optimizer_S = torch.optim.Adam(list(self.branch_S.parameters()), lr=config.trainer.lr, amsgrad=True)
        self.optimizer_Q = torch.optim.Adam(list(self.branch_Q.parameters()), lr=config.trainer.lr, amsgrad=True)
        print('Building the network')

    def forward_S(self, _input):
        return self.branch_S(_input)

    def forward_Q(self, _input):
        return self.branch_Q(_input)[:, :, :12*self.rotation_number]

    def forward_proj(self, _input):
        return self.branch_Q(_input)[:, :, -3] if self.config.arch.contact else self.branch_Q(_input)[:, :, -1]
    
    def forward_c(self, _input):
        return self.branch_Q(_input)[:, :, -2:]

    def D(self, rotations):
        frame_offset = 5
        delta_input = rotations[:, frame_offset:] - rotations[:, :-frame_offset]
        return self.rotation_D.forward(delta_input)

    def forward_fk(self, _input, norm_parameters):
        fake_bones = self.forward_S(_input)
        skeleton = bones2skel(fake_bones.clone().detach(), norm_parameters[2], norm_parameters[3])
        output_Q = self.branch_Q(_input)
        fake_rotations = output_Q[:, :, :12*self.rotation_number]
        fake_rotations_full = torch.zeros((fake_rotations.shape[0], fake_rotations.shape[1], 17*self.rotation_number), requires_grad=True).cuda()
        fake_rotations_full[:, :, np.arange(17)*self.rotation_number] = 1 if self.rotation_type == 'q' else 0# Set all to identity quaternion
        complate_indices = np.sort(np.hstack([np.array([0,1,2,4,5,7,8,9,11,12,14,15])*self.rotation_number + i for i in range(self.rotation_number)]))
        fake_rotations_full[:,:,complate_indices] = fake_rotations
        fake_pose_3d = self.fk_layer.forward(self.parents, skeleton.repeat(_input.shape[1], 1, 1), fake_rotations_full.contiguous().view(-1, 17, self.rotation_number)).view(_input.shape[0], _input.shape[1], -1)
        fake_c = output_Q[:, :, -2:] if self.config.arch.contact else None
        fake_proj = (self.branch_Q(_input)[:, :, -3] if self.config.arch.contact else self.branch_Q(_input)[:, :, -1]) if self.config.arch.translation else None
        
        if self.rotation_type == '6d':
            fake_rotations_full = self.fk_layer.convert_6d_to_quaternions(fake_rotations_full.detach()).reshape((-1, self.input_joint, 4))
        elif self.rotation_type == 'eular':
            fake_rotations_full = self.fk_layer.convert_eular_to_quaternions(fake_rotations_full.detach()).reshape((-1, self.input_joint, 4))
        else:
            fake_rotations_full = fake_rotations_full.detach().cpu().numpy()

        return fake_bones, fake_rotations, fake_rotations_full, fake_pose_3d, fake_c, fake_proj

    def lr_decaying(self, decay_rate):
        optimizer_set = [self.optimizer_length, self.optimizer_rotation]
        for optimizer in optimizer_set:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate


def distance(position1, position2):
    vector = torch.abs(position1 - position2)
    return torch.mean(torch.sqrt(torch.sum(torch.pow(vector, 2), dim=-1)), dim=-1)


def get_velocity(motions, joint_index):
    joint_motion = motions[..., [joint_index*3, joint_index*3 + 1, joint_index*3 + 2]]
    velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2])**2, dim=-1))
    return velocity

def bones2skel(bones, bone_mean, bone_std):
    unnorm_bones = bones * bone_std.unsqueeze(0) + bone_mean.repeat(bones.shape[0], 1, 1)
    skel_in = torch.zeros(bones.shape[0], 17, 3).cuda()
    skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
    skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
    skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 7, 1] = unnorm_bones[:, 0, 3]
    skel_in[:, 8, 1] = unnorm_bones[:, 0, 4]
    skel_in[:, 9, 1] = unnorm_bones[:, 0, 5]
    skel_in[:, 10, 1] = unnorm_bones[:, 0, 6]
    skel_in[:, 11, 0] = unnorm_bones[:, 0, 7]
    skel_in[:, 12, 0] = unnorm_bones[:, 0, 8]
    skel_in[:, 13, 0] = unnorm_bones[:, 0, 9]
    skel_in[:, 14, 0] = -unnorm_bones[:, 0, 7]
    skel_in[:, 15, 0] = -unnorm_bones[:, 0, 8]
    skel_in[:, 16, 0] = -unnorm_bones[:, 0, 9]
    return skel_in

