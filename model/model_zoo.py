import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base.base_model import base_model
from utils.Quaternions import Quaternions


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class linear(nn.Module):
    def __init__(self, linear_size, channel_size, p_dropout=0.5):
        super(linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(channel_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(channel_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class regression(base_model):
    def __init__(self, input_size=32, output_size=32, channel_size=243, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(regression, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = input_size
        self.output_size = output_size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(channel_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, channel_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y


class fk_layer(base_model):
    def __init__(self, rotation_type):
        super(fk_layer, self).__init__()
        self.rotation_type = rotation_type

    def normalize_vector(self, v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v

    def cross_product(self, u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

        return out

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def transforms_blank(self, rotations):
        diagonal = torch.diag(torch.ones(4))[None, None, :, :].cuda()
        ts = diagonal.repeat(int(rotations.shape[0]), int(rotations.shape[1]), 1, 1)
        return ts

    def transforms_rotations(self, rotations):
        if self.rotation_type == 'q':
            q_length = torch.sqrt(torch.sum(torch.pow(rotations, 2), dim=-1))
            qw = rotations[..., 0].clone() / q_length
            qx = rotations[..., 1].clone() / q_length
            qy = rotations[..., 2].clone() / q_length
            qz = rotations[..., 3].clone() / q_length
            qw[qw != qw] = 0
            qx[qx != qx] = 0
            qy[qy != qy] = 0
            qz[qz != qz] = 0
            """Unit quaternion based rotation matrix computation"""
            x2 = qx + qx
            y2 = qy + qy
            z2 = qz + qz
            xx = qx * x2
            yy = qy * y2
            wx = qw * x2
            xy = qx * y2
            yz = qy * z2
            wy = qw * y2
            xz = qx * z2
            zz = qz * z2
            wz = qw * z2

            dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
            dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
            dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
            m = torch.stack([dim0, dim1, dim2], dim=-2)
        elif self.rotation_type == '6d':
            rotations_reshape = rotations.view((-1, 6))
            x_raw = rotations_reshape[:, 0:3]  # batch*3
            y_raw = rotations_reshape[:, 3:6]  # batch*3

            x = self.normalize_vector(x_raw)  # batch*3
            z = self.cross_product(x, y_raw)  # batch*3
            z = self.normalize_vector(z)  # batch*3
            y = self.cross_product(z, x)  # batch*3

            x = x.view(-1, 3, 1)
            y = y.view(-1, 3, 1)
            z = z.view(-1, 3, 1)
            m = torch.cat((x, y, z), 2).reshape((rotations.shape[0], rotations.shape[1], 3, 3)) # batch*3*3
        elif self.rotation_type == 'eular':
            rotations_reshape = rotations.view((-1, 3))
            batch = rotations_reshape.shape[0]
            c1 = torch.cos(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
            s1 = torch.sin(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
            c2 = torch.cos(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
            s2 = torch.sin(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
            c3 = torch.cos(rotations_reshape[:, 1]).view(batch, 1)  # batch*1
            s3 = torch.sin(rotations_reshape[:, 1]).view(batch, 1)  # batch*1

            row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
            row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
            row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

            m = torch.cat((row1, row2, row3), 1).reshape((rotations.shape[0], rotations.shape[1], 3, 3))  # batch*3*3
        return m

    def transforms_local(self, positions, rotations):
        transforms = self.transforms_rotations(rotations)
        transforms = torch.cat([transforms, positions[:, :, :, None]], dim=-1)
        zeros = torch.zeros(
            [int(transforms.shape[0]),
             int(transforms.shape[1]), 1, 3]).cuda()
        ones = torch.ones([int(transforms.shape[0]), int(transforms.shape[1]), 1, 1]).cuda()
        zerosones = torch.cat([zeros, ones], dim=-1)
        transforms = torch.cat([transforms, zerosones], dim=-2)
        return transforms

    def transforms_global(self, parents, positions, rotations):
        locals = self.transforms_local(positions, rotations)
        globals = self.transforms_blank(rotations)

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        globals = list(torch.chunk(globals, int(globals.shape[1]), dim=1))
        for i in range(1, positions.shape[1]):
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0],
                                                  locals[:, i])[:, None, :, :]
        return torch.cat(globals, dim=1)

    def forward(self, parents, positions, rotations):
        positions = self.transforms_global(parents, positions,
                                           rotations)[:, :, :, 3]
        return positions[:, :, :3] / positions[:, :, 3, None]

    def convert_6d_to_quaternions(self, rotations):
        rotations_reshape = rotations.view((-1, 6))
        x_raw = rotations_reshape[:, 0:3]  # batch*3
        y_raw = rotations_reshape[:, 3:6]  # batch*3

        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrices = torch.cat((x, y, z), 2).cpu().numpy()
        q = Quaternions.from_transforms(matrices)
        return q.qs

    def convert_eular_to_quaternions(self, rotations):
        rotations_reshape = rotations.view((-1, 3))
        batch = rotations_reshape.shape[0]
        c1 = torch.cos(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
        s1 = torch.sin(rotations_reshape[:, 0]).view(batch, 1)  # batch*1
        c2 = torch.cos(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
        s2 = torch.sin(rotations_reshape[:, 2]).view(batch, 1)  # batch*1
        c3 = torch.cos(rotations_reshape[:, 1]).view(batch, 1)  # batch*1
        s3 = torch.sin(rotations_reshape[:, 1]).view(batch, 1)  # batch*1
        row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
        row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
        row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3
        matrices = torch.cat((row1, row2, row3), 1).cpu().numpy()
        q = Quaternions.from_transforms(matrices)
        return q.qs


class conv_shrink_net(base_model):
    def __init__(self, in_features, out_features, channels, n_downsampling=3):
        super(conv_shrink_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channels, kernel_size=3, stride=2, bias=True)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, out_features, 1)
        self.n_downsampling = n_downsampling
        layers = []

        for i in range(0, n_downsampling):
            layers.append(nn.Conv1d(channels, channels, 3, 2, dilation=1, bias=True))
            layers.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=True))
            layers.append(nn.BatchNorm1d(channels, momentum=0.1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(0, self.n_downsampling):
            # res = x[:, :, 3 // 2:: 3]

            x = self.drop(self.relu(self.layers[4*i + 1](self.layers[4*i](x))))
            x = self.drop(self.relu(self.layers[4*i + 3](self.layers[4*i + 2](x))))

        x = F.adaptive_max_pool1d(x, 1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class pooling_shrink_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(pooling_shrink_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=3, stride=2, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.shrink = nn.Conv1d(channel, out_features, 1)
        self.stage_number = stage_number
        layers = []

        for stage_index in range(0, stage_number):
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.stage_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for layer in self.stage_layers:
            x = self.drop(self.relu(layer(x)))
        x = F.adaptive_max_pool1d(x, 1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class pooling_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(pooling_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=1, stride=1, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        layers = []

        for stage_index in range(0, stage_number):
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for stage_index in range(0, self.stage_number):
            output = 0
            for conv_index in range(self.conv_depth):
                output += F.adaptive_avg_pool1d(self.drop(self.relu(self.layers[stage_index*self.conv_depth + conv_index](x))), x.shape[-1]) 
            x = output
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class rotation_D(base_model):
    def __init__(self, in_features, out_features, channel, joint_numbers):
        super(rotation_D, self).__init__()
        self.local_fc_layers = nn.ModuleList()
        self.joint_numbers = joint_numbers
        self.shrink_frame_number = 24
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.joint_numbers, self.joint_numbers, kernel_size=4, stride=1, bias=False)
        self.conv2 = nn.Conv1d(self.joint_numbers, self.joint_numbers, kernel_size=1, stride=1, bias=False)

        for i in range(joint_numbers):
            self.local_fc_layers.append(
                nn.Linear(in_features=self.shrink_frame_number, out_features=1)
            )

    # Get input B*T*J*4
    def forward(self, x):
        x = x.reshape((x.shape[0], -1, self.joint_numbers))
        x = torch.transpose(x, 1, 2)

        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        x = F.adaptive_avg_pool1d(x, self.shrink_frame_number)
        layer_output = []
        for i in range(self.joint_numbers):
            layer_output.append(self.local_fc_layers[i](x[:,i,:].clone()))
        return torch.cat(layer_output, -1)


class deconv_net(base_model):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(deconv_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        channels = [channel] * (len(stride_set) + 1)
        for stride_index, stride in enumerate(stride_set):
            channels[stride_index+1] = channels[stride_index]*stride
        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_features, channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(channel, momentum=0.1),
            self.relu, self.drop)
        stage_layers = []
        for stage_index in range(stage_number):
            for conv_index in range(len(kernel_size_set)):
                stage_layers.append(
                    nn.Sequential(
                        nn.ReflectionPad1d(int((kernel_size_set[conv_index] - stride_set[conv_index]) / 2)),
                        nn.Conv1d(channels[conv_index], channels[conv_index+1],
                                  kernel_size=kernel_size_set[conv_index], stride=stride_set[conv_index],
                                  dilation=dilation_set[conv_index]),
                        nn.BatchNorm1d(channels[conv_index+1], momentum=0.1),
                        self.relu, self.drop
                    )
                )
            for conv_index in range(len(kernel_size_set)):
                inverse_index = -conv_index - 1
                padding = int((kernel_size_set[inverse_index] - stride_set[inverse_index]) / 2)
                stage_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(channels[inverse_index], channels[inverse_index-1],
                                           kernel_size=kernel_size_set[inverse_index], stride=stride_set[inverse_index], padding=padding),
                        nn.BatchNorm1d(channels[inverse_index-1], momentum=0.1),
                        self.relu, self.drop,
                        nn.ConvTranspose1d(channels[inverse_index-1], channels[inverse_index-1], 1, 1),
                        nn.BatchNorm1d(channels[inverse_index-1], momentum=0.1),
                        self.relu, self.drop
                    )
                )
        self.shrink_conv = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(channels[0], out_features, kernel_size=1, stride=1)
            )
        )
        self.stage_layers = nn.ModuleList(stage_layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.expand_conv(x)
        for layer in self.stage_layers:
            x = layer(x)
        x = self.shrink_conv(x)
        return torch.transpose(x, 1, 2)


class deconv_net_1(base_model):
    def __init__(self, in_features, out_features, n_stage=3, kernel_size=5, stride=2):
        super(deconv_net_1, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        self.n_stage = n_stage
        padding = int((kernel_size-1) / 2)

        expand_channel = in_features*stride

        self.conv1 = nn.Sequential(nn.Conv1d(in_features, expand_channel, kernel_size=kernel_size, stride=stride), )

        # out_padding = 0 if stride // 2 == 1 else 1
        out_padding = stride - 1
        channels = []
        for i in range(0, n_stage+1):
            channels.append(pow(2, i) * in_features)
        layers_conv = []
        layers_deconv = []

        for i in range(0, n_stage):
            layers_conv.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            layers_conv.append(nn.BatchNorm1d(channels[i + 1], momentum=0.1))
            layers_conv.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, stride=stride, padding=0, bias=True))
            layers_conv.append(nn.BatchNorm1d(channels[i + 1], momentum=0.1))

        for i in range(0, n_stage - 1):
            layers_deconv.append(nn.ConvTranspose1d(channels[-i-1], channels[-i-2], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, bias=True))
            layers_deconv.append(nn.BatchNorm1d(channels[-i-2], momentum=0.1))
        layers_deconv.append(nn.ConvTranspose1d(channels[1], 64, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, bias=True))
        layers_deconv.append(nn.BatchNorm1d(out_features, momentum=0.1))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_deconv = nn.ModuleList(layers_deconv)

    def forward(self, x):

        x = torch.transpose(x, 1, 2)

        for i in range(0, self.n_stage):
            output1 = self.drop(self.relu(self.layers_conv[4 * i + 1](self.layers_conv[4 * i](x))))
            output2 = self.drop(self.relu(self.layers_conv[4 * i + 3](self.layers_conv[4 * i + 2](x))))
            x = output1 + output2

        # 128 384 32
        for i in range(0, self.n_stage):
            x = self.drop(self.relu(self.layers_deconv[2 * i + 1](self.layers_deconv[2 * i](x))))
        return torch.transpose(x, 1, 2)


class deconv_net_2(base_model):
    def __init__(self, in_features, out_features, n_stage=3, kernel_size=5, stride=2):
        super(deconv_net_2, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        self.n_stage = n_stage
        padding = int((kernel_size-1) / 2)
        # out_padding = 0 if stride // 2 == 1 else 1
        out_padding = stride - 1
        channels = []
        for i in range(0, n_stage+1):
            channels.append(pow(2, i) * in_features)
        layers_conv = []
        layers_deconv = []

        for i in range(0, n_stage):
            layers_conv.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            layers_conv.append(nn.BatchNorm1d(channels[i + 1], momentum=0.1))
            layers_conv.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, stride=stride, padding=0, bias=True))
            layers_conv.append(nn.BatchNorm1d(channels[i + 1], momentum=0.1))
            layers_deconv.append(nn.ConvTranspose1d(channels[-i-1], channels[-i-2], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, bias=True))
            layers_deconv.append(nn.BatchNorm1d(channels[-i-2], momentum=0.1))
        layers_deconv.append(nn.ConvTranspose1d(channels[1], out_features, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, bias=True))
        layers_deconv.append(nn.BatchNorm1d(out_features, momentum=0.1))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_deconv = nn.ModuleList(layers_deconv)

    def forward(self, x):

        x = torch.transpose(x, 1, 2)

        for i in range(0, self.n_stage):
            output1 = self.drop(self.relu(self.layers_conv[4 * i + 1](self.layers_conv[4 * i](x))))
            output2 = self.drop(self.relu(self.layers_conv[4 * i + 3](self.layers_conv[4 * i + 2](x))))
            x = output1 + output2

        # 128 384 32
        for i in range(0, self.n_stage):
            x = self.drop(self.relu(self.layers_deconv[2 * i + 1](self.layers_deconv[2 * i](x))))
        return torch.transpose(x, 1, 2)


