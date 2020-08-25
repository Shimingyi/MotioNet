import os
import copy
import json
import argparse
import shutil
import torch
import model.model as models

from data.data_loaders import h36m_loader
from trainer.trainer import fk_trainer
from types import SimpleNamespace as Namespace

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))

    config.device = int(args.device)

    config.arch.kernel_size = list(map(int, args.kernel_size.replace(' ', '').strip().split(','))) if args.kernel_size is not None else config.arch.kernel_size
    config.arch.stride = list(map(int, args.stride.replace(' ', '').strip().split(','))) if args.stride is not None else config.arch.stride
    config.arch.dilation = list(map(int, args.dilation.replace(' ', '').strip().split(','))) if args.dilation is not None else config.arch.dilation

    config.arch.channel = int(args.channel) if args.channel is not None else config.arch.channel
    config.arch.stage = int(args.stage) if args.stage is not None else config.arch.stage
    config.arch.n_type = int(args.n_type) if args.n_type is not None else config.arch.n_type
    config.arch.rotation_type = args.rotation_type if args.rotation_type is not None else config.arch.rotation_type
    config.arch.translation = True if args.translation == 1 else config.arch.translation
    config.arch.confidence = True if args.confidence == 1 else config.arch.confidence
    config.arch.contact = True if args.contact == 1 else config.arch.contact

    config.trainer.data = args.data
    config.trainer.lr = args.lr
    config.trainer.batch_size = args.batch_size
    config.trainer.train_frames = args.train_frames
    config.trainer.use_loss_foot = True if args.loss_terms[0] == '1' else False
    config.trainer.use_loss_3d = True if args.loss_terms[1] == '1' else False
    config.trainer.use_loss_2d = True if args.loss_terms[2] == '1' else False
    config.trainer.use_loss_D = True if args.loss_terms[3] == '1' else False
    config.trainer.data_aug_flip = True if args.augmentation_terms[0] == '1' else False
    config.trainer.data_aug_depth = True if args.augmentation_terms[1] == '1' else False

    config.trainer.checkpoint_dir = '%s/%s_%s_k%s_s%s_d%s_c%s_%s_%s_%s%s%s_%s_%s_loss%s_aug%s' % (config.trainer.save_dir, args.name, args.data, 
                                                                                           str(config.arch.kernel_size).strip('[]').replace(' ', ''),
                                                                                           str(config.arch.stride).strip('[]').replace(' ', ''),
                                                                                           str(config.arch.dilation).strip('[]').replace(' ', ''),
                                                                                           config.arch.channel, config.arch.stage, config.arch.rotation_type, 
                                                                                           't' if config.arch.translation else '',
                                                                                           'c' if config.arch.confidence else '',
                                                                                           'c' if config.arch.contact else '',
                                                                                           args.lr, args.batch_size, args.loss_terms, args.augmentation_terms)
    return config


def train(config, resume):

    print("Loading dataset..")

    train_data_loader = h36m_loader(config, is_training=True)
    test_data_loader = h36m_loader(config, is_training=False)

    model = getattr(models, config.arch.type)(config)
    # model.summary()

    trainer = fk_trainer(model, resume=resume, config=config, data_loader=train_data_loader, test_data_loader=test_data_loader)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='### MotioNet training')

    # Runtime parameters
    parser.add_argument('-c', '--config', default='./config_zoo/default.json', type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0', type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default='debug_model', type=str, help='The name of this training')
    parser.add_argument('--data', default='gt', type=str, help='The training data, gt - projected 2d pose; cpn; detectron')

    # Network definition
    parser.add_argument('--kernel_size', default=None, type=str, help='The kernel_size set of the convolution')
    parser.add_argument('--stride', default=None, type=str, help='The stride set of the convolution')
    parser.add_argument('--dilation', default=None, type=str, help='The dilation set of the convolution')
    parser.add_argument('--channel', default=None, type=int, help='The channel number of the network')
    parser.add_argument('--stage', default=None, type=int, help='The stage of the network')
    parser.add_argument('--n_type', default=None, type=int, help='The network architecture of rotation branch 0 - deconv 1- avgpool')
    parser.add_argument('--rotation_type', default=None, type=str, help='The type of rotations, including 6d, 5d, q, eular')
    parser.add_argument('--translation', default=None, type=int, help='If we want to use translation in the network, 0 - No, 1 - Yes')
    parser.add_argument('--confidence', default=None, type=int, help='If we want to use confidence map in the network, 0 - No, 1 - Yes')
    parser.add_argument('--contact', default=None, type=int, help='If we want to use foot contact in the network, 0 - No, 1 - Yes')

    # Training parameters
    parser.add_argument('--lr', default=0.001, type=float, help='The learning rate in the training')
    parser.add_argument('--batch_size', default=128, type=int, help='The batch size')
    parser.add_argument('--train_frames', default=0, type=int, help='The frames number for a training clip, 0 mean random number')
    parser.add_argument('--loss_terms', default='0100', type=str, help='The loss in training we want to use for [foot_contact, 3d_pose, 2d_pose, adversarial] we want to use, 0 - No, 1 - Yes, like: 11111')
    parser.add_argument('--augmentation_terms', default='00', type=str, help='Data augmentation in training we want to use for [pose_flip, projection_depth], 0 - No, 1 - Yes, like: 11')

    args = parser.parse_args()
    if args.config:
        config = config_parse(args)
    elif args.resume:
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if os.path.exists(config.trainer.checkpoint_dir) and 'debug' not in args.name and args.resume is None:
        allow_cover = input('Model file detected, do you want to replace it? (Y/N)')
        allow_cover = allow_cover.lower()
        if allow_cover == 'n':
            exit()
        else:
            shutil.rmtree(config.trainer.checkpoint_dir, ignore_errors=True)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    train(config, args.resume)
