import os
import json
import argparse
import torch
import numpy as np

import model.model as models

from data.data_loaders import h36m_loader
from utils import util, h36m_utils, visualization
from utils import Animation
from utils import BVH
from model import metric


def main(config, args, output_folder):
    resume = args.resume
    name_list = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
    model = getattr(models, config.arch.type)(config)

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if args.input == 'h36m':
        test_data_loader = h36m_loader(config, is_training=False)
        test_parameters = [torch.from_numpy(np.array(item)).float().to(device) for item in test_data_loader.dataset.get_parameters()]
        error_list = {}
        errors = []
        sampling_export = np.random.choice(test_data_loader.n_samples-1, size=4, replace=False)
        for video_idx, datas in enumerate(test_data_loader):
            video_name = datas[-1][0]
            datas = [item.float().to(device) for item in datas[:-1]]
            poses_2d_pixel, poses_2d, poses_3d, bones, contacts, alphas, proj_facters = datas
            with torch.no_grad():
                pre_bones, pre_rotations, pre_rotations_full, pre_pose_3d, pre_c, pre_proj = model.forward_fk(poses_2d, test_parameters)
            error = metric.mean_points_error(poses_3d, pre_pose_3d)*torch.mean(alphas[0]).data.cpu().numpy()
            errors.append(error)
            action_name = video_name.split('_')[1].split(' ')[0]
            if action_name in error_list.keys():
                error_list[action_name].append(error)
            else:
                error_list[action_name] = [error]
            if video_idx in sampling_export:
                if config.arch.translation:
                    R, T, f, c, k, p, res_w, res_h = test_data_loader.dataset.cameras[(int(video_name.split('_')[0].replace('S', '')), int(video_name.split('_')[-1]))]
                    pose_2d_film = (poses_2d_pixel[0, :, :2].cpu().numpy() - c[:, 0]) / f[:, 0]
                    translations = np.ones(shape=(pose_2d_film.shape[0], 3))
                    translations[:, :2] = pose_2d_film
                    translation = (translations * np.repeat(pre_proj[0].cpu().numpy(), 3, axis=-1).reshape((-1, 3))) * 8
                else:
                    translation = np.zeros((poses_2d.shape[1], 3))
                rotations = pre_rotations_full[0]
                length = (pre_bones * test_parameters[3].unsqueeze(0) + test_parameters[2].repeat(bones.shape[0], 1, 1))[0].cpu().numpy()
                BVH.save('%s/%s.bvh' % (output_folder, video_name), Animation.load_from_network(translation, rotations, length, third_dimension=1), names=name_list)
        error_file = '%s/errors.txt' % output_folder
        with open(error_file, 'w') as f:
            f.writelines('=====Action===== ==mm==\n')
            total = []
            for key in error_list.keys():
                mean_error = np.mean(np.array(error_list[key]))
                total.append(mean_error)
                print('%16s %.2f' % (key, mean_error))
                f.writelines('%16s %.2f \n' % (key, mean_error))
            print('%16s %.2f' % ('Average', np.mean(np.array(errors))))
            f.writelines('%16s %.2f \n' % ('Average', np.mean(np.array(errors))))
            f.close()
    else:
        parameters = [torch.from_numpy(np.array(item)).float().to(device) for item in h36m_loader(config, is_training=True).dataset.get_parameters()]
        def export(pose_folder):
            video_name = pose_folder.split('/')[-1]
            files = util.make_dataset([pose_folder], phase='json', data_split=1, sort=True, sort_index=1)
            IMAGE_WIDTH = 1080 # Should be changed refer to your test data
            IMAGE_HEIGHT = 1080
            pose_batch = []
            confidence_batch = []
            for pose_file_name in files:
                with open(pose_file_name, 'r') as f:
                    h36m_locations, h36m_confidence = h36m_utils.convert_openpose(json.load(f))
                    pose_batch.append(h36m_locations)
                    confidence_batch.append(h36m_confidence)
            poses_2d = np.concatenate(pose_batch, axis=0)
            poses_2d[:, np.arange(0, poses_2d.shape[-1], 2)] /= (IMAGE_WIDTH*1) # The last number 1 is an adjustable varible, if the person takes full space of image, try to use a bigger number like 2
            poses_2d[:, np.arange(1, poses_2d.shape[-1], 2)] /= (IMAGE_HEIGHT*1) # The last number 1 is an adjustable varible, if the person takes full space of image, try to use a bigger number like 2
            confidences = np.concatenate(confidence_batch, axis=0)
            poses_2d_root = (poses_2d - np.tile(poses_2d[:, :2], [1, int(poses_2d.shape[-1] / 2)]))
            if args.smooth:
                poses_2d_root, confidences = util.interp_pose(poses_2d_root, confidences, k=2)
            if config.arch.confidence:
                poses_2d_root_c = np.zeros((poses_2d_root.shape[0], int(poses_2d_root.shape[-1]/2*3)))
                for joint_index in range(int(poses_2d_root.shape[-1] / 2)):
                    poses_2d_root_c[:, 3 * joint_index] = poses_2d_root[:, 2 * joint_index].copy()
                    poses_2d_root_c[:, 3 * joint_index + 1] = poses_2d_root[:, 2 * joint_index + 1].copy()
                    poses_2d_root_c[:, 3 * joint_index + 2] = np.array(confidences)[:, joint_index].copy()
                poses_2d_root = poses_2d_root_c
            poses_2d_root = np.divide((poses_2d_root - parameters[0].cpu().numpy()), parameters[1].cpu().numpy())
            poses_2d_root = np.where(np.isfinite(poses_2d_root), poses_2d_root, 0)
            poses_2d_root = torch.from_numpy(np.array(poses_2d_root)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pre_bones, pre_rotations, pre_rotations_full, pre_pose_3d, pre_c, pre_proj = model.forward_fk(poses_2d_root, parameters)
            if config.arch.translation:
                pose_2d_film = (poses_2d[:, :2] - 0.5)
                translations = np.ones(shape=(pose_2d_film.shape[0], 3))
                translations[:, :2] = pose_2d_film
                translation = (translations * np.repeat(pre_proj[0].cpu().numpy(), 3, axis=-1).reshape((-1, 3)))
                translation[:] -= translation[[0]]
                # The scaling factor can be updated by different case
                translation[:, :2] = translation[:, :2]*3
                translation[:, 2] = translation[:, 2]*1.5
            else:
                translation = np.zeros((poses_2d.shape[0], 3))
            rotations = pre_rotations_full[0]
            length = (pre_bones * parameters[3].unsqueeze(0) + parameters[2].repeat(pre_bones.shape[0], 1, 1))[0].cpu().numpy()
            BVH.save('%s/%s.bvh' % (output_folder, video_name), Animation.load_from_network(translation, rotations, length, third_dimension=1), names=name_list)
            print('The bvh file of %s has been saved!' % video_name)
        if args.input == 'demo':
            for folder_name in [0, 2, 3, 4, 5, 6]: # The pretrained wild model required at least 101 frames as input, if you would like to use it in a short video, please train another version with --stage_number 1
                export('./data/example/%s' % folder_name)
        else:
            export(args.input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='### MotioNet eveluation')

    parser.add_argument('-r', '--resume', default='./checkpoints/h36m_gt.pth', type=str,
                           help='path to checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="0", type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-i', '--input', default='h36m', type=str,
                       help='h36m or demo or [input_folder_path]')
    parser.add_argument('-o', '--output', default='./output', type=str,
                        help='Output folder')
    parser.add_argument('-s', '--smooth', default=False, action='store_true', 
                        help='smooth function enabled')
    parser.add_argument('--interface', default='openpose', type=str,
                       help='2D detection interface')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.resume:
        config = torch.load(args.resume)['config']
    output_folder = util.mkdir_dir('%s/%s' % (args.output, config.trainer.checkpoint_dir.split('/')[-1]))
    main(config, args, output_folder)
