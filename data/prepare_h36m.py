import os
import numpy as np

from spacepy import pycdf

if __name__ == '__main__':
    extracted_folder = '/mnt/dataset/human3.6m/extracted'
    camera_ids = ['54138969', '55011271', '58860488', '60457274']
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    dim_to_use = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    data_3d_gt = {}
    data_2d_gt = {}

    for subject in subjects:
        data_3d_gt[subject] = {}
        data_2d_gt[subject] = {}
        subj_folder = '%s/%s' % (extracted_folder, subject)
        actions = [item.split('.')[0] for item in os.listdir(path=os.path.join(subj_folder, 'D3_Angles'))]
        for action in actions:
            print('Reading %s in %s' % (action, subject))
            if subject == 'S11' and action == 'Directions':
                continue
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                       .replace('WalkingDog', 'WalkDog')
            data_3d_gt[subject][canonical_name] = []
            data_2d_gt[subject][canonical_name] = []
            for camera_index, camera in enumerate(camera_ids):
                base_filename = '%s.%s' % (action, camera)
                with pycdf.CDF(os.path.join(subj_folder, 'Poses_D2_Positions', base_filename + '.cdf')) as cdf:
                    poses_2d = np.array(cdf['Pose'])
                    poses_2d = poses_2d.reshape(poses_2d.shape[1], 32, 2)[:, dim_to_use]
                with pycdf.CDF(os.path.join(subj_folder, 'Poses_D3_Positions_mono', base_filename + '.cdf')) as cdf:
                    poses_3d = np.array(cdf['Pose'])
                    poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)[:, dim_to_use]
                
                data_2d_gt[subject][canonical_name].append(poses_2d.astype('float32'))
                data_3d_gt[subject][canonical_name].append(poses_3d.astype('float32'))
    
    print('Saving...')
    np.savez_compressed('./data/data_h36m.npz', positions_3d=data_3d_gt)