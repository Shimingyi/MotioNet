"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import h5py
import json
import numpy as np

# Human3.6m IDs for training and testing
DEBUG_SUBJECTS = [1]
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]
ALL_SUBJECTS = [1, 5, 6, 7, 8, 9, 11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine' 
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose' 
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'


def convert_openpose(openpose_json):
    mapping = [8, 9, 10, 11, 12, 13, 14, [1, 8], 1, 0, [15, 16], 5, 6, 7, 2, 3, 4]
    h36m_locations = np.zeros((len(mapping), 2), dtype=np.float32)
    h36m_confidences = np.zeros(len(mapping), dtype=np.float32)
    if len(openpose_json['people']) > 0:
        openpose_pose = np.array(openpose_json['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        for map_index, map_item in enumerate(mapping):
            if isinstance(map_item, int):
                h36m_locations[map_index] = openpose_pose[map_item][:2]
                h36m_confidences[map_index] = openpose_pose[map_item][2]
            else:
                h36m_locations[map_index] = np.mean(openpose_pose[map_item], axis=0)[:2]
                h36m_confidences[map_index] = np.min(openpose_pose[map_item], axis=0)[2]
    return h36m_locations.reshape(1, -1), h36m_confidences.reshape(1, -1)


def load_camera_params(hf, path):
    """Load h36m camera parameters

    Args
      hf: hdf5 open file with h36m cameras data
      path: path or key inside hf to the camera we are interested in
    Returns
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: (scalar) Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
      name: String with camera id
    """

    R = hf[path.format('R')][:]
    R = R.T

    T = hf[path.format('T')][:]
    f = hf[path.format('f')][:]
    c = hf[path.format('c')][:]
    k = hf[path.format('k')][:]
    p = hf[path.format('p')][:]

    name = hf[path.format('Name')][:]
    name = "".join([chr(int(item[0])) for item in name])

    return R, T, f, c, k, p, name


def load_cameras(bpath):
    """Loads the cameras of h36m

    Args
      bpath: path to hdf5 file with h36m camera data
      subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
      rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath, 'r') as hf:
        for s in ALL_SUBJECTS:
            for c_idx in range(4):  # There are 4 cameras in human3.6m
                R, T, f, c, k, p, name = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s, c_idx + 1))
                if name == '54138969':
                    res_w, res_h = 1000, 1002
                elif name == '55011271':
                    res_w, res_h = 1000, 1000
                elif name == '58860488':
                    res_w, res_h = 1000, 1000
                elif name == '60457274':
                    res_w, res_h = 1000, 1002
                rcams[(s, c_idx)] = (R, T, f, c, k, p, res_w, res_h)
    return rcams


def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Posing", "Purchases", "Sitting",
               "SittingDown", "Smoking", "TakingPhoto", "Waiting",
               "Walking", "WalkingDog", "WalkingTogether"]

    if action == "All" or action == "all" or action == 'ALL':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def project_2d(P, R, T, f, c, k, p, augment_depth=0, from_world=False):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion

    Args
      P: Nx3 points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: (scalar) Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
    Returns
      Proj: Nx2 points in pixel space
      D: 1xN depth of each point in camera space
      radial: 1xN radial distortion per point
      tan: 1xN tangential distortion per point
      r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    if from_world:
        X = R.dot(P.T - T)  # rotate and translate
    else:
        X = P.T
    XX = X[:2, :] / (X[2, :] + augment_depth)
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2, XXX.T


def postprocess_3d(poses):
    """
    Center 3d points around root

    Args
      poses_set: dictionary with 3d data
    Returns
      poses_set: dictionary with 3d data centred around root (center hip) joint
      root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = poses[:, :3]
    poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
    return poses, root_positions


def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates

    Args
      P: Nx3 3d points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
    Returns
      X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot(P.T - T)  # rotate and translate

    return X_cam.T


def camera_to_world_frame(P, R, T):
    """ Inverse of world_to_camera_frame

    Args
      P: Nx3 points in camera coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
    Returns
      X_cam: Nx3 points in world coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot(P.T) + T  # rotate and translate

    return X_cam.T


def cam2world_centered(data_3d_camframe, R, T):
    data_3d_worldframe = camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
    data_3d_worldframe = data_3d_worldframe.reshape((-1, data_3d_camframe.shape[-1]))
    # subtract root translation
    return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, int(data_3d_camframe.shape[-1]/3)))


def dimension_reducer(dimension, predict_number):
    if not dimension in [1, 2, 3]:
        raise (ValueError, 'dim must be 2 or 3')
    if dimension == 2:
        if predict_number == 15:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Spine' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
    else:
        if predict_number == 15:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Spine' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                               dimensions_to_use * 3 + 1,
                                               dimensions_to_use * 3 + 2)))
    return dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=4):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):

        subj, action, seqname = t3dk
        t3d_world = poses_set[t3dk]

        for c in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, c + 1)]
            camera_coord = world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES) * 3])

            sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t3d_camera[(subj, action, sname)] = camera_coord

    return t3d_camera


def project_to_cameras(poses_set, cams, ncams=4):
    """
    Project 3d poses using camera parameters

    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with camera parameters
      ncams: number of cameras per subject
    Returns
      t2d: dictionary with 2d poses
    """
    t2d = {}

    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]

        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam + 1)]
            pts2d, _, _, _, _ = project_2d(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p, from_world=True)

            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES) * 2])
            sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t2d[(subj, a, sname)] = pts2d

    return t2d
