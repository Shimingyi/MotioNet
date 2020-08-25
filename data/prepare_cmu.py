import sys
sys.path.append('./')

import numpy as np
import utils.BVH as BVH

from utils.Quaternions import Quaternions
from utils import util

rotations_bvh = []
bvh_files = util.make_dataset(['/mnt/dataset/cmubvh'], phase='bvh', data_split=1, sort_index=0)
for file in bvh_files:
    original_anim, _, frametime = BVH.load(file, rotate=True)
    sampling = 3
    to_keep = [0, 7, 8, 2, 3, 12, 13, 15, 18, 19, 25, 26]
    real_rotations = original_anim.rotations.qs[1:, to_keep, :]
    rotations_bvh.append(real_rotations[np.arange(0, real_rotations.shape[0] // sampling) * sampling].astype('float32'))
np.savez_compressed('./data/data_cmu.npz', rotations=rotations_bvh)