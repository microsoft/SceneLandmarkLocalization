import argparse
import copy
import numpy as np
import os

import sys
sys.path.append(os.path.join(sys.path[0], '..'))

from utils.select_additional_landmarks import load_landmark_visibility_files

def save_landmark_visibility_mask(landmarks, visibility_mask, 
                                  landmark_path, visibility_path):
    
    num_landmarks = landmarks.shape[1]

    np.savetxt(visibility_path, visibility_mask, fmt='%d')

    f = open(landmark_path, 'w')
    f.write('%d\n' % num_landmarks)
    for i in range(num_landmarks):
        f.write('%d %4.4f %4.4f %4.4f\n' % (i, 
                                            landmarks[0, i], 
                                            landmarks[1, i], 
                                            landmarks[2, i]))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scene Landmark Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_folder', type=str, required=True,
        help='Root directory, where all data is stored')
    parser.add_argument(
        '--scene_id', type=str, default='scene6',
        help='Scene id')
    parser.add_argument(
        '--landmark_config', type=str, action='append',
        help='File containing scene-specific 3D landmarks.')
    parser.add_argument(
        '--visibility_config', type=str, action='append',
        help='File containing information about visibility of landmarks in cameras associated with training set.')
    parser.add_argument(
        '--output_format', type=str, required=True,
        help='Output file format.')

    opt = parser.parse_args()
    
    assert len(opt.landmark_config) > 1
    assert len(opt.landmark_config) == len(opt.visibility_config)

    num_landmarks = 0
    num_files = len(opt.landmark_config)
    ls = []
    vs = []
    for (lp, vp) in zip(opt.landmark_config, opt.visibility_config):
        landmark_path = os.path.join(opt.dataset_folder, opt.scene_id, lp + '.txt')
        vis_path = os.path.join(opt.dataset_folder, opt.scene_id, vp + '.txt')
        
        l, v = load_landmark_visibility_files(landmark_path=landmark_path,
                                                visibility_path=vis_path)
        
        num_landmarks += l.shape[1]

        ls.append(l)
        vs.append(v)

    ls = np.concatenate(ls, axis=1)
    vs = np.concatenate(vs, axis=0)

    output_landmark_path = os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks/landmarks-%d%s.txt' % (num_landmarks, opt.output_format))
    
    if 'depth_normal' in opt.visibility_config[0]:
        output_visibility_path = os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks/visibility-%d%s_depth_normal.txt' % (num_landmarks, opt.output_format))
    else:
        output_visibility_path = os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks/visibility-%d%s.txt' % (num_landmarks, opt.output_format))
    save_landmark_visibility_mask(ls, vs, output_landmark_path, output_visibility_path)