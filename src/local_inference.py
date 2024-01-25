# Copyright (c) Microsoft Corporation. All rights reserved.
#from __future__ import print_function
import argparse
import os
import time

Args = None

def local_inference():
    cmd = 'python main.py --action test --dataset_folder %s --scene_id %s --landmark_config %s --visibility_config %s' % (Args.dataset_dir, Args.scene_id, Args.landmark_config, Args.visibility_config)
    cmd += ' --output_downsample 8'
    cmd += ' --landmark_indices 0'
    for i in range(0, len(Args.landmark_indices)):
        cmd += ' --landmark_indices %d' % (Args.landmark_indices[i])
    for ckpt in Args.checkpoint_names:
        cmd += ' --pretrained_model %s/%s/%s/model-best_median.ckpt' % (Args.checkpoint_dir, Args.experimentGroupName, ckpt)
    cmd += ' --output_folder %s/%s' % (Args.checkpoint_dir, Args.experimentGroupName)
    print("Running [" + cmd + "]")
    os.system(cmd)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_file', default="", type=str, required=True,
        help="Experiment file path.")
    parser.add_argument(
        '--dataset_dir', default="", type=str, required=True,
        help="Dataset path.")
    parser.add_argument(
        '--checkpoint_dir', default="", type=str, required=True,
        help="Checkpoints folder path.")

    Args = parser.parse_args()

    tmp = os.path.basename(Args.experiment_file)
    Args.experimentGroupName = tmp[:tmp.rindex('.')]
    Args.landmark_indices = []
    Args.checkpoint_names = []
    exp_file = os.path.join(Args.checkpoint_dir, Args.experiment_file)
    fd = open(exp_file, 'r')
    while True:
        line = fd.readline()
        if line == '':
            break
        split_line = line.split()

        Args.scene_id = split_line[0]
        expName = split_line[1]

        Args.landmark_config = split_line[2]
        Args.visibility_config = split_line[3]

        Args.checkpoint_names.append(expName)
        fields = expName.split('-')
        Args.landmark_indices.append(int(fields[2]))

    local_inference()