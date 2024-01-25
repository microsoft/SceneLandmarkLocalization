# Copyright (c) Microsoft Corporation. All rights reserved.
import argparse
import os
#import re

Args = None

def launch_training():
    print("Experiment File: %s" % Args.experiment_file)
    print("Model Dir: %s" % Args.model_dir)
    cmd = 'python main.py --action train_patches'
    cmd += ' --training_batch_size %d' % (Args.training_batch_size)
    cmd += ' --output_downsample %d' % (Args.output_downsample)
    cmd += ' --num_epochs %d' % (Args.num_epochs)
    cmd += ' --dataset_folder %s' % (Args.dataset_dir)
    cmd += ' --scene_id %s' % (Args.scene_id)
    cmd += ' --landmark_config %s' % (Args.landmark_config)
    cmd += ' --visibility_config %s' % (Args.visibility_config)
    cmd += ' --output_folder %s' % (Args.model_dir)
    cmd += ' --landmark_indices %d' % (Args.landmark_index_start)
    cmd += ' --landmark_indices %d' % (Args.landmark_index_stop)
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', type=str, required=True,
        help="Dataset folder path.")
    parser.add_argument(
        '--experiment_file', type=str, required=True,
        help="Experiment file path.")
    parser.add_argument(
        '--scene_id', type=str, required=True,
        help="name of scene.")
    parser.add_argument(
        '--landmark_config', type=str, required=True,
        help='Landmark configuration.')
    parser.add_argument(
        '--visibility_config', type=str, required=True,
        help='Visibility configuration.')
    parser.add_argument(
        '--num_landmarks', type=int, required=True,
        help='number of landmarks.')
    parser.add_argument(
        '--block_size', type=int, required=True,
        help='number of landmarks in each block.')
    parser.add_argument(
        '--subset_index', type=int, required=True,
        help='index of landmark subset (starts from 0).')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='folder to save experiment file in.')
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='folder to save model ckpt file in.')
    parser.add_argument(
        '--training_batch_size', type=int, required=True,
        help='batch size.')
    parser.add_argument(
        '--output_downsample', type=int, required=True,
        help='Downsample factor for heat map resolution.')
    parser.add_argument(
        '--num_epochs', type=int, required=True,
        help='the number of epochs used for training.')
    Args = parser.parse_args()

    # Write the experiment file
    exp_fn = os.path.join(Args.output_dir, Args.experiment_file)
    fd = open(exp_fn, "w")
    for lid in range(0, Args.num_landmarks, Args.block_size):
        Args.landmark_index_start = lid
        Args.landmark_index_stop = lid + Args.block_size
        str = '%s %s-%03d-%03d %s %s local' % (Args.scene_id, Args.scene_id, Args.landmark_index_start, Args.landmark_index_stop, Args.landmark_config, Args.visibility_config)
        print(str, file=fd)
    fd.close()

    # Launch the training job for the specified subset only.
    Args.landmark_index_start = Args.block_size * Args.subset_index
    Args.landmark_index_stop = Args.block_size * (Args.subset_index + 1)
    launch_training()