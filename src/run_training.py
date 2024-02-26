from math import exp
import os
import statistics as st
from tabnanny import check

if __name__ == '__main__':

    home_dir = os.path.expanduser("~")

    # Specify the paths to the dataset and the output folders.
    dataset_dir = os.path.join(home_dir, "data/indoor6")
    output_dir = os.path.join(home_dir, "data/outputs")

    # Specify a version number which can be incremented when training multiple variants on 
    # the same scene.
    version_no = 10
    
    # Specify the scene name
    scene_name = 'scene6'

    # Specify the landmark file
    landmark_config = 'landmarks/landmarks-1000v10'

    # Specify the visibility file
    visibility_config = 'landmarks/visibility-1000v10_depth_normal'

    # Specify the batch size for the minibatches used for training.
    training_batch_size = 8
    
    # Specify the downsample factor for the output heatmap.
    output_downsample = 8
    
    # Specify the number of epochs to use during training.
    num_epochs = 200

    # Specify the number of landmarks and the block size. The number of landmarks should be 
    # identical to the number of landmarks in the landmark file specified for the 
    # landmark_config parameter.
    num_landmarks = 1000

    # Specify the number of landmarks that will be present in each subset when the set of 
    # landmarks is partitioned into mutually exclusive subsets. The value specified here 
    # should exactly divide the landmark count. For example, when num_landmarks = 1000 and 
    # block_size = 125, we get 1000/125 = 8 subsets of landmarks.
    block_size = 125

    # Specify which subset you want to train the model for. For example, when 
    # num_landmarks = 1000 and block_size = 125, then subset_index = 0 indicates that the 
    # range of indices of landmarks in the subset would be [0, 125]. If subset_index = 1,
    # then the range of indices would be [125, 250].
    subset_index = 0
    
    # Format the experiment name.
    experiment_name = '%s_%d-%d_v%d' % (scene_name, num_landmarks, block_size, version_no)

    # Format the model_dir string
    landmark_start_index = subset_index * block_size
    landmark_stop_index = (subset_index + 1) * block_size

    if landmark_start_index < 0 | landmark_stop_index > num_landmarks:
        raise Exception('landmark indices are outside valid range!')
    else:
        tmp = '%s-%03d-%03d' % (scene_name, landmark_start_index, landmark_stop_index)
        model_dir = os.path.join(output_dir, experiment_name, tmp)

        # Create the model_dir folder.
        os.makedirs(model_dir, exist_ok=True)

        # Create the command line string for the training job.
        cmd = 'python ./local_training.py'
        cmd += ' --dataset_dir %s' % dataset_dir
        cmd += ' --scene_id %s' % scene_name
        cmd += ' --experiment_file %s.txt' % experiment_name
        cmd += ' --num_landmarks %d' % num_landmarks
        cmd += ' --block_size %d' % block_size
        cmd += ' --landmark_config %s' % landmark_config
        cmd += ' --visibility_config %s' % visibility_config
        cmd += ' --subset_index %d' % subset_index
        cmd += ' --output_dir %s' % output_dir
        cmd += ' --model_dir %s' % model_dir
        cmd += ' --training_batch_size %d' % training_batch_size
        cmd += ' --output_downsample %d' % output_downsample 
        cmd += ' --num_epochs %d' % num_epochs

        # Launch training
        os.system(cmd)
