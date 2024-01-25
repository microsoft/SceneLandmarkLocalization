import os
import statistics as st

if __name__ == '__main__':

    # specify dataset path, location of checkpoints and the experiment name.
    checkpoint_dir = '../../data/checkpoints'
    dataset_dir = '../../data/indoor6'
    experiment = '1000-125_v10'

    # run inference for all six scenes of the indoor6 dataset
    for scene_name in ['scene1', 'scene2a', 'scene3', 'scene4a', 'scene5', 'scene6']:
        command = 'python .\local_inference.py --experiment_file %s_%s.txt --dataset_dir %s --checkpoint_dir %s' % (scene_name, experiment, dataset_dir, checkpoint_dir)
        os.system(command)

    # calculate metrics
    t1 = []
    t2 = []
    for scene_name in ['scene1', 'scene2a', 'scene3', 'scene4a', 'scene5', 'scene6']:
        subfolder = '%s_%s' % (scene_name, experiment)
        mfn = os.path.join(checkpoint_dir, subfolder, "metrics.txt")
        mfd = open(mfn, 'r')      
        idx = 0
        for line in mfd.readlines():
            if (idx % 2 == 0):
                t1.append(float(line))
            else:
                t2.append(float(line))
            idx+=1
        mfd.close();
    
    print(t1)
    print(t2)
    metricPcnt = 100.0 * st.fmean(t1)
    print('   mean = %s pcnt' % str(metricPcnt))
    print('   rate = %s imgs./sec.' % str(st.fmean(t2)))
    
    fname = 'RESULTS-%s.txt' % experiment  
    ffn = os.path.join(checkpoint_dir, fname)
    ffd = open(ffn, 'w')
    ffd.write(f"{metricPcnt}\n{st.fmean(t2)}\n")
    ffd.close();