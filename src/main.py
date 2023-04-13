import argparse
from inference import *
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scene Landmark Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_folder', type=str, required=True,
        help='Root directory, where all data is stored')
    parser.add_argument(
        '--output_folder', type=str, required=True,
        help='Output folder')
    parser.add_argument(
        '--landmark_config', type=str, default='landmarks/landmarks-300',
        help='File containing scene-specific 3D landmarks.')
    parser.add_argument(
        '--visibility_config', type=str, default='landmarks/visibility_aug-300',
        help='File containing information about visibility of landmarks in cameras associated with training set.')
    parser.add_argument(
        '--scene_id', type=str, default='scene6',
        help='Scene id')
    parser.add_argument(
        '--model', type=str, default='efficientnet',
        help='Network architecture backbone.')
    parser.add_argument(
        '--output_downsample', type=int, default=4,
        help='Down sampling factor for output resolution')
    parser.add_argument(
        '--gpu_device', type=str, default='cuda:0',
        help='GPU device')
    parser.add_argument(
        '--pretrained_model', type=str, default='',
        help='Pretrained detector model')
    parser.add_argument(
        '--num_epochs', type=int, default=200,
        help='Number of training epochs.')
    parser.add_argument(
        '--action', type=str, default='test',
        help='train/train_patches/test')
    parser.add_argument(
        '--training_batch_size', type=int, default=8,
        help='Batch size used during training.')

    opt = parser.parse_args()

    print('scene_id: ', opt.scene_id)
    print('action: ', opt.action)
    print('training_batch_size: ', opt.training_batch_size)
    print('output downsample: ', opt.output_downsample)

    if opt.action == 'train':
        train(opt)
    elif opt.action == 'train_patches':
        train_patches(opt)
        opt.pretrained_model = opt.output_folder + '/model-best_median.ckpt'
        eval_stats = inference(opt, minimal_tight_thr=1e-3, opt_tight_thr=1e-3)
        print("{:>10} {:>30} {:>30} {:>20}".format('Scene ID',
                                                   'Median trans error (cm)',
                                                   'Median rotation error (deg)',
                                                   'Recall 5cm5deg (%)'))
        print("{:>10} {:>30.4} {:>30.4} {:>20.2%}".format(opt.scene_id,
                                                          100. * eval_stats['median_trans_error'],
                                                          eval_stats['median_rot_error'],
                                                          eval_stats['r5p5']))
    elif opt.action == 'test':
        if opt.scene_id == 'all':
            eval_stats = {}
            pretrained_folder = opt.pretrained_model
            output_folder = opt.output_folder
            for scene_id in ['1', '2a', '3', '4a', '5', '6']:
                opt.scene_id = 'scene' + scene_id
                opt.pretrained_model = pretrained_folder + 'scene%s.ckpt' % scene_id
                opt.output_folder = os.path.join(output_folder, 'scene' + scene_id)
                eval_stats[opt.scene_id] = inference(opt, minimal_tight_thr=1e-3, opt_tight_thr=1e-3)

            print("{:>10} {:>30} {:>30} {:>20}".format('Scene ID',
                                                       'Median trans error (cm)',
                                                       'Median rotation error (deg)',
                                                       'Recall 5cm5deg (%)'))
            for x in eval_stats:
                print("{:>10} {:>30.4} {:>30.4} {:>20.2%}".format(x,
                                                                  100. * eval_stats[x]['median_trans_error'],
                                                                  eval_stats[x]['median_rot_error'],
                                                                  eval_stats[x]['r5p5']))
        else:
            eval_stats = inference(opt, minimal_tight_thr=1e-3, opt_tight_thr=1e-3)
            print("{:>10} {:>30} {:>30} {:>20}".format('Scene ID',
                                                       'Median trans error (cm)',
                                                       'Median rotation error (deg)',
                                                       'Recall 5cm5deg (%)'))
            print("{:>10} {:>30.4} {:>30.4} {:>20.2%}".format(opt.scene_id,
                                                       100. * eval_stats['median_trans_error'],
                                                       eval_stats['median_rot_error'],
                                                       eval_stats['r5p5']))
