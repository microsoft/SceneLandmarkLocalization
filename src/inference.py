import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.indoor6 import Indoor6
from models.efficientlitesld import EfficientNetSLD
from utils.pnp import *

def compute_error(C_R_G, C_t_G, C_R_G_hat, C_t_G_hat):

    rot_err = 180 / np.pi * np.arccos(np.clip(0.5 * (np.trace(C_R_G.T @ C_R_G_hat) - 1.0), a_min=-1., a_max=1.))
    trans_err = np.linalg.norm(C_R_G_hat.T @ C_t_G_hat - C_R_G.T @ C_t_G)

    return rot_err, trans_err

def compute_2d3d(opt, pred_heatmap, peak_threshold, landmark2d, landmark3d, C_b_f_gt, H_hm, W_hm, K_inv,
                 METRICS_LOGGING=None):
    G_p_f = []
    C_b_f_hm = []
    weights = []

    for l in range(pred_heatmap.shape[0]):
        pred_heatmap_l = pred_heatmap[l]
        max_pred_heatmap_l = np.max(pred_heatmap_l)

        if max_pred_heatmap_l > peak_threshold:
            peak_yx = np.unravel_index(np.argmax(pred_heatmap_l), np.array(pred_heatmap_l).shape)
            peak_yx = np.array(peak_yx)

            # Patch size extraction
            P = int(min(1+2*np.min(np.array([peak_yx[0], H_hm-1.0-peak_yx[0], peak_yx[1], W_hm-1.0-peak_yx[1]])),
                        1+64//opt.output_downsample))

            patch_peak_yx = pred_heatmap_l[peak_yx[0] - P // 2:peak_yx[0] + P // 2 + 1,
                            peak_yx[1] - P // 2:peak_yx[1] + P // 2 + 1]
            xx_patch, yy_patch = np.meshgrid(np.arange(peak_yx[1] - P // 2, peak_yx[1] + P // 2 + 1, 1),
                                             np.arange(peak_yx[0] - P // 2, peak_yx[0] + P // 2 + 1, 1))

            refine_y = np.sum(patch_peak_yx * yy_patch) / np.sum(patch_peak_yx)
            refine_x = np.sum(patch_peak_yx * xx_patch) / np.sum(patch_peak_yx)

            if METRICS_LOGGING is not None:
                METRICS_LOGGING['pixel_error'].append(np.linalg.norm(landmark2d[:2, l] -
                                              opt.output_downsample * np.array([refine_x, refine_y])))

            pred_bearing = K_inv @ np.array([refine_x, refine_y, 1])
            pred_bearing = pred_bearing / np.linalg.norm(pred_bearing)
            gt_bearing = C_b_f_gt[:, l]
            gt_bearing = gt_bearing / np.linalg.norm(gt_bearing)
            angular_error_batch = np.arccos(
                np.clip(pred_bearing @ gt_bearing, a_min=-1, a_max=1)) * 180 / np.pi
            if METRICS_LOGGING is not None:
                METRICS_LOGGING['angular_error'].append(angular_error_batch)

            weights.append(max_pred_heatmap_l)
            C_b_f_hm.append(pred_bearing.reshape(3, 1))
            G_p_f.append(landmark3d[l].reshape(3, 1))

    return G_p_f, C_b_f_hm, weights


def compute_pose(G_p_f, C_b_f_hm, weights, minimal_tight_thr, opt_tight_thr, img_id, OUTPUT_FOLDER):
    Ndetected_landmarks = len(C_b_f_hm)

    ### Saving 2D-3D correspondences
    if Ndetected_landmarks > 0:
        C_b_f_hm = np.concatenate(C_b_f_hm, axis=1)
        G_p_f = np.concatenate(G_p_f, axis=1)
        weights = np.asarray(weights)

        if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'sld2d3d')):
            os.makedirs(os.path.join(OUTPUT_FOLDER, 'sld2d3d'))

        np.savetxt('%s/sld2d3d/%06d.txt' % (OUTPUT_FOLDER, img_id),
                   np.concatenate((C_b_f_hm, G_p_f), axis=0))
    else:
        C_b_f_hm = None
        G_p_f = None
        weights = None


    if Ndetected_landmarks >= 4:
        ## P3P ransac
        C_T_G_hat, PnP_inlier = P3PKe_Ransac(G_p_f, C_b_f_hm, weights,
                                             thres=minimal_tight_thr)
        # print('inlier: ', np.sum(PnP_inlier))
        if np.sum(PnP_inlier) >= 4:
            C_T_G_opt = RunPnPNL(C_T_G_hat,
                                 G_p_f[:, PnP_inlier], C_b_f_hm[:, PnP_inlier],
                                 weights[PnP_inlier],
                                 cutoff=opt_tight_thr)
            return np.sum(PnP_inlier), C_T_G_opt

    return 0, None

def inference(opt, minimal_tight_thr=1e-2, opt_tight_thr=5e-3, mode='test'):

    PRETRAINED_MODEL = opt.pretrained_model

    if not os.path.exists(PRETRAINED_MODEL):
        print(PRETRAINED_MODEL)
        print('ckpt path not exist')
        exit(1)

    device = opt.gpu_device

    test_dataset = Indoor6(landmark_idx=-1,
                           scene_id=opt.scene_id,
                           mode=mode,
                           root_folder=opt.dataset_folder,
                           input_image_downsample=2,
                           landmark_config=opt.landmark_config,
                           visibility_config=opt.visibility_config,
                           skip_image_index=1)

    test_dataloader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

    num_landmarks = test_dataset.landmarks.shape[0]
    landmark_data = test_dataset.landmarks

    if opt.model == 'efficientnet':
        cnn = EfficientNetSLD(num_landmarks=num_landmarks, output_downsample=opt.output_downsample).to(device=device)


    cnn.load_state_dict(torch.load(PRETRAINED_MODEL))
    cnn = cnn.to(device=device)
    cnn.eval()

    landmark_idx = np.arange(num_landmarks)
    peak_threshold = 1e-1
    img_id = 0
    C_T_G_all = []

    METRICS_LOGGING = {'angular_error': [],
                       'pixel_error': [],
                       'rot_err_all': [],
                       'trans_err_all': [],
                       'ndetected': 0
                       }

    with torch.no_grad():

        for idx, batch in enumerate(tqdm(test_dataloader)):

            image = batch['image'].to(device=device)

            B, _, H, W = image.shape

            K_inv = batch['inv_intrinsics']

            pred = cnn(image)
            pred_heatmap = pred['1']
            pred_heatmap *= (pred_heatmap > 0).float()

            K_inv[:, :, :2] *= opt.output_downsample

            H_hm = H // opt.output_downsample
            W_hm = W // opt.output_downsample

            pred_heatmap = pred_heatmap.cpu().numpy()
            K_inv = K_inv.cpu().numpy()
            C_T_G_gt = batch['pose_gt'].cpu().numpy()

            landmark2d = batch['intrinsics'] @ batch['landmark3d'][:, :, landmark_idx].reshape(-1, 3, len(landmark_idx))
            landmark2d /= landmark2d[:, 2:].clone()
            landmark2d = landmark2d.numpy()

            ## Compute error
            for b in range(B):
                G_p_f, C_b_f_hm, weights = compute_2d3d(opt, pred_heatmap[b], peak_threshold,
                                                        landmark2d[b], landmark_data,
                                                        batch['landmark3d'][b].cpu().numpy(),
                                                        H_hm, W_hm, K_inv[b], METRICS_LOGGING)

                pnp_inlier, C_T_G_hat = compute_pose(G_p_f, C_b_f_hm, weights,
                                                    minimal_tight_thr, opt_tight_thr,
                                                    img_id, opt.output_folder)
                if pnp_inlier >= 4:
                    rot_err, trans_err = compute_error(C_T_G_gt[b][:3, :3], C_T_G_gt[b][:3, 3],
                                                       C_T_G_hat[:3, :3], C_T_G_hat[:3, 3])
                    C_T_G_all.append(C_T_G_hat[:3].reshape(1, 3, 4))

                    METRICS_LOGGING['rot_err_all'].append(rot_err)
                    METRICS_LOGGING['trans_err_all'].append(trans_err)
                    METRICS_LOGGING['ndetected'] += 1
                else:
                    ## Invalid data to prevent stalling at early epoch
                    C_T_G_all.append(np.eye(4)[:3].reshape(1, 3, 4))
                    METRICS_LOGGING['rot_err_all'].append(180.)
                    METRICS_LOGGING['trans_err_all'].append(100.)
                    METRICS_LOGGING['ndetected'] += 1

                img_id += 1

    if len(METRICS_LOGGING['rot_err_all']) > 0:
        rot_err_all = np.asarray(METRICS_LOGGING['rot_err_all'])
        trans_err_all = np.asarray(METRICS_LOGGING['trans_err_all'])
        METRICS_LOGGING['r5'] = np.sum(rot_err_all < 5) / len(test_dataset)
        METRICS_LOGGING['r10'] = np.sum(rot_err_all < 10) / len(test_dataset)
        METRICS_LOGGING['r5p5'] = np.sum((rot_err_all < 5) * (trans_err_all < 0.05))/len(test_dataset)
        METRICS_LOGGING['r10p10'] = np.sum((rot_err_all < 10) * (trans_err_all < 0.1)) / len(test_dataset)
        METRICS_LOGGING['median_rot_error'] = np.median(rot_err_all)
        METRICS_LOGGING['median_trans_error'] = np.median(trans_err_all)

        np.save('%s/rot_error_all.npy' % opt.output_folder, rot_err_all)
        np.save('%s/trans_error_all.npy' % opt.output_folder, trans_err_all)
        np.save('%s/C_T_G_pred_all.npy' % opt.output_folder, np.concatenate(C_T_G_all, axis=0))

    ## Invalid data to prevent stalling at early epoch
    if len(METRICS_LOGGING['pixel_error']) == 0:
        METRICS_LOGGING['pixel_error'] = [1800.]
        METRICS_LOGGING['angular_error'] = [180.]

    return METRICS_LOGGING
