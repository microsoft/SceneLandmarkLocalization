import copy
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
    N = pred_heatmap.shape[0]
    G_p_f = np.zeros((3, N))
    C_b_f_hm = np.zeros((3, N))
    weights = np.zeros(N)
    validIdx = 0

    pixel_error = []
    angular_error = []
    for l in range(N):
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

            
            pixel_error.append(np.linalg.norm(landmark2d[:2, l] -
                                              opt.output_downsample * np.array([refine_x, refine_y])))

            pred_bearing = K_inv @ np.array([refine_x, refine_y, 1])
            pred_bearing = pred_bearing / np.linalg.norm(pred_bearing)
            gt_bearing = C_b_f_gt[:, l]
            gt_bearing = gt_bearing / np.linalg.norm(gt_bearing)
            angular_error_batch = np.arccos(
                np.clip(pred_bearing @ gt_bearing, a_min=-1, a_max=1)) * 180 / np.pi
            
            angular_error.append(angular_error_batch)

            weights[validIdx] = max_pred_heatmap_l
            C_b_f_hm[:, validIdx] = pred_bearing
            G_p_f[:, validIdx] = landmark3d[:, l]
            validIdx += 1

    return G_p_f[:, :validIdx], C_b_f_hm[:, :validIdx], weights[:validIdx], np.asarray(pixel_error), np.asarray(angular_error)


def compute_pose(G_p_f, C_b_f_hm, weights, minimal_tight_thr, opt_tight_thr):

    Ndetected_landmarks = C_b_f_hm.shape[1]

    if Ndetected_landmarks >= 4:
        ## P3P ransac
        C_T_G_hat, PnP_inlier = P3PKe_Ransac(G_p_f, C_b_f_hm, weights,
                                             thres=minimal_tight_thr)
        
        if np.sum(PnP_inlier) >= 4:
            C_T_G_opt = RunPnPNL(C_T_G_hat,
                                 G_p_f[:, PnP_inlier], 
                                 C_b_f_hm[:, PnP_inlier],
                                 weights[PnP_inlier],
                                 cutoff=opt_tight_thr)
            return np.sum(PnP_inlier), C_T_G_opt

    return 0, None


def inference(opt, minimal_tight_thr=1e-2, opt_tight_thr=5e-3, mode='test'):

    PRETRAINED_MODEL = opt.pretrained_model

    device = opt.gpu_device

    test_dataset = Indoor6(landmark_idx=np.arange(opt.landmark_indices[0], opt.landmark_indices[-1]),
                           scene_id=opt.scene_id,
                           mode=mode,
                           root_folder=opt.dataset_folder,
                           input_image_downsample=2,
                           landmark_config=opt.landmark_config,
                           visibility_config=opt.visibility_config,
                           skip_image_index=1)

    test_dataloader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

    num_landmarks = test_dataset.landmark.shape[1]
    landmark_data = test_dataset.landmark

    cnns = []
    nLandmarks = opt.landmark_indices
    num_landmarks = opt.landmark_indices[-1] - opt.landmark_indices[0]

    for idx, pretrained_model in enumerate(PRETRAINED_MODEL):
        if opt.model == 'efficientnet':
            cnn = EfficientNetSLD(num_landmarks=nLandmarks[idx+1]-nLandmarks[idx], output_downsample=opt.output_downsample).to(device=device)

        cnn.load_state_dict(torch.load(pretrained_model))
        cnn = cnn.to(device=device)
        cnn.eval()
        
        # Adding pretrained model
        cnns.append(cnn)

    peak_threshold = 2e-1
    img_id = 0

    METRICS_LOGGING = {'image_name': '',
                       'angular_error': [],
                       'pixel_error': [],
                       'rot_err_all': 180.,
                       'trans_err_all': 180.,
                       'heatmap_peak': 0.0,
                       'ndetected': 0,                       
                       }
    test_image_logging = []    

    with torch.no_grad():

        ## Only works for indoor-6
        indoor6W = 640 // opt.output_downsample
        indoor6H = 352 // opt.output_downsample
        HH, WW = torch.meshgrid(torch.arange(indoor6H), torch.arange(indoor6W))
        WW = WW.reshape(1, 1, indoor6H, indoor6W).to('cuda')
        HH = HH.reshape(1, 1, indoor6H, indoor6W).to('cuda')

        for idx, batch in enumerate(tqdm(test_dataloader)):

            image = batch['image'].to(device=device)
            B, _, H, W = image.shape

            K_inv = batch['inv_intrinsics'].to(device=device)
            C_T_G_gt = batch['pose_gt'].cpu().numpy()

            landmark2d = batch['intrinsics'] @ batch['landmark3d'].reshape(B, 3, num_landmarks)
            landmark2d /= landmark2d[:, 2:].clone()
            landmark2d = landmark2d.numpy()


            pred_heatmap = []
            for cnn in cnns:
                pred = cnn(image)
                pred_heatmap.append(pred['1'])

            pred_heatmap = torch.cat(pred_heatmap, axis=1)
            pred_heatmap *= (pred_heatmap > peak_threshold).float()

            K_inv[:, :, :2] *= opt.output_downsample

            ## Compute 2D location of landmarks
            P = torch.max(torch.max(pred_heatmap, dim=3)[0], dim=2)[0]
            pred_normalized_heatmap = pred_heatmap / (torch.sum(pred_heatmap, axis=(2, 3), keepdim=True) + 1e-4)
            projx = torch.sum(WW * pred_normalized_heatmap, axis=(2, 3)).reshape(B, 1, num_landmarks)
            projy = torch.sum(HH * pred_normalized_heatmap, axis=(2, 3)).reshape(B, 1, num_landmarks)
            xy1 = torch.cat((projx, projy, torch.ones_like(projx)), axis=1)
            uv1 = K_inv @ xy1
            C_B_f = uv1 / torch.sqrt(torch.sum(uv1 ** 2, axis=1, keepdim=True))
            C_B_f = C_B_f.cpu().numpy()
            P = P.cpu().numpy()
            xy1 = xy1.cpu().numpy()

            ## Compute error
            for b in range(B):
                # G_p_f, C_b_f, weights, pixel_error, angular_error = compute_2d3d(
                #                                         opt, pred_heatmap[b].cpu().numpy(), 
                #                                         peak_threshold, landmark2d[b], landmark_data,
                #                                         batch['landmark3d'][b].cpu().numpy(),
                #                                         H_hm, W_hm, K_inv[b].cpu().numpy())

                Pb = P[b]>peak_threshold
                G_p_f = landmark_data[:, Pb]
                C_b_f = C_B_f[b][:, Pb]                
                weights = P[b][Pb]                
                xy1b = xy1[b][:2, Pb]                

                pnp_inlier, C_T_G_hat = compute_pose(G_p_f, C_b_f, weights,
                                                    minimal_tight_thr, opt_tight_thr,
                                                    img_id, opt.output_folder)
                
                rot_err, trans_err = 180., 1800.
                if pnp_inlier >= 4:
                    rot_err, trans_err = compute_error(C_T_G_gt[b][:3, :3], C_T_G_gt[b][:3, 3],
                                                       C_T_G_hat[:3, :3], C_T_G_hat[:3, 3])
                
                ## Logging information                
                pixel_error = np.linalg.norm(landmark2d[b][:2, Pb] - opt.output_downsample * xy1b, axis=0)                
                C_b_f_gt = batch['landmark3d'][b]
                C_b_f_gt = torch.nn.functional.normalize(C_b_f_gt, dim=0).cpu().numpy()
                angular_error = np.arccos(np.clip(np.sum(C_b_f * C_b_f_gt[:, Pb], axis=0), -1, 1)) * 180. / np.pi

                m = copy.deepcopy(METRICS_LOGGING)
                m['image_name'] = test_dataset.image_files[img_id]                
                m['pixel_error'] = pixel_error 
                m['angular_error'] = angular_error
                m['heatmap_peak'] = weights
                m['rot_err_all'] = np.array([rot_err])
                m['trans_err_all'] = np.array([trans_err])
                
                test_image_logging.append(m)
                
                img_id += 1


    metrics_output = {'angular_error': [], 
                      'pixel_error': [], 
                      'heatmap_peak': [], 
                      'rot_err_all': [], 
                      'trans_err_all': []}
    
    for k in metrics_output:        
        for imgdata in test_image_logging:            
            metrics_output[k].append(imgdata[k])
        metrics_output[k] = np.concatenate(metrics_output[k])

    metrics_output['r5'] = np.sum(metrics_output['rot_err_all'] < 5) / len(test_dataset)
    metrics_output['r10'] = np.sum(metrics_output['rot_err_all'] < 10) / len(test_dataset)
    metrics_output['p5'] = np.sum(metrics_output['trans_err_all'] < 0.05) / len(test_dataset)
    metrics_output['p10'] = np.sum(metrics_output['trans_err_all'] < 0.1) / len(test_dataset)
    metrics_output['r1p1'] = np.sum((metrics_output['rot_err_all'] < 1) * (metrics_output['trans_err_all'] < 0.01))/len(test_dataset)
    metrics_output['r2p2'] = np.sum((metrics_output['rot_err_all'] < 2) * (metrics_output['trans_err_all'] < 0.02))/len(test_dataset)
    metrics_output['r5p5'] = np.sum((metrics_output['rot_err_all'] < 5) * (metrics_output['trans_err_all'] < 0.05))/len(test_dataset)
    metrics_output['r10p10'] = np.sum((metrics_output['rot_err_all'] < 10) * (metrics_output['trans_err_all'] < 0.1)) / len(test_dataset)
    metrics_output['median_rot_error'] = np.median(metrics_output['rot_err_all'])
    metrics_output['median_trans_error'] = np.median(metrics_output['trans_err_all'])


    return metrics_output
