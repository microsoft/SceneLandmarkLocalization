import argparse
import copy
import numpy as np
import os
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(sys.path[0], '..'))

from dataloader.indoor6 import Indoor6
from models.efficientlitesld import EfficientNetSLD
from utils.pnp import *

from PIL import Image

# import open3d as o3d

def load_landmark_files(landmark_path, visibility_path):
    landmark_file = open(landmark_path, 'r')
    num_landmark = int(landmark_file.readline())
    landmark = []
    for l in range(num_landmark):
        pl = landmark_file.readline().split()
        pl = np.array([float(pl[i]) for i in range(len(pl))])
        landmark.append(pl)
    landmark = np.asarray(landmark)[:, 1:].T

    visibility = np.loadtxt(visibility_path)

    return landmark, visibility


def load_landmark_visibility_files(landmark_path, visibility_path):
    landmark_file = open(landmark_path, 'r')
    num_landmark = int(landmark_file.readline())
    landmark = []
    for l in range(num_landmark):
        pl = landmark_file.readline().split()
        pl = np.array([float(pl[i]) for i in range(len(pl))])
        landmark.append(pl)
    landmark = np.asarray(landmark)[:, 1:].T

    visibility = np.loadtxt(visibility_path)

    return landmark, visibility


def visualize_keypoint_np(image_, y, x, kp_color):
    image = image_.copy()
    if np.sum(kp_color) == 255:
        square_size = 5
    else:
        square_size = 3
    for c in range(3):
        image[y - square_size:y + square_size, x - square_size:x + square_size, c] = kp_color[c]

    return image


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


def compute_pose(G_p_f, C_b_f_hm, weights, minimal_tight_thr, opt_tight_thr, img_id, OUTPUT_FOLDER):
    Ndetected_landmarks = C_b_f_hm.shape[1]

    # ### Saving 2D-3D correspondences
    # if Ndetected_landmarks > 0:
    #     if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'sld2d3d')):
    #         os.makedirs(os.path.join(OUTPUT_FOLDER, 'sld2d3d'))

    #     np.savetxt('%s/sld2d3d/%06d.txt' % (OUTPUT_FOLDER, img_id),
    #                np.concatenate((C_b_f_hm, G_p_f), axis=0))
    # else:
    #     C_b_f_hm = None
    #     G_p_f = None
    #     weights = None


    if Ndetected_landmarks >= 4:
        ## P3P ransac
        C_T_G_hat, PnP_inlier = P3PKe_Ransac(G_p_f, C_b_f_hm, weights,
                                             thres=minimal_tight_thr)
        # print('inlier: ', np.sum(PnP_inlier))
        if np.sum(PnP_inlier) >= 4:
            # C_T_G_opt = PnP(C_T_G_hat, G_p_f[:, PnP_inlier], C_b_f_hm[:, PnP_inlier], weights[PnP_inlier])
            C_T_G_opt = RunPnPNL(C_T_G_hat,
                                 G_p_f[:, PnP_inlier], C_b_f_hm[:, PnP_inlier],
                                 weights[PnP_inlier],
                                 cutoff=opt_tight_thr)
            return np.sum(PnP_inlier), C_T_G_opt, PnP_inlier

    return 0, None, np.empty((0))


def select_additional_landmarks(opt, minimal_tight_thr=1e-2, opt_tight_thr=5e-3, mode='test', peak_threshold=0.6):

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
    
    landmark_data = test_dataset.landmark

    cnns = []
    nLandmarks = opt.landmark_indices
    num_landmarks = opt.landmark_indices[-1]

    if len(PRETRAINED_MODEL) == 0:
        use_gt_2d3d = True
    else:
        use_gt_2d3d = False
        for idx, pretrained_model in enumerate(PRETRAINED_MODEL):
            if opt.model == 'efficientnet':
                cnn = EfficientNetSLD(num_landmarks=nLandmarks[idx+1]-nLandmarks[idx], output_downsample=opt.output_downsample).to(device=device)

            cnn.load_state_dict(torch.load(pretrained_model))
            cnn = cnn.to(device=device)
            cnn.eval()
            
            # Adding pretrained model
            cnns.append(cnn)
    
    img_id = 0

    METRICS_LOGGING = {'image_name': '',
                       'angular_error': [],
                       'pixel_error': [],
                       'rot_err_all': 180.,
                       'trans_err_all': 180.,
                       'heatmap_peak': 0.0,
                       'ndetected': 0,              
                       'pnp_inlier': np.zeros(num_landmarks),
                       'pixel_inlier_error': np.array([1800.]),
                       }
    test_image_logging = []    



    LANDMARKS_METRICS_LOGGING = {'image_name': [],
                                'angular_error': [],
                                'pixel_error': [],
                                'heatmap_peak': 0.0,
                                'ndetected': 0,                       
                                }
    test_landmarks_logging = [copy.deepcopy(LANDMARKS_METRICS_LOGGING) for _ in range(num_landmarks)]
    print(len(test_landmarks_logging))

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
                # xy1b = xy1[b][:2, Pb]

                pnp_inlier, C_T_G_hat, pnp_inlier_mask = compute_pose(G_p_f, C_b_f, weights,
                                                                        minimal_tight_thr, opt_tight_thr,
                                                                        img_id, opt.output_folder)
                
                rot_err, trans_err = 180., 1800.
                if pnp_inlier >= 4:
                    rot_err, trans_err = compute_error(C_T_G_gt[b][:3, :3], C_T_G_gt[b][:3, 3],
                                                       C_T_G_hat[:3, :3], C_T_G_hat[:3, 3])
                
                ## Logging information                
                pixel_error = np.linalg.norm(landmark2d[b][:2, Pb] - opt.output_downsample * xy1[b][:2, Pb], axis=0)
                C_b_f_gt = batch['landmark3d'][b]
                C_b_f_gt = torch.nn.functional.normalize(C_b_f_gt, dim=0).cpu().numpy()
                angular_error = np.arccos(np.clip(np.sum(C_b_f * C_b_f_gt[:, Pb], axis=0), -1, 1)) * 180. / np.pi                

                m = copy.deepcopy(METRICS_LOGGING)
                m['image_name'] = test_dataset.image_files[img_id]
                m['rgb'] = batch['image'][b].cpu().numpy().transpose(1, 2, 0)
                m['pixel_error'] = pixel_error 
                m['angular_error'] = angular_error
                m['heatmap_peak'] = P[b]
                m['pixel_detected'] = xy1[b] * opt.output_downsample
                m['pixel_gt'] = landmark2d[b]
                m['visibility_gt'] = batch['visibility'][b] > 0.5
                m['rot_err_all'] = np.array([rot_err])
                m['trans_err_all'] = np.array([trans_err])
                m['K'] = batch['intrinsics'][b].cpu().numpy()
                m['C_T_G_gt'] = C_T_G_gt[b]

                if len(pnp_inlier_mask):
                    m['pnp_inlier'][Pb] = pnp_inlier_mask
                    pixel_inlier_error = np.linalg.norm(landmark2d[b][:2, m['pnp_inlier']==1] - 
                                                        opt.output_downsample * xy1[b][:2, m['pnp_inlier']==1], axis=0)
                    m['pixel_inlier_error'] = pixel_inlier_error
                
                test_image_logging.append(m)

                for l in range(num_landmarks):
                    if batch['visibility'][b, l] > 0.5:
                        test_landmarks_logging[l]['image_name'].append(test_dataset.image_files[img_id])
                        if P[b, l]:
                            test_landmarks_logging[l]['pixel_error'].append(np.linalg.norm(landmark2d[b][:2, l] - 
                                                                                           opt.output_downsample * xy1[b][:2, l], axis=0))
                        else:
                            test_landmarks_logging[l]['pixel_error'].append(1e3)

                test_landmarks_logging.append(m)
                
                img_id += 1


        ## 2D visualization of images
        # test_image_logging.sort(key = lambda x: x['trans_err_all'][0])
        # for m in test_image_logging:
        #     print(m['image_name'], ': ', m['trans_err_all'][0])
        #     img = np.array(m['rgb'] * 255, dtype=np.uint8)
        #     for l in range(len(m['visibility_gt'])):
        #         if m['pnp_inlier'][l]:
        #             img = visualize_keypoint_np(img, 
        #                                         int(m['pixel_detected'][1, l]),
        #                                         int(m['pixel_detected'][0, l]),
        #                                         np.array([0., 255., 0.]))
        #         if m['visibility_gt'][l]:                                        
        #             img = visualize_keypoint_np(img, 
        #                                         int(m['pixel_gt'][1, l]),
        #                                         int(m['pixel_gt'][0, l]),
        #                                         np.array([200., 0., 0.]))
                    
        #     Image.fromarray(img).save('%s/%2.2f_%2.2f_%s.jpg' % (opt.output_folder, 
        #                                                          m['trans_err_all'][0], 
        #                                                          np.mean(m['pixel_inlier_error']),
        #                                                          m['image_name']))


        ###########################################################################################
        ############################ Extra landmark selection analysis ############################
        ###########################################################################################

        ## Some more additional points to improve wacky poses
        # lm_file = os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks/landmarks-2000v8.txt')
        # vis_file = os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks/visibility-2000v8_depth_normal.txt')
        # full_landmarks, full_vis = load_landmark_files(lm_file, vis_file)
        # full_landmarks, full_vis = full_landmarks[:, :200], full_vis[:200]


        ## Colmap file
        from utils.read_write_models import read_model
        cameras, images, points = read_model(os.path.join(opt.dataset_folder, 'indoor6-colmap/%s/sparse/0' % opt.scene_id), ext='.bin')
        indoor6_name_2to_colmap_index = {}
        for k in images:
            indoor6_name_2to_colmap_index[images[k].name] = k


        ## Images with bad poses
        ## Adding more landmarks on top
        ## For each test image, pick 10 landmarks that have highest score, 
        ## adding to the high accuracy of camera position triangulation
        additional_landmarks = set()
        for idx, m in enumerate(test_image_logging):
            if m['trans_err_all'][0] > 1.0:

                ## We want to add unseen points that isn't near the 2D detected points
                img_vis_id = test_dataset.image_files.index(m['image_name'])

                # print('---------------------------')
                # print(m['image_name'])                
                # print(test_dataset.original_image_name(img_vis_id))
                # print(images[indoor6_name_2to_colmap_index[test_dataset.original_image_name(img_vis_id)]].name)

                xys = images[indoor6_name_2to_colmap_index[test_dataset.original_image_name(img_vis_id)]].xys
                point3dids = images[indoor6_name_2to_colmap_index[test_dataset.original_image_name(img_vis_id)]].point3D_ids
                xys = xys[point3dids != -1]
                point3dids = point3dids[point3dids != -1]

                img = np.array(m['rgb'] * 255, dtype=np.uint8)
                
                for l in range(xys.shape[0]):
                    img = visualize_keypoint_np(img,
                                            int(xys[l, 1] * 352 / 720),
                                            int(xys[l, 0] * 0.5),
                                            np.array([200., 0., 0.]))
                    
                    xy_scaled = np.array([xys[l, 0] * 0.5, xys[l, 1] * 352 / 720])

                    if np.sum(m['pnp_inlier']) > 0:
                        dist_other_2d_kpts = np.linalg.norm(xy_scaled.reshape(2, 1) - m['pixel_detected'][:2, m['pnp_inlier']==1], axis=0)
                        
                        if np.min(dist_other_2d_kpts) > 20: # 20 pixels, 1/10 of the image size
                            additional_landmarks.add(point3dids[l])
                    else:
                        additional_landmarks.add(point3dids[l])
                    
                # for l in range(len(m['visibility_gt'])):
                #     if m['pnp_inlier'][l]:
                #         img = visualize_keypoint_np(img, 
                #                                     int(m['pixel_detected'][1, l]),
                #                                     int(m['pixel_detected'][0, l]),
                #                                     np.array([0., 255., 0.]))
                    
                # visible_landmarks_in_the_next_1k = np.where(full_vis[:, test_dataset.image_indices[img_vis_id]] == 1)[0]
                # visible_landmarks_in_the_next_1k += 1000
                # print(visible_landmarks_in_the_next_1k)
                # img = np.array(m['rgb'] * 255, dtype=np.uint8)
                # for l in visible_landmarks_in_the_next_1k:
                #     pix = m['K'] @ (m['C_T_G_gt'][:3, :3] @ full_landmarks[:, l] + m['C_T_G_gt'][:3, 3])
                #     img = visualize_keypoint_np(img,
                #                                 int(pix[1] / pix[2]),
                #                                 int(pix[0] / pix[2]),
                #                                 np.array([200., 0., 0.]))

                
                ## Re-do pnp, save new image with new translation error


                # Image.fromarray(img).save('%s/%2.2f_%2.2f_%s_after.jpg' % (opt.output_folder, 
                #                                                             m['trans_err_all'][0], 
                #                                                             np.mean(m['pixel_inlier_error']),
                #                                                             m['image_name']))
        

        ### Given additional set of landmarks, re-run the landmark selection to get 200 points
        from landmark_selection import ComputePerPointAngularSpan, ComputePerPointDepth, SaveLandmarksAndVisibilityMask

        ### Adding a bank of new points
        numPoints3D = len(additional_landmarks)
        points3D_ids = np.zeros(numPoints3D)
        points3D_scores = np.zeros(numPoints3D)
        points3D_depth = np.zeros(numPoints3D)
        points3D_tracklength = np.zeros(numPoints3D)
        points3D_anglespan = np.zeros(numPoints3D)

        validIdx = 0
        ## Compute score for each landmark    
        for i, k in enumerate(tqdm(additional_landmarks)):
            pointInGlobal = points[k].xyz
            image_ids = points[k].image_ids
            trackLength = len(image_ids)
                
            depthMean, depthStd = ComputePerPointDepth(pointInGlobal, image_ids, images)        
            # timespan = ComputePerPointTimeSpan(image_ids, images)
            anglespan = ComputePerPointAngularSpan(pointInGlobal, image_ids, images)

            if depthMean < 5.0 and trackLength > 5:
                depthScore = min(1.0, depthStd / depthMean) 
                trackLengthScore = 0.25 * np.log2(trackLength)
                
                points3D_depth[validIdx] = depthMean
                points3D_tracklength[validIdx] = trackLength
                points3D_anglespan[validIdx] = anglespan

                points3D_ids[validIdx] = k
                points3D_scores[validIdx] = depthScore + trackLengthScore + anglespan

                validIdx += 1

        points3D_depth = points3D_depth[:validIdx]
        points3D_tracklength = points3D_tracklength[:validIdx]
        points3D_anglespan = points3D_anglespan[:validIdx]
        point3dids = points3D_ids[:validIdx]
        points3D_scores = points3D_scores[:validIdx]

        print('Number of additional points: ', validIdx)
        print('[Depth mean] Max: %2.2f/Median: %2.2f/Mean: %2.2f/Min: %2.2f' 
              % (np.max(points3D_depth), np.median(points3D_depth), np.mean(points3D_depth), np.min(points3D_depth)))
        print('[Track length] Max: %2.2f/Median: %2.2f/Mean: %2.2f/Min: %2.2f' 
              % (np.max(points3D_tracklength), np.median(points3D_tracklength), np.mean(points3D_tracklength), np.min(points3D_tracklength)))
        print('[Angle span] Max: %2.2f/Median: %2.2f/Mean: %2.2f/Min: %2.2f' 
              % (np.max(points3D_anglespan), np.median(points3D_anglespan), np.mean(points3D_anglespan), np.min(points3D_anglespan)))
        

        num_selected_landmark = opt.num_landmarks
        ## Sort scores
        sorted_indices = np.argsort(points3D_scores)

        ## Greedy selection
        selected_landmarks = {'id': np.zeros(num_selected_landmark), 
                            'xyz': np.zeros((3, num_selected_landmark)), 
                            'score': np.zeros(num_selected_landmark)}

        ## Selecting first point
        selected_landmarks['id'][0] = points3D_ids[sorted_indices[-1]]
        selected_landmarks['xyz'][:, 0] = points[selected_landmarks['id'][0]].xyz
        selected_landmarks['score'][0] = points3D_scores[sorted_indices[-1]]

        nselected = 1
        radius = 5.0

        while nselected < num_selected_landmark:
            for i in reversed(sorted_indices):
                id = points3D_ids[i]
                xyz = points[id].xyz        

                if np.sum(np.linalg.norm(xyz.reshape(3, 1) - selected_landmarks['xyz'][:, :nselected], axis=0) < radius):
                    continue
                else:
                    selected_landmarks['id'][nselected] = id
                    selected_landmarks['xyz'][:, nselected] = xyz
                    selected_landmarks['score'][nselected] = points3D_scores[i]
                    nselected += 1

                if nselected == num_selected_landmark:
                    break
                
            radius *= 0.5

        ## Saving
        import pickle

        indoor6_images = pickle.load(open(os.path.join(opt.dataset_folder, '%s/train_test_val.pkl' % opt.scene_id), 'rb'))
        indoor6_imagename_to_index = {}

        for i, f in enumerate(indoor6_images['train']):
            image_name = open(os.path.join(opt.dataset_folder, 
                                        opt.scene_id, 'images', 
                                        f.replace('color.jpg', 
                                                    'intrinsics.txt'))).readline().split(' ')[-1][:-1]
            indoor6_imagename_to_index[image_name] = indoor6_images['train_idx'][i]
        
        for i, f in enumerate(indoor6_images['val']):
            image_name = open(os.path.join(opt.dataset_folder, 
                                        opt.scene_id, 'images', 
                                        f.replace('color.jpg', 
                                                    'intrinsics.txt'))).readline().split(' ')[-1][:-1]
            indoor6_imagename_to_index[image_name] = indoor6_images['val_idx'][i]

        for i, f in enumerate(indoor6_images['test']):
            image_name = open(os.path.join(opt.dataset_folder, 
                                        opt.scene_id, 'images', 
                                        f.replace('color.jpg', 
                                                    'intrinsics.txt'))).readline().split(' ')[-1][:-1]
            indoor6_imagename_to_index[image_name] = indoor6_images['test_idx'][i]

        num_images = len(indoor6_images['train']) + len(indoor6_images['val']) + len(indoor6_images['test'])
        
        SaveLandmarksAndVisibilityMask(selected_landmarks, points, images, indoor6_imagename_to_index, num_images, 
                                    os.path.join(opt.dataset_folder, opt.scene_id), 
                                    opt.landmark_config, opt.visibility_config, opt.output_format)
        

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
        '--num_landmarks', type=int, default=300,
        help='Number of selected landmarks.')
    parser.add_argument(
        '--output_format', type=str, default='',
        help='Landmark file output.')
    parser.add_argument(
        '--output_folder', type=str, required=True,
        help='Output folder')
    parser.add_argument(
        '--landmark_config', type=str, default='landmarks/landmarks-300',
        help='File containing scene-specific 3D landmarks.')
    parser.add_argument(
        '--landmark_indices', type=int, action='append',
        help = 'Landmark indices, specify twice',
        required=True)
    parser.add_argument(
        '--visibility_config', type=str, default='landmarks/visibility_aug-300',
        help='File containing information about visibility of landmarks in cameras associated with training set.')
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
        '--pretrained_model', type=str, action='append', default=[],
        help='Pretrained detector model')


    opt = parser.parse_args()
    select_additional_landmarks(opt, minimal_tight_thr=1e-3, opt_tight_thr=1e-3)