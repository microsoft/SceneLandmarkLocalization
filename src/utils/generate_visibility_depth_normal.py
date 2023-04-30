import argparse
import copy
import fnmatch
import numpy as np
import open3d as o3d
import os
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloader.indoor6 import Indoor6

def extract(opt):

    DATASET_FOLDER = os.path.join(opt.dataset_folder)

    test_dataset = Indoor6(scene_id=opt.scene_id,
                         mode='all',
                         root_folder=DATASET_FOLDER,
                         input_image_downsample=1,
                         landmark_config=opt.landmark_config,
                         visibility_config=opt.visibility_config,
                         skip_image_index=1)

    test_dataloader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

    return test_dataloader, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scene Landmark Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_folder', type=str, required=False,
        help='Root directory, where all data is stored')
    parser.add_argument(
        '--output_folder', type=str, required=False,
        help='Output folder')
    parser.add_argument(
        '--landmark_config', type=str, default='landmarks/landmarks-300',
        help='Landmark configuration.')
    parser.add_argument(
        '--visibility_config', type=str, default='landmarks/visibility-300',
        help='Visibility configuration.')
    parser.add_argument(
        '--scene_id', type=str, default='scene1',
        help='Scene id')

    opt = parser.parse_args()
    monodepth_folder = os.path.join(opt.dataset_folder, opt.scene_id, 'depth')

    from read_write_models import *
    cameras, images, points = read_model(os.path.join(opt.dataset_folder, 'indoor6-colmap/%s/sparse/0' % opt.scene_id), ext='.bin')
    indoor6_name_2to_colmap_index = {}
    for k in images:
        indoor6_name_2to_colmap_index[images[k].name] = k
        # print(images[k])

    dataloader, data = extract(opt)

    augmented_visibility = copy.deepcopy(data.visibility)
    monodepth_folder = os.path.join(opt.dataset_folder,
                                    opt.scene_id,
                                    'depth')

    count_invalid_images = 0

    ##############################################################
    ### Creating depth images and augment visibility based on ####
    ### the consistency between depth and 3D points from colmap ##
    ##############################################################

    for idx, batch in enumerate(tqdm(dataloader)):        
        _, _, H, W = batch['image'].shape
        # batch['intrinsic']

        original_image_name = data.original_image_name(idx)
        colmap_index = indoor6_name_2to_colmap_index[original_image_name]
        if images[colmap_index].name != original_image_name:
            print('indoor6 name: ', data.image_files[idx], ', original name ', original_image_name)        


        point3D_ids = images[colmap_index].point3D_ids
        
        dmonodense = np.load(os.path.join(monodepth_folder, data.image_files[idx].replace('jpg', 'npy')))

        K = batch['intrinsics'][0].cpu().numpy()
        R = batch['pose_gt'][0, :3, :3].cpu().numpy()
        t = batch['pose_gt'][0, :3, 3].cpu().numpy()

        xys = images[colmap_index].xys

        monoscaled_depth_path = os.path.join(monodepth_folder, data.image_files[idx].replace('.jpg', '.scaled_depth.npy'))
        if not os.path.exists(monoscaled_depth_path):
            ds = np.zeros(len(point3D_ids))
            dmono = np.zeros(len(point3D_ids))
            validIdx = 0

            for i, k in enumerate(point3D_ids):            
                if k != -1:
                    Cp = R @ points[k].xyz + t
                    xyz = K @ Cp
                    proj_x = xyz[0] / xyz[2]
                    proj_y = xyz[1] / xyz[2]

                    px = xys[i][0]
                    py = xys[i][1]

                    if Cp[2] < 15.0 and proj_x >= 0 and proj_x < W and proj_y >= 0 and proj_y < H and np.abs(proj_x-px) < 5.0 and np.abs(proj_y-py) < 5.0:
                        ds[validIdx] = Cp[2]
                        dmono[validIdx] = dmonodense[int(proj_y), int(proj_x)]

                        ## Doing sth here to compute surface normal
                        validIdx += 1
            
            if validIdx < 10:
                dmonodense_scaled = None
                count_invalid_images += 1
            else:
                ds = ds[:validIdx]
                dmono = dmono[:validIdx]
                A = np.array([[np.sum(dmono**2), np.sum(dmono)], [np.sum(dmono), validIdx]])
                b = np.array([np.sum(dmono*ds), np.sum(ds)])
                k = np.linalg.solve(A, b)

                dmonodense_scaled = k[0] * dmonodense + k[1]
                np.save(monoscaled_depth_path, dmonodense_scaled)
        else:
            dmonodense_scaled = np.load(monoscaled_depth_path)

        if dmonodense_scaled is not None:
            Cplm = batch['landmark3d'][0].cpu().numpy()            
            pixlm = K @ Cplm
            px = pixlm[0] / pixlm[2]
            py = pixlm[1] / pixlm[2]
            infront_infrustum = (Cplm[2] > 0.3) * (Cplm[2] < 15.0) * (px >= 0) * (px < W) * (py >=0) * (py < H)

            vis = copy.deepcopy(augmented_visibility[:, data.image_indices[idx]])
            count_colmap_vs_depth_incompatibility = 0
            count_infront_infrustum = 0
            for l in range(data.landmark.shape[1]):
                if infront_infrustum[l]:
                    count_infront_infrustum += 1

                    depth_from_scaled_mono = dmonodense_scaled[int(py[l]), int(px[l])]
                    depth_from_lm_proj = Cplm[2, l]
                    rel_depth = np.abs(depth_from_lm_proj - depth_from_scaled_mono) / depth_from_lm_proj

                    if vis[l]==0:                    
                        if rel_depth < 0.3: ## 30% depth compatible
                            vis[l] = True

            augmented_visibility[:, data.image_indices[idx]] = vis

    np.savetxt(os.path.join(opt.dataset_folder, opt.scene_id, opt.visibility_config + '_depth.txt'), augmented_visibility, fmt='%d')    


    #########################################################
    ### Adding visibility refinement using surface normal ###
    #########################################################
    root_folder=opt.dataset_folder
    scene_id=opt.scene_id

    data = pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))
    imgs = data['train'] + data['val'] + data['test']
    idx = data['train_idx'] + data['val_idx'] + data['test_idx']

    landmark_config = opt.landmark_config
    visibility_config = opt.visibility_config
    visibility_depth_config = visibility_config + '_depth'

    np.random.seed(100)
    landmark_colors = np.random.rand(10000, 3)

    landmark_file = open(root_folder + '/' + scene_id + '/%s.txt' % landmark_config, 'r')
    num_landmark = int(landmark_file.readline())

    lm = []
    for l in range(num_landmark):
        pl = landmark_file.readline().split()
        pl = np.array([float(pl[i]) for i in range(len(pl))])
        lm.append(pl)
    lm = np.asarray(lm)[:, 1:].T

    visibility_file = root_folder + '/' + scene_id + '/%s.txt' % visibility_config
    visibility = np.loadtxt(visibility_file).astype(bool)

    visibility_file = root_folder + '/' + scene_id + '/%s.txt' % visibility_depth_config
    visibility_depth = np.loadtxt(visibility_file).astype(bool)
    new_visibility = copy.deepcopy(visibility_depth)

    lm_spheres = []
    mesh_arrows = []
    mesh_arrows_ref = []
    H = 720
    W = 1280

    WW, HH = np.meshgrid(np.arange(W), np.arange(H))
    WW = WW.reshape(1, H, W)
    HH = HH.reshape(1, H, W)
    wh1 = np.concatenate((WW, HH, np.ones_like(HH)), axis=0)
    lm_sn = np.zeros((num_landmark, 6))
    lm_sn[:, :3] = lm.T

    for lm_idx in tqdm(range(visibility.shape[0])):
        ## Observe from colmap

        visibility_matrix_ids = [i for i in np.where(visibility[lm_idx, idx])[0]]

        images_observe_lm = [imgs[i] for i in visibility_matrix_ids]
        pose_paths = [os.path.join(root_folder, scene_id, 'images', ifile.replace('color.jpg', 'pose.txt')) for ifile in images_observe_lm]
        depth_paths = [os.path.join(root_folder, scene_id, 'depth', ifile.replace('.jpg', '.scaled_depth.npy')) for ifile in images_observe_lm]
        intrinsic_paths = [os.path.join(root_folder, scene_id, 'images', ifile.replace('color.jpg', 'intrinsics.txt')) for ifile in images_observe_lm]

        depths = np.zeros((len(pose_paths), H, W))
        Ts = np.zeros((len(pose_paths), 4, 4))
        Ks = np.zeros((len(pose_paths), 3, 3))
        for i, pp in enumerate(pose_paths):
            T = np.loadtxt(pp)
            T = np.concatenate( (T, np.array([[0, 0, 0, 1]])), axis=0)
            Ts[i] = T

            intrinsics = open(intrinsic_paths[i])
            intrinsics = intrinsics.readline().split()
            fx = float(intrinsics[2])
            fy = float(intrinsics[2])

            cx = float(intrinsics[3])
            cy = float(intrinsics[4])

            K = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]])
            Ks[i] = K
        

        ## First estimate for surface normal using just visibility vector
        bsum = np.zeros(3)    
        for i in range(Ts.shape[0]):
            Gpt = lm[:, lm_idx] + Ts[i, :3, :3].T @ Ts[i, :3, 3]
            bsum -= (Gpt / np.linalg.norm(Gpt))                        
        bsum /= np.linalg.norm(bsum)
        
        ## Refine the surface normal based on depth image
        bref = np.zeros(3)
        patch_size = 50
        for i in range(Ts.shape[0]):
            if os.path.exists(depth_paths[i]):
                cp = Ts[i, :3, :3] @ lm[:, lm_idx] + Ts[i, :3, 3]
                cp = Ks[i] @ cp
                cp = cp.reshape(-1)
                proj_x = int(cp[0] / cp[2])
                proj_y = int(cp[1] / cp[2])

                if proj_x >= patch_size and proj_x < W-patch_size and proj_y >= patch_size and proj_y < H-patch_size:
                    patch_x0, patch_x1 = proj_x-patch_size, proj_x+patch_size
                    patch_y0, patch_y1 = proj_y-patch_size, proj_y+patch_size

                    d = np.load(depth_paths[i])[patch_y0:patch_y1, patch_x0:patch_x1].reshape((1, patch_size * 2, patch_size * 2))
                    pcd = np.linalg.inv(Ks[i]) @ (wh1[:, patch_y0:patch_y1, patch_x0:patch_x1] * d).reshape(3, 4 * patch_size ** 2)

                    A = np.concatenate((pcd, np.ones((1, 4 * patch_size ** 2))), axis=0)
                    D, U = np.linalg.eig(A @ A.T)
                    
                    sn = Ts[i, :3, :3].T @ U[:3, np.argsort(D)[0]]
                    sn /= np.linalg.norm(sn)

                    if np.sum(bsum * sn) > 0.0:
                        bref += sn
                    elif np.sum(bsum * sn) < 0.0:
                        bref -= sn
        
        if np.linalg.norm(bref) == 0:
            lm_sn[lm_idx, 3:] = bsum
        else:
            bref /= np.linalg.norm(bref)
            lm_sn[lm_idx, 3:] = bref

        visibility_matrix_ids = [i for i in np.where(visibility_depth[lm_idx, idx])[0]]
        images_observe_lm = [imgs[i] for i in np.where(visibility_depth[lm_idx, idx])[0]]
        pose_paths = [os.path.join(root_folder, scene_id, 'images', ifile.replace('color.jpg', 'pose.txt')) for ifile in images_observe_lm]
        for i, pp in enumerate(pose_paths):
            T = np.loadtxt(pp)
            if visibility_depth[lm_idx, idx[visibility_matrix_ids[i]]]:
                Gpt = lm[:, lm_idx] + T[:3, :3].T @ T[:3, 3]
                Gpt /= np.linalg.norm(Gpt)
                if np.sum(bref * Gpt) > -0.2: ## violate visibility direction
                    new_visibility[lm_idx, idx[visibility_matrix_ids[i]]] = 0
    
    np.savetxt(os.path.join(root_folder, scene_id, '%s_normal.txt' % (landmark_config)), lm_sn)
    np.savetxt(os.path.join(root_folder, scene_id, '%s_depth_normal.txt' % (visibility_config)), new_visibility, fmt='%d')