import argparse
import copy
import fnmatch
import numpy as np
import os
import pickle
from PIL import Image

import sys
sys.path.append('../utils')

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pnp import Quaternion2Rotation

np.random.seed(0)

class Indoor6(Dataset):
    def __init__(self, root_folder="",
                 scene_id='', mode='all',
                 landmark_idx=[None], skip_image_index=1,
                 input_image_downsample=1, gray_image_output=False,
                 landmark_config='landmarks/landmarks-50',
                 visibility_config='landmarks/visibility-50'):
        super(Indoor6, self).__init__()

        self.to_tensor = transforms.ToTensor()

        self.image_folder = os.path.join(root_folder,
                                        scene_id,
                                        'images')
        image_files_all = fnmatch.filter(os.listdir(self.image_folder), '*.color.jpg')
        image_files_all = sorted(image_files_all)[::skip_image_index]

        self.image_files = []
        if mode == 'train':
            self.image_files = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'train'][::skip_image_index]
            self.image_indices = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'train_idx'][::skip_image_index]
        elif mode == 'test':
            self.image_files = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'test'][::skip_image_index]
            self.image_indices = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'test_idx'][::skip_image_index]
        elif mode == 'val':
            self.image_files = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'val'][::skip_image_index]
            self.image_indices = \
                pickle.load(open('%s/%s/train_test_val.pkl' % (root_folder, scene_id), 'rb'))[
                    'val_idx'][::skip_image_index]
        else:
            self.image_files = image_files_all
            self.image_indices = np.arange(0, len(image_files_all))

        self.image_indices = np.asarray(self.image_indices)
        self.num_images = len(self.image_files)
        self.gray_image_output = gray_image_output
        self.mode = mode

        landmark_file = open(root_folder + '/' + scene_id
                                         + '/%s.txt' % landmark_config, 'r')
        num_landmark = int(landmark_file.readline())
        self.landmark = []
        for l in range(num_landmark):
            pl = landmark_file.readline().split()
            pl = np.array([float(pl[i]) for i in range(len(pl))])
            self.landmark.append(pl)
        self.landmark = np.asarray(self.landmark)[:, 1:]

        self.image_downsampled = input_image_downsample

        visibility_file = root_folder + '/' + scene_id + '/%s.txt' % visibility_config
        self.visibility = np.loadtxt(visibility_file).astype(bool)
        
        if landmark_idx[0] != None:
            self.landmark = self.landmark[landmark_idx]
            self.visibility = self.visibility[landmark_idx]
        
        self.landmark = self.landmark.transpose()


    def _modify_intrinsic(self, index):
        W = None
        H = None
        K = None
        K_inv = None

        while K_inv is None:
            try:
                intrinsics = open(os.path.join(self.image_folder,
                                               self.image_files[index].replace('color.jpg', 'intrinsics.txt')))

                intrinsics = intrinsics.readline().split()

                W = int(intrinsics[0]) // (self.image_downsampled * 32) * 32
                H = int(intrinsics[1]) // (self.image_downsampled * 32) * 32

                scale_factor_x = W / float(intrinsics[0])
                scale_factor_y = H / float(intrinsics[1])

                fx = float(intrinsics[2]) * scale_factor_x
                fy = float(intrinsics[2]) * scale_factor_y

                cx = float(intrinsics[3]) * scale_factor_x
                cy = float(intrinsics[4]) * scale_factor_y

                K = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.]], dtype=float)

                K_inv = np.linalg.inv(K)

            except(RuntimeError, TypeError, NameError):
                pass
        return K, K_inv, W, H

    def _load_and_resize_image(self, index, W, H):
        color_img_rs = None
        while color_img_rs is None:
            try:
                # Load color image
                color_img = Image.open(os.path.join(self.image_folder, self.image_files[index]))
                color_img_rs = color_img.resize((W, H), resample=Image.BILINEAR)
            except(RuntimeError, TypeError, NameError):
                pass

        color_tensor = self.to_tensor(color_img_rs)

        return color_tensor

    def _load_pose(self, index):
        pose = None
        while pose is None:
            try:
                # Load 3x4 pose matrix and make it 4x4 by appending vector [0., 0., 0., 1.]
                pose = np.loadtxt(os.path.join(self.image_folder, self.image_files[index].replace('color.jpg', 'pose.txt')))
            except (RuntimeError, TypeError, NameError):
                pass

        pose_s = np.vstack((pose, np.array([0., 0., 0., 1.])))

        return pose_s

    def __getitem__(self, index):
        K, K_inv, W_modified, H_modified = self._modify_intrinsic(index)
        color_tensor = self._load_and_resize_image(index, W_modified, H_modified)
        C_T_G = self._load_pose(index)

        landmark3d = C_T_G @ np.vstack((self.landmark, np.ones((1, self.landmark.shape[1]))))

        output = {'pose_gt': torch.tensor(C_T_G),
                  'image': color_tensor,
                  'intrinsics': torch.tensor(K, dtype=torch.float32, requires_grad=False),
                  'inv_intrinsics': torch.tensor(K_inv, dtype=torch.float32, requires_grad=False),
                  'landmark3d': torch.tensor(landmark3d[:3], dtype=torch.float32, requires_grad=False),
                  }

        if self.mode == 'train':
            proj = K @ (C_T_G[:3, :3] @ self.landmark + C_T_G[:3, 3:])
            landmark2d = proj / proj[2:]
            output['landmark2d'] = landmark2d[:2]

            inside_patch = (landmark2d[0] < W_modified) * \
                           (landmark2d[0] >= 0) * \
                           (landmark2d[1] < H_modified) * \
                           (landmark2d[1] >= 0)  # L vector

            # visible by propagated colmap visibility and inside image
            _mask1 = self.visibility[:, self.image_indices[index]] * inside_patch

            # outside patch
            # _mask2 = ~inside_patch

            # inside image but not visible by colmap
            _mask3 = (self.visibility[:, self.image_indices[index]] == 0) * inside_patch

            visibility_mask = 1.0 * _mask1 + 0.5 * _mask3
            output['visibility'] = visibility_mask

        return output

    def __len__(self):
        return self.num_images


class Indoor6Patches(Indoor6):
    def __init__(self, root_folder="",
                 scene_id='', mode='all',
                 landmark_idx=[None], skip_image_index=1,
                 input_image_downsample=1, gray_image_output=False,
                 patch_size=96,
                 positive_samples=4, random_samples=4,
                 landmark_config='landmarks/landmarks-50',
                 visibility_config='landmarks/visibility-50',
                 augmentation=True):
        super().__init__(root_folder=root_folder,
                         scene_id=scene_id, mode=mode,
                         landmark_idx=landmark_idx, skip_image_index=skip_image_index,
                         input_image_downsample=input_image_downsample, gray_image_output=gray_image_output,
                         landmark_config=landmark_config,
                         visibility_config=visibility_config)
        self.patch_size = patch_size
        self.positive_samples = positive_samples
        self.random_samples = random_samples
        self.landmark_idx = landmark_idx
        self.augmentation = augmentation

        self.num_landmarks = self.landmark.shape[1]

    def _extract_patch(self, C_T_G, lm_idx, K, W_modified, H_modified, center=False, adjust_boundary=True):

        proj = K @ (C_T_G[:3, :3] @ self.landmark[:, lm_idx:(lm_idx + 1)] + C_T_G[:3, 3:])
        proj /= copy.copy(proj[2:])

        # Extract patch
        y = int(proj[1, 0])
        x = int(proj[0, 0])

        if center:
            dy = -self.patch_size // 2
            dx = -self.patch_size // 2
        else:
            dy = -np.random.rand(1) * self.patch_size
            dx = -np.random.rand(1) * self.patch_size

        _top = int(y + dy)
        _bottom = _top + int(self.patch_size)
        _left = int(x + dx)
        _right = _left + int(self.patch_size)

        if adjust_boundary:
            # Adjust the boundary
            if _top < 0:
                _top = 0
                _bottom = int(self.patch_size)
            elif _bottom >= H_modified:
                _top = H_modified - int(self.patch_size)
                _bottom = H_modified

            if _left < 0:
                _left = 0
                _right = int(self.patch_size)
            elif _right >= W_modified:
                _left = W_modified - int(self.patch_size)
                _right = W_modified

        return _left, _right, _top, _bottom

    def _project_landmarks_into_patch(self, K, C_T_G, img_idx, _top, _bottom, _left, _right):
        proj = K @ (C_T_G[:3, :3] @ self.landmark + C_T_G[:3, 3:])
        in_front_of_camera = proj[2] > 0.0
        proj /= copy.copy(proj[2:])

        proj_patch = np.zeros_like(proj[:2])
        proj_patch[0] = proj[0] - _left
        proj_patch[1] = proj[1] - _top

        # L vector
        inside_patch = (proj[0] < _right) * (proj[0] >= _left) * (proj[1] < _bottom) * (
                    proj[1] >= _top) * in_front_of_camera

        # visible by propagated colmap visibility and inside patch
        _mask1 = self.visibility[:, self.image_indices[img_idx]] * inside_patch

        # outside patch
        # _mask2 = ~inside_patch

        # inside patch but not visible by colmap
        _mask3 = (self.visibility[:, self.image_indices[img_idx]] == 0) * inside_patch

        visibility_mask = 1.0 * _mask1 + 0.5 * _mask3

        return proj_patch, visibility_mask

    def __getitem__(self, index):

        patches = []
        keypoint_locations = []
        landmark_visibility_on_patch = []
        L = self.landmark.shape[1]  # number of keypoints

        list_landmarks = np.random.permutation(L)[:self.positive_samples]
        
        ## Create positive examples
        for lm_idx in list_landmarks:
            ## Randomly draw image index from visibility mask
            training_img_ids_observe_lm_idx = self.visibility[lm_idx, self.image_indices].reshape(-1)
            total_images_observed_this_lm = np.sum(training_img_ids_observe_lm_idx)
            if total_images_observed_this_lm == 0:
                print('no positive example')
                img_idx_positive_sample_for_lm_idx = np.random.randint(self.num_images)
            else:
                # img_idx_observe_lm_idx = (index % int(np.sum(training_img_ids_observe_lm_idx)))
                random_indices_observe_this_lm = np.random.randint(0, total_images_observed_this_lm)
                img_idx_positive_sample_for_lm_idx = np.where(training_img_ids_observe_lm_idx==1)[0][random_indices_observe_this_lm]

            K, K_inv, W_modified, H_modified = self._modify_intrinsic(img_idx_positive_sample_for_lm_idx)
            C_T_G = self._load_pose(img_idx_positive_sample_for_lm_idx)
            color_tensor = self._load_and_resize_image(img_idx_positive_sample_for_lm_idx, W_modified, H_modified)

            if not self.augmentation:
                _left, _right, _top, _bottom = self._extract_patch(C_T_G, lm_idx, K, W_modified, H_modified,
                                                                   center=False, adjust_boundary=True)
                color_patch = color_tensor.reshape(1, 3, H_modified, W_modified)[:, :, _top:_bottom, _left:_right]
                Cg_T_G = C_T_G
                K_scale = K
            else:
                ## Random rotation, change K, T
                q = np.random.rand(4) - 0.5
                q[1] *= 0.1  # pitch
                q[2] *= 0.1  # yaw
                q[3] *= 0.1  # roll
                q[0] = 1.0
                q /= np.linalg.norm(q)
                Cg_R_C = Quaternion2Rotation(q)
                Cg_T_C = np.eye(4)
                Cg_T_C[:3, :3] = Cg_R_C

                Cg_T_G = Cg_T_C @ C_T_G
                K_scale = K.copy()
                K_scale[:2, :2] *= (0.9 + 0.2*np.random.rand())
                K_scale_inv = np.linalg.inv(K_scale)

                _left, _right, _top, _bottom = self._extract_patch(Cg_T_G, lm_idx, K_scale, W_modified, H_modified,
                                                                   center=False, adjust_boundary=False)

                ## Extract patch
                YY_patch, XX_patch = torch.meshgrid(torch.arange(_top, _bottom, 1),
                                                    torch.arange(_left, _right, 1))
                XX_patch = XX_patch.reshape(1, self.patch_size, self.patch_size).float()
                YY_patch = YY_patch.reshape(1, self.patch_size, self.patch_size).float()

                in_H_out = K @ Cg_R_C.T @ K_scale_inv
                in_H_out = torch.tensor(in_H_out, dtype=torch.float)
                in_p_out = in_H_out @ torch.cat((XX_patch,
                                                 YY_patch,
                                                 torch.ones_like(XX_patch)), dim=1).reshape((3, self.patch_size**2))
                in_p_out = in_p_out / in_p_out[2:].clone()

                scale = torch.tensor([[2. / W_modified, 0.],
                                      [0., 2. / H_modified]], dtype=torch.float).reshape(2, 2)
                center = torch.tensor([0.5 * (W_modified - 1),
                                       0.5 * (H_modified - 1)], dtype=torch.float).reshape(2, 1)
                in_p_out_normalized = scale @ (in_p_out[:2] - center)

                invalid_pixel_mask = (in_p_out_normalized[0] < -1) + \
                                     (in_p_out_normalized[0] > 1) + \
                                     (in_p_out_normalized[1] < -1) + \
                                     (in_p_out_normalized[1] > 1)

                if torch.sum(invalid_pixel_mask>0) > 0.25 * self.patch_size ** 2:
                    _left, _right, _top, _bottom = self._extract_patch(C_T_G, lm_idx, K, W_modified, H_modified,
                                                                      center=False, adjust_boundary=True)
                    color_patch = color_tensor.reshape(1, 3, H_modified, W_modified)[:, :, _top:_bottom, _left:_right]

                    # Not using augmented transformation
                    K_scale = K.copy()
                    Cg_T_G = C_T_G
                else:
                    grid_sampler = in_p_out_normalized.reshape(1, 2, self.patch_size, self.patch_size).permute(0, 2, 3, 1)
                    color_tensor = color_tensor.reshape(1, 3, H_modified, W_modified)
                    color_patch = torch.nn.functional.grid_sample(color_tensor, grid_sampler,
                                                                 padding_mode='zeros', mode='bilinear', align_corners=False)
                    color_patch = torch.nn.functional.interpolate(color_patch, size=(self.patch_size, self.patch_size))

            keypoints_2d, visibility_mask = self._project_landmarks_into_patch(K_scale, Cg_T_G, img_idx_positive_sample_for_lm_idx, _top, _bottom, _left, _right)
            patches.append(color_patch)
            keypoint_locations.append(keypoints_2d.reshape((1, 2, L)))
            landmark_visibility_on_patch.append(visibility_mask.reshape((1, L)))

        ## Create random examples
        patches_random = []
        keypoint_locations_random = []
        landmark_visibility_on_patch_random = []

        C_T_G = self._load_pose(index)
        K, K_inv, W_modified, H_modified = self._modify_intrinsic(index)
        color_tensor = self._load_and_resize_image(index, W_modified, H_modified)

        for _ in range(self.random_samples):
            _top = int(np.random.rand(1) * (H_modified - self.patch_size))
            _bottom = _top + self.patch_size
            _left = int(np.random.rand(1) * (W_modified - self.patch_size))
            _right = _left + self.patch_size

            keypoints_2d, visibility_mask = self._project_landmarks_into_patch(K, C_T_G, index, _top, _bottom, _left, _right)

            patches_random.append(color_tensor[:, _top:_bottom, _left:_right].clone().reshape(1, 3, self.patch_size, self.patch_size))
            keypoint_locations_random.append(keypoints_2d.reshape((1, 2, L)))
            landmark_visibility_on_patch_random.append(visibility_mask.reshape((1, L)))

        patches = torch.cat(patches+patches_random, dim=0)
        keypoint_locations = np.concatenate(keypoint_locations+keypoint_locations_random, axis=0)
        landmark_visibility_on_patch = np.concatenate(landmark_visibility_on_patch+landmark_visibility_on_patch_random, axis=0)

        ## COLOR AUGMENTATION
        if self.augmentation:
            if torch.rand(1) > 0.5:
                patches += 0.02 * (
                            torch.rand((patches.shape[0], patches.shape[1], 1, 1)) - 0.5) * torch.ones_like(patches)
            else:
                patches += 0.2 * (
                        torch.rand((patches.shape[0], 1, 1, 1)) - 0.5) * torch.ones_like(patches)
        clipped_patches = torch.clip(patches, 0, 1)


        output = {'patches': clipped_patches,
                  'landmark2d': torch.tensor(keypoint_locations, dtype=torch.float, requires_grad=False),
                  'visibility': torch.tensor(landmark_visibility_on_patch, requires_grad=False),
                  }

        return output
