import argparse
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloader.indoor6 import Indoor6Patches


def visualize_keypoint_np(image_, y, x, kp_color):
    image = image_.copy()
    if np.sum(kp_color) == 255:
        square_size = 7
    else:
        square_size = 3
    for c in range(3):
        image[y - square_size:y + square_size, x - square_size:x + square_size, c] = kp_color[c]

    return image


def row_to_matrix_image(img, columns=30, block=101):
    W = img.shape[1]
    rows = W // (columns * block)

    matrix_image = []
    for r in range(rows):
        matrix_image.append(img[:, (columns * block * r):(columns * block * (r+1))])

    leftover = W - rows * (columns * block)
    if leftover > 0:
        matrix_image.append(np.concatenate((img[:, (columns * block * rows):],
                                            np.zeros((block, columns*block-leftover, 3), dtype=np.uint8)), axis=1))

    matrix_image = np.concatenate(matrix_image, axis=0)

    return matrix_image


class Indoor6PatchesPerLandmarks(Indoor6Patches):
    def __init__(self, root_folder="",
                        scene_id='', mode='train',
                        landmark_idx=-1, skip_image_index=1,
                        input_image_downsample=2, gray_image_output=False,
                        patch_size=96,
                        landmark_config='landmarks/landmarks-50',
                        visibility_config='landmarks/visibility-50'):
        super().__init__(root_folder=root_folder,
                        scene_id=scene_id, mode=mode,
                        landmark_idx=landmark_idx, skip_image_index=skip_image_index,
                        input_image_downsample=input_image_downsample, gray_image_output=gray_image_output,
                        landmark_config=landmark_config,
                        visibility_config=visibility_config,
                        augmentation=False, patch_size=192, positive_samples=8, random_samples=8)
    def __getitem__(self, index):

        patches = []
        keypoint_locations = []
        landmark_visibility_on_patch = []

        L = self.landmark.shape[1]  # number of keypoints

        ## Randomly draw image index from visibility mask
        training_img_ids_observe_lm_idx = self.visibility[lm_idx, self.image_indices].reshape(-1)
        img_idx_observe_lm_idx = np.where(training_img_ids_observe_lm_idx == 1)[0][index]

        K, K_inv, W_modified, H_modified = self._modify_intrinsic(img_idx_observe_lm_idx)
        C_T_G = self._load_pose(img_idx_observe_lm_idx)
        color_tensor = self._load_and_resize_image(img_idx_observe_lm_idx, W_modified, H_modified)


        _left, _right, _top, _bottom = self._extract_patch(C_T_G, self.landmark_idx, K, W_modified, H_modified,
                                                           center=False, adjust_boundary=True)
        color_patch = color_tensor.reshape(1, 3, H_modified, W_modified)[:, :, _top:_bottom, _left:_right]
        Cg_T_G = C_T_G
        K_scale = K

        keypoints_2d, visibility_mask = self._project_landmarks_into_patch(K_scale, Cg_T_G, img_idx_observe_lm_idx, _top, _bottom, _left, _right)
        patches.append(color_patch)
        keypoint_locations.append(keypoints_2d.reshape((1, 2, L)))
        landmark_visibility_on_patch.append(visibility_mask.reshape((1, L)))

        patches = torch.cat(patches, dim=0)
        keypoint_locations = np.concatenate(keypoint_locations, axis=0)
        landmark_visibility_on_patch = np.concatenate(landmark_visibility_on_patch, axis=0)

        clipped_patches = torch.clip(patches, 0, 1)

        output = {'patches': clipped_patches,
                  'landmark2d': torch.tensor(keypoint_locations, dtype=torch.float, requires_grad=False),
                  'visibility': torch.tensor(landmark_visibility_on_patch, requires_grad=False),
                  }

        return output

    def __len__(self):
        training_img_ids_observe_lm_idx = self.visibility[self.landmark_idx, self.image_indices].reshape(-1)
        total_images_observed_this_lm = np.sum(training_img_ids_observe_lm_idx)
        return total_images_observed_this_lm


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
    '--landmark_config', type=str, default='landmarks/landmarks-400',
    help='Landmark configuration.')
parser.add_argument(
    '--visibility_config', type=str, default='landmarks/visibility-400',
    help='Visibility configuration.')
parser.add_argument(
    '--scene_id', type=str, default='scene6',
    help='Scene id')
parser.add_argument(
    '--num_landmarks', type=int, default=300,
    help='Number of landmarks')

opt = parser.parse_args()

if not os.path.exists(opt.output_folder):
    os.mkdir(opt.output_folder)

num_landmarks = opt.num_landmarks

for lm_idx in tqdm(range(num_landmarks)):

    if not os.path.exists('%s/%03d' % (opt.output_folder, lm_idx)):
        os.mkdir('%s/%03d' % (opt.output_folder, lm_idx))

    train_dataset = Indoor6PatchesPerLandmarks(landmark_idx=lm_idx,
                                   scene_id=opt.scene_id,
                                   mode='train',
                                   root_folder=opt.dataset_folder,
                                   input_image_downsample=2,
                                   landmark_config=opt.landmark_config,
                                   visibility_config=opt.visibility_config,
                                   skip_image_index=1,
                                   patch_size=192)

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=64, shuffle=False, pin_memory=True)

    for idx, batch in enumerate(train_dataloader):

        B1, B2, _, H, W = batch['patches'].shape
        B = B1 * B2

        patches = batch['patches'].reshape(B, 3, H, W)
        visibility = batch['visibility'].reshape(B, num_landmarks)
        landmark2d = batch['landmark2d'].reshape(B, 2, num_landmarks)

        # Batch randomization
        input_batch_random = np.arange(0, B)
        landmark2d_rand = [landmark2d[input_batch_random[b:b + 1]] for b in range(B)]
        patches_rand = [patches[input_batch_random[b:b + 1]] for b in range(B)]
        visibility_rand = [visibility[input_batch_random[b:b + 1]] for b in range(B)]

        landmark2d_rand = torch.cat(landmark2d_rand, dim=0)
        patches_rand = torch.cat(patches_rand, dim=0)
        visibility_rand = torch.cat(visibility_rand, axis=0)

        patches_np = []
        for b in range(B):
            patchb = patches_rand[b].numpy().transpose(1, 2, 0) * 255
            y = int(landmark2d_rand[b, 1, lm_idx])
            x = int(landmark2d_rand[b, 0, lm_idx])
            patchb = visualize_keypoint_np(patchb, y, x, np.array([0., 255., 0.]))
            # for l in range(num_landmarks):
            #     if visibility_rand[b, l] > 0:
            #         y = int(landmark2d_rand[b, 1, l])
            #         x = int(landmark2d_rand[b, 0, l])
            #         if l == lm_idx:
            #             color = np.array([255., 0., 0.])
            #         else:
            #             color = visibility_rand[b, l].numpy() * np.array([0., 255., 0.])
            #         patchb = visualize_keypoint_np(patchb, y, x, color)
            patches_np.append(patchb)

        reprojection_patches_l = np.concatenate(patches_np, axis=1)
        img = row_to_matrix_image(reprojection_patches_l.astype(np.uint8), columns=16, block=192)
        img = Image.fromarray(img.astype(np.uint8))
        img.save('%s/%03d/batch_%06d.jpg' % (opt.output_folder, lm_idx, idx))