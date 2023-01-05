import numpy as np
import torch


def generate_heat_maps(landmarks, visibility_mask, heatmap_size, K, sigma=3):
    '''
    :param landmarks:  [3, L]
    :param visibility_mask: [L]
    :return: hms, hms_weight(1: visible, 0: invisible)
    '''


    hms = np.zeros((landmarks.shape[1],
                       heatmap_size[0],
                       heatmap_size[1]),
                       dtype=np.float32)

    hms_weights = np.ones((landmarks.shape[1]), dtype=np.float32)

    tmp_size = sigma * 3

    for lm_id in range(landmarks.shape[1]):
        landmark_2d = K @ landmarks[:, lm_id]
        landmark_2d /= landmark_2d[2]

        mu_x = int(landmark_2d[0] + 0.5)
        mu_y = int(landmark_2d[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_y - tmp_size), int(mu_x - tmp_size)]
        br = [int(mu_y + tmp_size + 1), int(mu_x + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0 or landmarks[2, lm_id] < 0:
            continue

        if visibility_mask[lm_id]:
            ## Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_y = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_x = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]

            # Image range
            img_y = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_x = max(0, ul[1]), min(br[1], heatmap_size[1])

            hms[lm_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            hms_weights[lm_id] = 0.0
    return hms, hms_weights


def generate_heat_maps_gpu(landmarks_2d, visibility_mask, heatmap_size, sigma=3):
    '''
    gpu version of heat map generation
    :param landmarks:  [3, L]
    :return: hms
    '''

    B, _, L = landmarks_2d.shape
    H, W = heatmap_size[0], heatmap_size[1]

    yy_grid, xx_grid = torch.meshgrid(torch.arange(0, heatmap_size[0]),
                                      torch.arange(0, heatmap_size[1]))
    xx_grid, yy_grid = xx_grid.to(device=landmarks_2d.device), yy_grid.to(device=landmarks_2d.device)
    hms = torch.exp(-((xx_grid.reshape(1, 1, H, W)-landmarks_2d[:, 0].reshape(B, L, 1, 1))**2 +
                      (yy_grid.reshape(1, 1, H, W)-landmarks_2d[:, 1].reshape(B, L, 1, 1))**2)/(2*sigma**2))
    hms_vis = hms * visibility_mask.reshape(B, L, 1, 1).float()
    hms_vis[hms_vis < 0.1] = 0.0
    normalizing_factor, _ = torch.max(hms_vis.reshape(B, L, -1), dim=2)
    hms_vis[normalizing_factor > 0.5] = hms_vis[normalizing_factor > 0.5] / \
                                        normalizing_factor.reshape(B, L, 1, 1)[normalizing_factor > 0.5]

    return hms_vis