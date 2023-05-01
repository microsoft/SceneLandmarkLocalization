import argparse
import numpy as np
import os
import pickle
from read_write_models import qvec2rotmat, read_model
from tqdm import tqdm


def ComputePerPointTimeSpan(image_ids, images):
    timespan = {}
    
    for imageID in image_ids:
        session_id = int(images[imageID].name.split('-')[0])
        if session_id in timespan:
            timespan[session_id] += 1
        else:
            timespan[session_id] = 1

    return len(timespan)


def ComputePerPointDepth(pointInGlobal, image_ids, images):
    d = np.zeros(len(image_ids))
    for i, imageID in enumerate(image_ids):
        R = qvec2rotmat(images[imageID].qvec)
        t = images[imageID].tvec
        pointInCamerai = R @ pointInGlobal + t
        d[i] = pointInCamerai[2]
    
    pointDepthMean, pointDepthStd = np.mean(d), np.std(d)

    return pointDepthMean, pointDepthStd


def ComputePerPointAngularSpan(pointInGlobal, image_ids, images):
    N = len(image_ids)
    H = np.zeros((3, 3))
    for i, imageID in enumerate(image_ids):
        Ri = qvec2rotmat(images[imageID].qvec)
        ti = images[imageID].tvec
        bi = Ri.T @ (pointInGlobal - ti)
        bi = bi / np.linalg.norm(bi)
        H += (np.eye(3) - np.outer(bi, bi))
    
    H /= N
    eigH = np.linalg.eigvals(0.5*(H + H.T))
    
    return np.arccos(np.clip(1 - 2.0 * np.min(eigH)/np.max(eigH), 0, 1))


def SaveLandmarksAndVisibilityMask(selected_landmarks, points3D, images, indoor6_imagename_to_index, num_images, root_path, outformat):
    
    num_landmarks = len(selected_landmarks['id'])

    visibility_mask = np.zeros((num_landmarks, num_images), dtype=np.uint8)

    for i, pid in enumerate(selected_landmarks['id']):
        for imgid in points3D[pid].image_ids:
            if images[imgid].name in indoor6_imagename_to_index:
                visibility_mask[i, indoor6_imagename_to_index[images[imgid].name]] = 1

    np.savetxt(os.path.join(root_path, 'visibility-%d%s.txt' % (num_landmarks, outformat)), visibility_mask, fmt='%d')

    f = open(os.path.join(root_path, 'landmarks-%d%s.txt' % (num_landmarks, outformat)), 'w')
    f.write('%d\n' % num_landmarks)
    for i in range(selected_landmarks['xyz'].shape[1]):
        f.write('%d %4.4f %4.4f %4.4f\n' % (i, 
                                            selected_landmarks['xyz'][0, i], 
                                            selected_landmarks['xyz'][1, i], 
                                            selected_landmarks['xyz'][2, i]))
    f.close()



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
        '--output_format', type=str, default='v2',
        help='Landmark file output.')

    opt = parser.parse_args()

    scene = opt.scene_id
    path = os.path.join(opt.dataset_folder, 'indoor6-colmap/%s-tr/sparse/0/' % scene)
    cameras, images, points3D = read_model(path, ext='.bin')
    
    ## Max number of sessions
    sessions = {}
    for i in images:
        print(images[i].name)
        session_id = int(images[i].name.split('-')[0])
        sessions[session_id] = 1
    maxSession = len(sessions)

    ## Initialization
    numPoints3D = len(points3D)
    points3D_ids = np.zeros(numPoints3D)
    points3D_scores = np.zeros(numPoints3D)
    validIdx = 0

    ## Compute score for each landmark    
    for i, k in enumerate(tqdm(points3D)):            
        pointInGlobal = points3D[k].xyz
        image_ids = points3D[k].image_ids
        trackLength = len(image_ids)
            
        if trackLength > 25:        
            depthMean, depthStd = ComputePerPointDepth(pointInGlobal, image_ids, images)        
            timespan = ComputePerPointTimeSpan(image_ids, images)
            anglespan = ComputePerPointAngularSpan(pointInGlobal, image_ids, images)
            
            depthScore = min(1.0, depthStd / depthMean) 
            trackLengthScore = 0.25 * np.log2(trackLength)
            timeSpanScore = timespan / maxSession
            
            if timespan >= 1 and depthMean < 10.0 and anglespan > 0.3:
                points3D_ids[validIdx] = k
                points3D_scores[validIdx] = depthScore + trackLengthScore + timeSpanScore + anglespan
                validIdx += 1                
        
    
    ## Sort scores
    points3D_ids = points3D_ids[:validIdx]
    points3D_scores = points3D_scores[:validIdx]
    sorted_indices = np.argsort(points3D_scores)


    ## Greedy selection
    selected_landmarks = {'id': np.zeros(opt.num_landmarks), 
                        'xyz': np.zeros((3, opt.num_landmarks)), 
                        'score': np.zeros(opt.num_landmarks)}

    ## Selecting first point
    selected_landmarks['id'][0] = points3D_ids[sorted_indices[-1]]
    selected_landmarks['xyz'][:, 0] = points3D[selected_landmarks['id'][0]].xyz
    selected_landmarks['score'][0] = points3D_scores[sorted_indices[-1]]

    nselected = 1
    radius = 5.0

    while nselected < opt.num_landmarks:
        for i in reversed(sorted_indices):
            id = points3D_ids[i]
            xyz = points3D[id].xyz        

            if np.sum(np.linalg.norm(xyz.reshape(3, 1) - selected_landmarks['xyz'][:, :nselected], axis=0) < radius):
                continue
            else:
                selected_landmarks['id'][nselected] = id
                selected_landmarks['xyz'][:, nselected] = xyz
                selected_landmarks['score'][nselected] = points3D_scores[i]
                nselected += 1

            if nselected == opt.num_landmarks:
                break
            
        radius *= 0.5

    ## Saving
    indoor6_images = pickle.load(open(os.path.join(opt.dataset_folder, '%s/train_test_val.pkl' % opt.scene_id), 'rb'))
    indoor6_imagename_to_index = {}

    for i, f in enumerate(indoor6_images['train']):
        image_name = open(os.path.join(opt.dataset_folder, 
                                    opt.scene_id, 'images', 
                                    f.replace('color.jpg', 
                                                'intrinsics.txt'))).readline().split(' ')[-1][:-1]
        indoor6_imagename_to_index[image_name] = indoor6_images['train_idx'][i]

    num_images = len(indoor6_images['train']) + len(indoor6_images['val']) + len(indoor6_images['test'])
    SaveLandmarksAndVisibilityMask(selected_landmarks, points3D, images, indoor6_imagename_to_index, num_images, 
                                   os.path.join(opt.dataset_folder, opt.scene_id, 'landmarks'), opt.output_format)