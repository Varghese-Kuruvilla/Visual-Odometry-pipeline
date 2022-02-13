# Copyright (C) Huangying Zhan 2019. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import cv2
import matplotlib as mpl
import numpy as np
import os
from time import time
import copy
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob
import pickle
import sys
from kittievalodom import KittiEvalOdom
import pandas as pd

np.set_printoptions(suppress=True, precision = 3)


do_gt = True



def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def convert_SE3_to_arr(SE3_dict, timestamps=None):
    """Convert SE3 dictionary to array dictionary
    Args:
        SE3_dict (SE3 dict): SE3 dictionary
        timestamps (float list): timestamp list
    Returns:
        poses_dict (array dict): each pose contains 4x4 array
    """
    poses_dict = {}
    if timestamps is None:
        key_list = sorted(list(SE3_dict.keys()))
    else:
        key_list = timestamps
    for cnt, i in enumerate(SE3_dict):
        poses_dict[key_list[cnt]] = SE3_dict[i].pose
    return poses_dict


def save_traj(txt, poses, format="kitti"):
    """Save trajectory (absolute poses) as KITTI odometry file format
    Args:
        txt (str): pose text file path
        poses (array dict): poses, each pose is 4x4 array
        format (str): trajectory format
            - kitti: 12 parameters
            - tum: timestamp tx ty tz qx qy qz qw
    """
    with open(txt, "w") as f:
        for i in poses:
            pose = poses[i]
            if format == "kitti":
                pose = pose.flatten()[:12]
                line_to_write = " ".join([str(j) for j in pose])
            elif format == "tum":
                qw, qx, qy, qz = rot2quat(pose[:3, :3])
                tx, ty, tz = pose[:3, 3]
                line_to_write = " ".join([
                                    str(i), 
                                    str(tx), str(ty), str(tz),
                                    str(qx), str(qy), str(qz), str(qw)]
                                    )
            f.writelines(line_to_write+"\n")
    # print("Trajectory saved.")

class SE3():
    """SE3 object consists rotation and translation components
    Attributes:
        pose (4x4 numpy array): camera pose
        inv_pose (4x4 numpy array): inverse camera pose
        R (3x3 numpy array): Rotation component
        t (3x1 numpy array): translation component,
    """
    def __init__(self, np_arr=None):
        if np_arr is None:
            self._pose = np.eye(4)
        else:
            self._pose = np_arr

    @property
    def pose(self):
        """ pose (4x4 numpy array): camera pose """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def inv_pose(self):
        """ inv_pose (4x4 numpy array): inverse camera pose """
        return np.linalg.inv(self._pose)

    @inv_pose.setter
    def inv_pose(self, value):
        self._pose = np.linalg.inv(value)

    @property
    def R(self):
        return self._pose[:3, :3]

    @R.setter
    def R(self, value):
        self._pose[:3, :3] = value

    @property
    def t(self):
        return self._pose[:3, 3:]

    @t.setter
    def t(self, value):
        self._pose[:3, 3:] = value

# from ..flowlib.flowlib import flow_to_image
# from .utils import mkdir_if_not_exists


def draw_match_temporal(img1, kp1, img2, kp2, N):
    """Draw matches temporally
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    r1, g1, b1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
    r2, g2, b2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    out_img = img2.copy()
    # out_img[:,:,0] = r1
    # out_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0, out_img)

    # kp_list = np.random.randint(0, min(kp1.shape[0], kp2.shape[0]), N)
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                                            dtype=np.int
                                            )
    for i in kp_list:
        # print(kp1[i])
        # print(kp2[i])
        # input("debug")
        center1 = (kp1[i][0].astype(np.int), kp1[i][1].astype(np.int))
        center2 = (kp2[i][0].astype(np.int), kp2[i][1].astype(np.int))

        color = np.random.randint(0, 255, 3)
        color = tuple([int(i) for i in color])

        cv2.line(out_img, center1, center2, color, 2)
    return out_img


def draw_match_2_side(img1, kp1, img2, kp2, N):
    """Draw matches on 2 sides
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                                            dtype=np.int
                                            )

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    return out_img

def trajectory_distances(poses):

    
    relpose = {}

    """Compute distance for each pose w.r.t frame-0
    Args:
        poses (dict): {idx: 4x4 array}
    Returns:
        dist (float list): distance of each pose w.r.t frame-0
    """
    dist = [0]
    sort_frame_idx = sorted(poses.keys())
    relpose[0] = poses[0]
    for i in range(len(sort_frame_idx)-1):
        # if i>800:
        #     break
        cur_frame_idx = sort_frame_idx[i]
        next_frame_idx = sort_frame_idx[i+1]
        
        
        
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        
        cur_pred = np.linalg.inv(P1) @ P2
        # print(cur_pred)
        relpose[i+1] = cur_pred

    return relpose


class FrameDrawer():
    """
    Attributes
        h (int): drawer image height
        w (int): drawer image width
        img (hxwx3): drawer image
        data (dict): linking between item and img
        display (dict): options to display items
    """
    def __init__(self, h, w):
        """
        Args:
            h (int): drawer height
            w (int): drawer width
        """
        self.h = h
        self.w = w
        self.img = np.zeros((h, w, 3), dtype=np.uint8)
        self.data = {}
        self.display = {}
        self.colgrad = 0

    def assign_data(self, item, top_left, bottom_right):
        """assign data to the drawer image
        Args:
            top_left (list): [y, x] position of top left corner
            bottom_right (list): [y, x] position of bottom right corner
            item (str): item name
        """
        self.data[item] = self.img[
                                    top_left[0]:bottom_right[0],
                                    top_left[1]:bottom_right[1]
                                    ]
        self.display[item] = True

    def update_data(self, item, data):
        """update drawer content
        Args:
            item (str): item to be updated
            data (HxWx3 array): content to be updated, RGB format
        """
        data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        vis_h, vis_w, _ = self.data[item].shape
        self.data[item][...] = cv2.resize(data_bgr, (vis_w, vis_h))

    def update_display(self, item):
        """update display option by inversing the current setup
        Args:
            item (str): item to be updated
        """
        self.display[item] = not(self.display[item])

    def get_traj_init_xy(self, vis_h, vis_w, gt_poses):
        """Get [x,y] of initial pose
        Args:
            vis_h (int): visualization image height
            vis_w (int): visualization image width
            gt_poses (dict): ground truth poses
        Returns:
            [x_off, y_off] (int): x,y offset of initial pose
        """
        # Get max and min X,Z; [x,y] of
        gt_Xs = []
        gt_Zs = []
        for cnt, i in enumerate(gt_poses):
            trueX, trueY, trueZ = gt_poses[i][:3, 3]
            gt_Xs.append(trueX)
            gt_Zs.append(trueZ)
            if cnt == 0:
                x0 = trueX
                z0 = trueZ
        min_x, max_x = np.min(gt_Xs), np.max(gt_Xs)
        min_z, max_z = np.min(gt_Zs), np.max(gt_Zs)

        # Get ratio
        ratio_x = (x0 - min_x)/(max_x-min_x)
        ratio_z = (z0 - min_z)/(max_z-min_z)

        # Get offset (only using [0.2:0.8])
        crop = [0.1, 0.9]
        x_off = int(vis_w * (crop[1]-crop[0]) * ratio_x + vis_w * crop[0])
        y_off = int(vis_h * crop[1] - vis_h * (crop[1]-crop[0]) * (ratio_z))
        self.traj_x0, self.traj_y0 = x_off, y_off
        # return x_off, y_off

    def interface(self):
        key = cv2.waitKey(10) or 0xff

        # pause
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) or 0xff

                # Match_temp
                if key2 == ord('1'):
                    self.display['match_temp'] = not(self.display['match_temp'])
                    print("Match(1): {}".format(self.display['match_temp']))

                # Match side
                if key2 == ord('2'):
                    self.display['match_side'] = not(self.display['match_side'])
                    print("Match(2): {}".format(self.display['match_side']))

                # depth
                if key2 == ord('3'):
                    self.display['depth'] = not(self.display['depth'])
                    print("depth: {}".format(self.display['depth']))

                if key2 == ord('4'):
                    self.display['flow1'] = not(self.display['flow1'])
                    self.display['flow2'] = not(self.display['flow2'])
                    self.display['flow_diff'] = not(self.display['flow_diff'])
                    print("flow: {}".format(self.display['flow1']))

                # Continue
                if key2 == ord('c'):
                    return

        # Match_temp
        if key == ord('1'):
            self.display['match_temp'] = not(self.display['match_temp'])
            print("Match(1): {}".format(self.display['match_temp']))

        # Match side
        if key == ord('2'):
            self.display['match_side'] = not(self.display['match_side'])
            print("Match(2): {}".format(self.display['match_side']))

        # depth
        if key == ord('3'):
            self.display['depth'] = not(self.display['depth'])
            print("depth: {}".format(self.display['depth']))

        # flow
        if key == ord('4'):
            self.display['flow1'] = not(self.display['flow1'])
            self.display['flow2'] = not(self.display['flow2'])
            self.display['flow_diff'] = not(self.display['flow_diff'])
            print("flow: {}".format(self.display['flow1']))

    def draw_traj(self, pred_poses, gt_poses, idx, color = (0, 255, 0)):
        """draw trajectory and related information
        Args:
            pred_poses (dict): predicted poses w.r.t world coordinate system
        """
        traj = self.data["traj"]
        latest_id = i
        # print(latest_id)

        # draw scales
        draw_scale = 1
        mono_scale = 1
        pred_draw_scale = draw_scale * mono_scale

        # Draw GT trajectory

        cur_t = gt_poses[latest_id][:3,3]
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        draw_x = int(x*pred_draw_scale) + self.traj_x0
        draw_y = -(int(z*pred_draw_scale)) + self.traj_y0
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0,255), 1)

        
        # Draw prediction trajectory
        cur_t = pred_poses[latest_id][:3,3]
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        draw_x = int(x*pred_draw_scale) + self.traj_x0
        draw_y = -(int(z*pred_draw_scale)) + self.traj_y0
        self.colgrad += 0.5 
        cv2.circle(traj, (draw_x, draw_y), 1, color, 1)

        # Draw coordinate information
        cv2.rectangle(traj, (600, 20), (600+int(self.w/2), 60), (0, 0, 0), -1)
        text = "Coordinates, Inidex: x=%2fm y=%2fm z=%2fm i = %d" % (x, y, z, idx)
        cv2.putText(traj, text, (600+20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        

    def main(self, pred_poses, gt_poses, i, out, lent, color):
        start_time = time()
        self.draw_traj(
                pred_poses=pred_poses, gt_poses=gt_poses, idx = i, color = color)
        cv2.imshow('DF-VO', self.img)
        ch = cv2.waitKey(1)
        if i == lent-1:
            # print("DONE", i, lent)		
            cv2.imwrite('results/trajsaved.png', self.img[300:835, 435:1050])


drawer = FrameDrawer(1280, 1920)
visual_h = 1280
visual_w = 1920
visual_h_gt = 900
visual_w_gt = 1500
drawer.assign_data(
            item="traj",
            top_left=[0, 0], 
            bottom_right=[int(visual_h), int(visual_w)],
            )

drawer.assign_data(
            item="match_temp",
            top_left=[int(visual_h/4*0), int(visual_w/4*2)], 
            bottom_right=[int(visual_h/4*1), int(visual_w/4*4)],
            )

drawer.assign_data(
            item="match_side",
            top_left=[int(visual_h/4*1), int(visual_w/4*2)], 
            bottom_right=[int(visual_h/4*2), int(visual_w/4*4)],
            )

drawer.assign_data(
            item="depth",
            top_left=[int(visual_h/4*2), int(visual_w/4*2)], 
            bottom_right=[int(visual_h/4*3), int(visual_w/4*3)],
            )

drawer.assign_data(
            item="flow1",
            top_left=[int(visual_h/4*2), int(visual_w/4*3)], 
            bottom_right=[int(visual_h/4*3), int(visual_w/4*4)],
            )

drawer.assign_data(
            item="flow2",
            top_left=[int(visual_h/4*3), int(visual_w/4*2)], 
            bottom_right=[int(visual_h/4*4), int(visual_w/4*3)],
            )

drawer.assign_data(
            item="flow_diff",
            top_left=[int(visual_h/4*3), int(visual_w/4*3)], 
            bottom_right=[int(visual_h/4*4), int(visual_w/4*4)],
            )



def load_poses_from_txt(file_name):
    """Load poses from txt (KITTI format)
    Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!=""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses

out = 0


color = (0, 255, 0)
do_umeyama = True

color_dict = {'r2d2' : (0,255,0), 'sift' : (255, 0, 255), 'surf' : (255, 255, 0), 'orb' : (0, 255, 255)}

errors_dict = {}
for seqno in range(3, 5):


    seq_dict = {}
    for method in ['r2d2']: #add in more methods here such as sift, surf etc.
        drawer = FrameDrawer(1280, 1920)
        visual_h = 1280
        visual_w = 1920
        visual_h_gt = 900
        visual_w_gt = 1500
        drawer.assign_data(
                item="traj",
                top_left=[0, 0], 
                bottom_right=[int(visual_h), int(visual_w)],
                )
        
        method_dict = {}
        print("METHOD, SEQ NO", method, seqno)
        seq_no = '%02d'%seqno
        if do_gt == False:
            drawer.traj_x0, drawer.traj_y0 = (200, 500)

        else:
            gt_poses = load_poses_from_txt('/home/olorin/Downloads/IISC_Tests/data_odometry_poses/dataset/poses/%s.txt'%(seq_no))
            rel_gt_poses = trajectory_distances(gt_poses)
            drawer.get_traj_init_xy(visual_h_gt, visual_w_gt, gt_poses)
        with open('/home/olorin/Desktop/IISc/OdometryProject/kitti-comparisons/Outputs_stereo/%s_%02d.pkl'%(method, int(seq_no)), 'rb') as f: pred_poses_x = pickle.load(f)

        transarrgt= []
        transdictgt = {}
        rmatr_basegt = np.eye(3)
        tvec_basegt = np.array([0,0,0])

        transarr = []
        rmatr_base = np.eye(3)
        tvec_base = np.array([0,0,0])

        transformationarr_pred = {}
        transdict_pred = {}

        transformationarr_gt = {}
        transdict_gt = {}
        lent = len(pred_poses_x)
        # print(pred_poses_x[0], len(pred_poses_x))
        # print(lent)
        # done
        dict_ctr = 0
        skipped_ctr = 0
        prev_trans = np.asarray([0.0, 0.0, 0.0])
        for i, idx in enumerate(pred_poses_x):
            pose = SE3()

            if do_gt:
                gt_pose = SE3(gt_poses[i])
            else:
                gt_pose = SE3()

            # POSE READER. SHOULD BE A 4x4 MATRIX
            pose_cur = SE3(pred_poses_x[idx])



            rmatr = pose_cur.R
            tvec = np.squeeze(pose_cur.t)

            if np.linalg.norm(prev_trans - gt_pose.t) <= 0.05:
                skipped_ctr += 1
                continue


            tvec_base *= 0
            rmatr_base = np.eye(3)

            # tvec = tvec / np.linalg.norm(tvec)
            
            dict_ctr+=1
            tvec_base = tvec_base + rmatr_base @ tvec
            rmatr_base = rmatr_base @ rmatr


            pose.R = rmatr_base
            pose.t = tvec_base.reshape((3,1))
            transformationarr_pred[dict_ctr] = pose
            transdict_pred[dict_ctr] = pose.pose

            if do_gt:
                prev_trans = gt_pose.t.copy()

            transformationarr_gt[dict_ctr] = gt_pose
            transdict_gt[dict_ctr] = gt_pose.pose
        print(skipped_ctr)
        # done

        if do_umeyama and do_gt:
            xyz_gt = []
            xyz_result = []
            for cnt in transdict_pred:
                xyz_gt.append([transdict_gt[cnt][0, 3], transdict_gt[cnt][1, 3], transdict_gt[cnt][2, 3]])
                xyz_result.append([transdict_pred[cnt][0, 3], transdict_pred[cnt][1, 3], transdict_pred[cnt][2, 3]])
            # print(len(xyz_gt), len(xyz_result))
            xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
            xyz_result = np.asarray(xyz_result).transpose(1, 0)
            # print(xyz_result.shape)


            r, t, scale = umeyama_alignment(xyz_result, xyz_gt, True)
            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t
            # print(scale, t)

            for cnt in transdict_pred:
                transdict_pred[cnt][:3, 3] *= scale
                transdict_pred[cnt] = align_transformation @ transdict_pred[cnt]



        lent = len(transdict_pred)
        for i in range(1, len(transdict_pred)+1):
            drawer.main(transdict_pred, transdict_gt, i, out, lent, color_dict[method])


        posesdict = convert_SE3_to_arr(transformationarr_pred)
        save_traj('results/transformedorb.txt', posesdict)
        if do_gt:
            posesdict = convert_SE3_to_arr(transformationarr_gt)
            save_traj('results/transformedgt.txt', posesdict)

        if do_gt:
            keo = KittiEvalOdom()
            ate_error, rpe_error, rot_error, tot_distance = keo.eval('resdir',1, alignment='6dof')
            seq_dict['Tot_Dist'] = tot_distance
            method_dict['ATE'] = ate_error
            method_dict['RPE'] = rpe_error
            # method_dict['Rotation RPE Error'] = rot_error
    
        seq_dict[method] = method_dict
        print(seq_dict)
        cv2.waitKey(0)
        # cv2.write('/home/Desktop/img.png', )
            

    errors_dict[seq_no] = seq_dict
        # cv2.waitKey(10)
print(errors_dict)
d1 = {'level_0':'Sequence Number','level_1':'Which Error'}
df = pd.concat({k: pd.DataFrame(v) for k,v in errors_dict.items()}).reset_index().rename(columns=d1)
# print(df)
# df.to_csv('Errors_noskip_noumeyama_5030_xzyaw1.csv', float_format='%.4f')
