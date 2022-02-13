import cv2
import numpy as np
import time
import copy
import pickle
from tqdm import tqdm
import sys
import glob
import re
import shutil
from sklearn import linear_model
import os
import yaml
np.set_printoptions(suppress=True, precision = 2)

with open("config/vo_params.yaml") as f:
        vo_params = yaml.load(f, Loader=yaml.FullLoader)

if(vo_params['feature_extractor'] == 'sift'):
        from feature_extractors.SIFT import *
elif(vo_params['feature_extractor'] == 'orb'):
        from feature_extractors.ORB import *
elif(vo_params['feature_extractor'] == 'r2d2'):
        from R2D2 import *


from Utils.frame_utils import *
from Utils.park_utils import *
from Utils.geom_utils import *
from Utils.SE3_utils import SE3
from Utils.debug_utils import *

listlength = 1

def append_to_list(l, ele, listlen= listlength):
    l.append(ele)
    return l[-listlen:]



class VisualOdometry:
        def __init__(self, camera_intrinsics, seq=0):

                
                self.cam_intr = camera_intrinsics.copy()            

                self.ref_data = []
                self.global_poses = {0: SE3().pose}
                self.global_pose = SE3()
                self.pose_ctr = 0
                self.cur_data = None
                self.frame_pairs = []
                self.viz = True
                self.scale = 1
                self.img_id = 0

                self.bad_pnp = 0

                self.no_matches = []
                self.inlier_pts = []
                self.cheirality_pts = []

                self.ref_kps_index = None
                self.ref_pose = SE3()
                self.good_points_3D = None

                self.initialized = False
                self.pose_ctr = 0
                self.ref_len = 1
                self.method = 'r2d2'

                self.find_ps_homography = False

                self.seq = seq 

                self.essentialMat = {'iter' : 3, 'thresh' : 0.3, 'prob' : 0.99}

                if os.path.exists(self.method.lower()) == False:
                        os.mkdir(self.method.lower())


        def update_global_pose(self, poss):
            self.global_pose.t += self.global_pose.R @ poss.t
            self.global_pose.R = self.global_pose.R @ poss.R


        def computepose_3D_2D(self,framepair):

                best_inlier_cnt = 0
                retval = False
                pose = SE3()

                left_kp, right_kp = framepair.getkeypts()
               
                left_depth_image = framepair.frame1.depth.copy()
                threeDImage = cv2.rgbd.depthTo3d(left_depth_image.astype(np.float32), self.cam_intr)
                left_3D_points = threeDImage[left_kp[...,1].astype(np.int32), left_kp[...,0].astype(np.int32)]
                
                #Choosing only points that are within 50 metres
                good_3D_mask = (left_3D_points[..., 2] > 0) & (left_3D_points[..., 2] < 50)


                left_3D_points = left_3D_points[good_3D_mask]
                left_kp = left_kp[good_3D_mask]
                right_kp = right_kp[good_3D_mask]

                framepair.left_kp = left_kp.copy()
                framepair.right_kp = right_kp.copy()
                framepair.left_kp_og = left_kp.copy()
                framepair.right_kp_og = right_kp.copy()
                
                if(vo_params['visualize_results'] == True):
                        self.vizualize_custom_matches(framepair.frame1.image, framepair.frame2.image, left_kp, right_kp)


                best_rt = []
                best_inliers = None
                best_inlier = 0
                
                for loopindex in range(self.essentialMat['iter']):
                        #Chooses points in a random order to be fed into solvePnPRansac
                        new_list = np.random.randint(0, left_3D_points.shape[0], (left_3D_points.shape[0]))
                        new_left_kp = left_kp.copy()[new_list]
                        new_right_kp = right_kp.copy()[new_list]
                        new_points_3D = left_3D_points.copy()[new_list]
                        
                        new_right_kp = np.ascontiguousarray(new_right_kp).reshape(new_right_kp.shape[0],1,2)
                        # print("new_right_kp:",new_right_kp)
                        flag, r, t, inlier = cv2.solvePnPRansac(objectPoints = new_points_3D, imagePoints = new_right_kp, cameraMatrix = self.cam_intr, distCoeffs = None, iterationsCount = 100, reprojectionError=1.500)
                       

                        if flag and inlier.shape[0] > best_inlier and inlier.shape[0] > 20:
                            best_rt = [r, t]
                            best_inlier = inlier.shape[0]
                            best_inliers = np.squeeze(inlier).copy()

                pose = SE3()
                if len(best_rt) != 0:
                        retval = True
                        r, t = best_rt
                        pose.R = cv2.Rodrigues(r)[0]
                        pose.t = t
                        pose.pose = pose.inv_pose.copy()
                        framepair.pose = pose
                else:
                    bad=True
                    print("NO IT IS A BAD PNP")
                    self.bad_pnp += 1
                return retval, framepair, len(left_kp), best_inlier



        def vizualize_matches(self,framepair, imgidx, val):
                cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.left_kp]
                cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.right_kp]
                dmtches = [cv2.DMatch(_imgIdx=0,_queryIdx=i, _trainIdx=i, _distance = 0) for i in range(len(cv_kp1))]
                out_img = cv2.drawMatches(framepair.frame1.image, cv_kp1, framepair.frame2.image, cv_kp2, dmtches[::50], None, (0,255,255), -1, None, 2)
                #display_image("Vizualize matches:",out_img)
                # cv2.imwrite('match', out_img)


        def vizualize_custom_matches(self,img1, img2, kps1, kps2, name = 'match_inl'):
                cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kps2]
                cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kps1]
                dmtches = [cv2.DMatch(_imgIdx=0,_queryIdx=i, _trainIdx=i, _distance = 0) for i in range(len(cv_kp1))]
                # print("len(dmtches):",len(dmtches))
                out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, dmtches[::10], None, (0,255,255), -1, None, 2)
                display_image("Feature Matching",out_img)
                # cv2.imwrite("test.jpg",out_img)
                
                cv2.imwrite('outfolder/TRT_matches/%02d_%06d_match_inl.jpg'%(self.seq, self.img_id), out_img)


        def vizualize_kps(self,img1, img2, kps, val, to_update = False):

              cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=5) for pt in kps]
              if img1.shape[0]!=0:
                out_img = cv2.drawKeypoints(img1.astype(np.uint8), cv_kp1, None, (0,255,255))

        def update_frames_data(self, framepair):
                self.ref_data = append_to_list(self.ref_data, self.cur_data, 2)
        
        def update_framepairs(self, fp):
                self.frame_pairs = append_to_list(self.frame_pairs, fp, 4)


        def get_point_in_other_image(self, framepair, to_update = False):

            temp_pose = framepair.frame2.pose.inv_pose
            projectedpts = cv2.projectPoints(self.midpoint_3D, cv2.Rodrigues(temp_pose[:3, :3])[0], temp_pose[:3,3], self.cam_intr, None)[0]
            ps_keypoint = projectedpts[0].astype(np.int32)
            cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=5) for pt in ps_keypoint]
            out_img = cv2.circle(self.cur_data.image.astype(np.uint8),(ps_keypoint[0][0],ps_keypoint[0][1]),5, (255, 255, 0),thickness=2, lineType=8, shift=0)
            out_img = cv2.putText(out_img, "Distance of midpoint from vehicle : " + str(np.squeeze(transform_points(self.midpoint_3D, self.cur_data.pose.inv_pose))), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2 )
            out_img = cv2.putText(out_img, "Distance travelled by vehicle : " + str(np.squeeze(self.cur_data.pose.t.T)), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2 )
            cv2.imshow('Points', out_img)


        def get_midpoint(self, d_img, midpoint):
            # print("GET MIDPOINT!")
            if self.find_ps_homography:
                H = self.homography_matrix.copy()
                H_inv = np.linalg.inv(H)
                mid_pt_homo = np.asarray([midpoint[0], midpoint[1], 1]).reshape((3,1))
                midpoint_3D = np.dot(H_inv,mid_pt_homo)
                midpoint_3D_normalized = midpoint_3D / abs(midpoint_3D[-1])
                # print("MIDPOINT 3D " , midpoint_3D, midpoint,    midpoint_3D_normalized, framepair.pose.t.T, np.linalg.norm(framepair.pose.t.T))
                midpoint_3D_depth = abs(midpoint_3D_normalized[0] / 100) #Midpoint of the potential parking spot in metres
            else:
                print("d_img.shape",d_img.shape)
                midpoint_3D_depth = d_img[midpoint[1], midpoint[0]]
            
        #     print("midpoint:",midpoint)
        #     print("midpoint_3d_depth:",midpoint_3D_depth)
            midpoint_3D = unprojection_kp(np.asarray([midpoint]), midpoint_3D_depth, self.cam_intr)
            self.midpoint_3D = midpoint_3D.reshape((1,3)) #3D coordinates of the midpoint


        def save_poses(self, save_name = 'r2d2parking.pkl'):
                with open(save_name, 'wb') as f: pickle.dump(self.global_poses, f)


        def process_frame(self, img, depth_img, midpoint, frame_no):
                '''
                img: OpenCV image
                midpoint: Midpoint to be tracked
                frame_no: Frame number
                '''
                #NOTE:
                #Use midpoint and contours for 2 frames i.e until we have established 2D-2D correspondencies
                #Followed by this use pure Visual Odometry for servoing
                #For the first frame
                if(frame_no == 0): #First frame   
                        kp, desc = extract_features_and_desc(img)          
                        frame1 = Frame(id = 0, img = img, kps = kp, desc = desc, fil = '%06d'%frame_no, pose = SE3(), depth = depth_img)
                        self.ref_data = append_to_list(self.ref_data, frame1)
                        self.initialized = False
                        self.get_midpoint(depth_img, midpoint)
                        return SE3()
                
                else:

                        cur_img = img
                        imgidx = frame_no
                        cur_fil = ''
                        self.img_id = imgidx
                        to_update = False
                        start_time = time.time()           
                        cur_kp, cur_desc = extract_features_and_desc(img)           
                        
                        frame1 = self.ref_data[-1]
                        ref_kp, ref_desc = self.ref_data[-1].get_kp_desc()
                        frame2 = Frame(id = imgidx, img = cur_img, kps = cur_kp, desc = cur_desc, fil = '%06d'%frame_no, pose = SE3(), depth = depth_img)
                        self.cur_data = frame2

                        matches = get_matches(ref_kp, ref_desc, cur_kp, cur_desc, cur_img.shape)
                        ref_keypoints = ref_kp[matches[:, 0], : 2].astype(np.float32)
                        cur_keypoints = cur_kp[matches[:, 1], : 2].astype(np.float32)

                        diff = np.linalg.norm(ref_keypoints- cur_keypoints, axis=1)
                      
                        #Extracting the motion only sense only for at least 3pixels of displacement between ref_keypoints and cur_keypoints
                        cur_keypoints = cur_keypoints[diff>=3]
                        ref_keypoints = ref_keypoints[diff>=3]

                        framepair = FramePair(frame1, frame2, matches, ref_keypoints, cur_keypoints, matches)

                        try:
                            retval, framepair, common_pts, best_inliers = self.computepose_3D_2D(framepair)
                            dist_scale = np.linalg.norm(framepair.pose.t)
                            if(dist_scale > 1.5 * (framepair.frame2.id - framepair.frame1.id)): #If the vehicle moves more than 1.5m between the current frame and the keyframe
                                retval = False
                                self.bad_pnp += 1
                                print("Inside false PnP condition")
                        except Exception as e:
                            print(e)
                            print("Inside bad PnP")
                            self.bad_pnp += 1
                            retval = False

                        if retval:   
                            self.bad_pnp = 0
                            framepair.frame2.pose._pose = framepair.frame1.pose._pose @ framepair.pose._pose.copy()

                            if (common_pts < 200 or best_inliers < 100 or dist_scale>1.5):
                              to_update = True
        

                        else:
                          framepair.frame2.pose._pose = framepair.frame1.pose._pose.copy()

                        self.pose_ctr += 1
                        self.global_poses[self.pose_ctr] = framepair.frame2.pose._pose
                        # self.get_point_in_other_image(framepair, to_update) #Use this function to track the midpoint
                        if to_update or self.bad_pnp >3:
                            self.update_frames_data(framepair)
                        return framepair.frame2.pose

