# %%writefile /content/Vis_Odometry/main_cell.py
# Parking Spot
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
#from sklearn import linear_model
import os
# from SURF import *
# from SIFT import *
from R2D2 import *

from frame_utils import *
from park_utils import *
from geom_utils import *
from SE3_utils import SE3

np.random.seed(4123)

listlength = 1

def append_to_list(l, ele, listlen= listlength):
    l.append(ele)
    return l[-listlen:]

class VisualOdometry:
        def __init__(self, cam_intr, homo_matrix, seqno):

 
                self.cam_intr = cam_intr.copy()            

                H = homo_matrix.copy()
                                                                                                     

                self.homography_matrix = H.copy()

                self.ref_data = []
                self.global_poses = {0: SE3().pose}
                self.global_pose = SE3()
                self.pose_ctr = 0
                self.cur_data = None
                self.frame_pairs = []
                self.viz = True
                self.scale = 1
                self.img_id = 0

                self.no_matches = []
                self.inlier_pts = []
                self.cheirality_pts = []


                self.ref_kps_index = None
                self.ref_pose = SE3()
                self.good_points_3D = None

                # self.contour = np.array([[405,390],[615,350],[605,450],[385,480] ], dtype=np.int32)

                self.initialized = False
                self.pose_ctr = 0

                self.seq_no = seqno
                

                self.ref_len = 1
                self.method = 'learned'

                self.essentialMat = {'iter' : 3, 'thresh' : 0.3, 'prob' : 0.99}

                if os.path.exists(self.method.lower()) == False:
                        os.mkdir(self.method.lower())


        def update_global_pose(self, poss):
            self.global_pose.t += self.global_pose.R @ poss.t
            self.global_pose.R = self.global_pose.R @ poss.R

        def computepose_2D_2D(self,framepair):

                # 
                best_inlier_cnt = 0
                retval = False
                pose = SE3()
                left_kp, right_kp = framepair.getkeypts()

                E1, inliers = cv2.findEssentialMat(right_kp.copy(), left_kp.copy(),
                                                                                    self.cam_intr,
                                                                                    method = cv2.RANSAC,
                                                                                    prob = self.essentialMat['prob'], threshold = 5.0)

                inliers = np.squeeze(inliers)
                inliers_copy_og = inliers.copy()
                pts, R1, t1, _ = cv2.recoverPose(E1, right_kp.copy(), left_kp.copy(), self.cam_intr, mask = inliers_copy_og.copy())

                left_kp = left_kp.copy()[inliers_copy_og==1]
                right_kp = right_kp.copy()[inliers_copy_og==1]
                framepair.left_kp = left_kp.copy()
                framepair.right_kp = right_kp.copy()
                framepair.left_kp_og = left_kp.copy()
                framepair.right_kp_og = right_kp.copy()

                framepair.frame_index = framepair.frame_index[inliers_copy_og==1]

                # self.vizualize_custom_matches(framepair.frame1.image, framepair.frame2.image, left_kp, right_kp)
                


                avgoptflow = np.sum(np.linalg.norm(left_kp-right_kp, axis = 1))/len(right_kp)
                if avgoptflow < 5:
                #     print("NO PARALLAX")
                    return retval, framepair
                
                if pts > 70:
                    pose.R = R1
                    pose.t = t1
                    retval = True
                    framepair.pose = pose
                else:
                    retval = False
                    return retval, framepair


                for loopindex in range(self.essentialMat['iter']):

                        
                        new_list = np.random.randint(0, left_kp.shape[0], (left_kp.shape[0]))
                        new_left_kp = left_kp.copy()[new_list]
                        new_right_kp = right_kp.copy()[new_list]
                        E, inliers = cv2.findEssentialMat(new_right_kp, new_left_kp,
                                                                                            self.cam_intr,
                                                                                            method = cv2.RANSAC,
                                                                                            prob = self.essentialMat['prob'], threshold = self.essentialMat['thresh'])
                        inliers = np.squeeze(inliers)
                        inliers_copy = inliers.copy()
                        left_kp_inliers = new_left_kp.copy()[inliers_copy==1]
                        right_kp_inliers = new_right_kp.copy()[inliers_copy==1]
                        avgoptflow = np.sum(np.linalg.norm(left_kp_inliers-right_kp_inliers, axis = 1))/len(left_kp_inliers)
                        if avgoptflow < 5:
                                # print("LESS THAN 5")
                                continue
                        points, R_tmp, t_tmp, _ = cv2.recoverPose(E, new_right_kp, new_left_kp, self.cam_intr, mask = inliers_copy)

                        #ADJUSTMENT STEP

                        # t_tmp[1] = 0
                        # t_tmp = t_tmp / np.linalg.norm

                        if points < 50:
                                print("LESS THAN 50", points)
                                continue
                        
                        if inliers.sum() > best_inlier_cnt and points>50:
                                retval = True
                                best_inlier_cnt = inliers.sum()
                                
                                pose.R = R_tmp
                                pose.t = t_tmp
                                
                                framepair.cheirality_pts_ct = points
                                framepair.inlier_pts_ct = best_inlier_cnt
                                framepair.pose = pose
                                framepair.ess_mat = E
                                framepair.inl = inliers


                return retval, framepair


        def computepose_3D_2D(self,framepair):

                # 
                best_inlier_cnt = 0
                retval = False
                pose = SE3()

                left_kp, right_kp = framepair.getkeypts()

                # self.vizualize_custom_matches(framepair.frame1.image, framepair.frame2.image, left_kp, right_kp, 'BEFORE ESSMAT')
                # print("PNP BETWEEN ", framepair.frame1.filename, framepair.frame2.filename, len(left_kp))

                E1, inliers = cv2.findEssentialMat(right_kp.copy(), left_kp.copy(),
                                                                                    self.cam_intr,
                                                                                    method = cv2.RANSAC,
                                                                                    prob = self.essentialMat['prob'], threshold = 3.0)

                inliers = np.squeeze(inliers)
                inliers_copy_og = inliers.copy()


                left_kp = left_kp.copy()[inliers_copy_og==1]
                right_kp = right_kp.copy()[inliers_copy_og==1]
                # print(len(left_kp), len(inliers_copy_og))
                # print(framepair.frame_index)

                # print("#################### ESS MAT INLIERS : ", len(left_kp), " OUT OF ", len(inliers))
                # self.vizualize_custom_matches(framepair.frame1.image, framepair.frame2.image, left_kp, right_kp, 'AFTER ESSMAT')

                points_3D = self.good_points_3D[framepair.frame_index[..., 0]][inliers_copy_og==1]



                framepair.left_kp = left_kp.copy()
                framepair.right_kp = right_kp.copy()
                framepair.left_kp_og = left_kp.copy()
                framepair.right_kp_og = right_kp.copy()
                framepair.frame_index = framepair.frame_index[inliers_copy_og==1]


                avgoptflow = np.sum(np.linalg.norm(left_kp-right_kp, axis = 1))/len(right_kp)
                if avgoptflow < 5 or len(left_kp) < 50:
                #     print("NO PARALLAX")
                    return retval, framepair, 1000, 0
                

                # print("DOING PNP with %d points out of %d points between %s and %s"%(len(i1), len(left_kp), framepair.frame1.filename, framepair.frame2.filename))
                best_rt = []
                best_inliers = None
                best_inlier = 0
                
                # t1 = time.time()
                for loopindex in range(self.essentialMat['iter']):
                     
                        new_list = np.random.randint(0, points_3D.shape[0], (points_3D.shape[0]))
                        new_left_kp = left_kp.copy()[new_list]
                        new_right_kp = right_kp.copy()[new_list]
                        new_points_3D = points_3D.copy()[new_list]
                        # print("$$$$$ PNP ", len(new_right_kp))
                        # flag, r, t = cv2.solvePnP(objectPoints = new_points_3D, imagePoints = new_right_kp, cameraMatrix = self.cam_intr, distCoeffs = None, rvec = cv2.Rodrigues(R1_tmp)[0], tvec = t1_tmp, useExtrinsicGuess = True, flags = cv2.SOLVEPNP_ITERATIVE)
                        # inlier = np.random.rand(60,2)
                        # print(inlier.shape[0])
                        flag, r, t, inlier = cv2.solvePnPRansac(objectPoints = new_points_3D, imagePoints = new_right_kp, cameraMatrix = self.cam_intr, distCoeffs = None, iterationsCount = 100, reprojectionError=1.50)
                        # flag, r, t, inlier = cv2.solvePnPRansac(objectPoints = new_points_3D, imagePoints = new_right_kp, cameraMatrix = self.cam_intr, distCoeffs = np.array([0.0427517,     0.20910611,    0.00161125,    0.00081772, -0.79102898]), iterationsCount = 1000, reprojectionError=1.75)
                        # print(inlier.shape)

                        if flag and inlier.shape[0] > best_inlier and inlier.shape[0] > 25:
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

                return retval, framepair, len(left_kp), best_inlier



        def vizualize_matches(self,framepair, imgidx, val):
                cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.left_kp]
                cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.right_kp]
                dmtches = [cv2.DMatch(_imgIdx=0,_queryIdx=i, _trainIdx=i, _distance = 0) for i in range(len(cv_kp1))]
                out_img = cv2.drawMatches(framepair.frame1.image, cv_kp1, framepair.frame2.image, cv_kp2, dmtches[::50], None, (0,255,255), -1, None, 2)
                # cv2.imwrite('outfolder/%06d_match.jpg'%imgidx, out_img)
                cv2.imshow('match', out_img)


        def vizualize_custom_matches(self,img1, img2, kps1, kps2, name = 'match_inl'):
                cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kps1]
                cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kps2]
                dmtches = [cv2.DMatch(_imgIdx=0,_queryIdx=i, _trainIdx=i, _distance = 0) for i in range(len(cv_kp1))]
                out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, dmtches[::5], None, (0,255,255), -1, None, 2)
                cv2.imwrite('outfolder/%06d_match_inl.jpg'%self.img_id, out_img)


        def vizualize_kps(self,img1, img2, kps, val, to_update = False, draw_contour = False):


              cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=5) for pt in kps]
              if img1!=None:
                out_img = cv2.drawKeypoints(img1.astype(np.uint8), cv_kp1, None, (0,255,255))
                cv2_imshow(out_img)
                cv2.imwrite('outfolder/%06d_d.jpg'%val, out_img)
              # if img2 != None:
              out_img = cv2.drawKeypoints(img2.astype(np.uint8), cv_kp1, None, (0,255,255))
        #       out_img = cv2.circle(img2.astype(np.uint8),(kps[0][0],kps[0][1]),5, (255, 255, 0),thickness=2, lineType=8, shift=0) 
              # cv2_imshow(out_img)
              if draw_contour:
                out_img = cv2.drawContours(out_img, [self.contours], 0, (0,255,0), 3)

            #   cv2.imwrite('outfolder/t%d_%s.jpg'%(self.seq_no, val), out_img)
        #       cv2.imshow('PointsOnRoad', out_img)

        def update_frames_data(self, framepair):
                self.ref_data = append_to_list(self.ref_data, self.cur_data, 2)
                self.good_points_3D = transform_points(self.good_points_3D, framepair.pose.inv_pose)
        
        def update_framepairs(self, fp):
                self.frame_pairs = append_to_list(self.frame_pairs, fp, 4)




        def triangulate_new_ref_points(self, framepair):

                ref_kp, ref_desc = framepair.frame1.get_kp_desc()
                cur_kp, cur_desc = framepair.frame2.get_kp_desc()
                matches = get_matches(ref_kp, ref_desc, cur_kp, cur_desc, framepair.frame1.image.shape)
                left_kp = ref_kp[matches[:, 0], : 2].astype(np.float32)
                right_kp = cur_kp[matches[:, 1], : 2].astype(np.float32)

                diff = np.linalg.norm(left_kp- right_kp, axis=1)
                diff_mask = diff > 5
                matches = matches[diff_mask]
                left_kp = left_kp[diff_mask]
                right_kp = right_kp[diff_mask]
                
                E1, inliers = cv2.findEssentialMat(left_kp.copy(), right_kp.copy(),
                                                   self.cam_intr, method = cv2.RANSAC,
                                                   prob = self.essentialMat['prob'], threshold = 3.0)

                
                inliers = np.squeeze(inliers)
                inliers_copy_og = inliers.copy()
                left_kp = left_kp.copy()[inliers_copy_og==1]
                right_kp = right_kp.copy()[inliers_copy_og==1]
                matches = matches[inliers_copy_og==1]

                # self.vizualize_custom_matches(framepair.frame1.image, framepair.frame2.image, left_kp, right_kp, "triangulation")

                frame_3D_pts, _, _ = triangulation(left_kp, right_kp, np.eye(4), framepair.pose.inv_pose, self.cam_intr)
                non_zero_mask_tri = frame_3D_pts[...,2] > 0

                if len(non_zero_mask_tri[non_zero_mask_tri > 0]) > 50:
                        self.ref_kps_index = matches[non_zero_mask_tri]
                        self.good_points_3D = frame_3D_pts[non_zero_mask_tri]
                # print("\n -- %d points triangulated out of %d points --"%(len(self.good_points_3D), len(frame_3D_pts)))
                return len(non_zero_mask_tri[non_zero_mask_tri > 0])


        def get_point_in_other_image(self, framepair, to_update = False):

            temp_pose = framepair.frame2.pose.inv_pose
            projectedpts = cv2.projectPoints(self.midpoint_3D, cv2.Rodrigues(temp_pose[:3, :3])[0], temp_pose[:3,3], self.cam_intr, None)[0]
        #     print("IN GETPTS")

            ps_keypoint = projectedpts[0].astype(np.int32)
            cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=5) for pt in ps_keypoint]
            out_img = cv2.circle(self.cur_data.image.astype(np.uint8),(ps_keypoint[0][0],ps_keypoint[0][1]),5, (255, 255, 0),thickness=2, lineType=8, shift=0)
            # out_img = cv2.putText(out_img, "Distance of midpoint from vehicle : " + str(np.squeeze(transform_points(self.midpoint_3D, self.cur_data.pose.inv_pose))), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2 )
            # out_img = cv2.putText(out_img, "Distance travelled by vehicle : " + str(np.squeeze(self.cur_data.pose.t.T)), (200,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2 )
        #     cv2.imshow('Points', out_img)
        #     cv2.waitKey(0)


            # self.vizualize_kps(None, self.cur_data.image, projectedpts[0].astype(np.int32), self.img_id, to_update)
        #     cv2.waitKey(0)


        def save_poses(self):
                with open('r2d2parking.pkl', 'wb') as f: pickle.dump(self.global_poses, f)


        def get_global_scale(self,framepair,midpoint):
                '''
                Find out the global scale to update the 3D coordinates of the midpoint
                '''
                

                seg_map = framepair.frame1.seg_img
                H = self.homography_matrix.copy()
                
                H_inv = np.linalg.inv(H)
                mid_pt_homo = np.asarray([midpoint[0], midpoint[1], 1]).reshape((3,1))
                midpoint_3D = np.dot(H_inv,mid_pt_homo)
                midpoint_3D_normalized = midpoint_3D / abs(midpoint_3D[-1])
                # print("MIDPOINT 3D " , midpoint_3D, midpoint,    midpoint_3D_normalized, framepair.pose.t.T, np.linalg.norm(framepair.pose.t.T))

                depth = abs(midpoint_3D_normalized[0] / 100)

                midpoint_3D = unprojection_kp(np.asarray([midpoint]), depth, self.cam_intr)


                corresponding_2D_pts = framepair.frame1.keypoints[self.ref_kps_index[:, 0]].astype(np.int32)[..., :2]
                seg_map_mask = seg_map[corresponding_2D_pts[..., 1], corresponding_2D_pts[..., 0]] == 255
                # done
                # print("seg_map_mask.shape",seg_map_mask.shape)
                # print("corresponding_2D_pts.shape",corresponding_2D_pts.shape)
                cnt_keypoints = corresponding_2D_pts[seg_map_mask]
                
                # print(cnt_keypoints)
                cnt_keypoints_homo = np.c_[cnt_keypoints,np.ones(len(cnt_keypoints))]


                good_road_points_3D = self.good_points_3D[seg_map_mask]
                # print(good_road_points_3D.shape)


                # Find 3D points
                cnt_keypoints_3d = []
                for pt in cnt_keypoints_homo:
                        pt = pt.reshape((3,1))
                        cnt_keypoints_3d.append(np.dot(H_inv,pt))
                
                
                cnt_keypoints_3d = np.squeeze(np.asarray(cnt_keypoints_3d))
                cnt_keypoints_3d_normalized = np.divide(cnt_keypoints_3d.T, abs(cnt_keypoints_3d[..., -1])).T

                cnt_keypoints_depth = abs(cnt_keypoints_3d_normalized[..., 0] / 100)
                cnt_keypoints_3d = unprojection_kp(cnt_keypoints, cnt_keypoints_depth, self.cam_intr)
                cnt_depth_mask = cnt_keypoints_depth < 40
                # self.vizualize_kps(None, framepair.frame1.image,cnt_keypoints[cnt_depth_mask], 0 )

                if len(cnt_keypoints_3d[cnt_depth_mask]) > 30:
                        true_scale = find_scale(good_road_points_3D[cnt_depth_mask], cnt_keypoints_3d[cnt_depth_mask])
                else:
                    print("TOO FEW POINTS, CANNOT FIND SCALE!")
                    return -1



                self.midpoint_3D = midpoint_3D.reshape((1,3))
                self.midpoint_3D /= true_scale
                # cv2.waitKey(1)
                return true_scale


        def process_frame(self, img, seg_img, midpoint, frame_no, imgfile):
                '''
                contours: Area within which keypoints are to be tracked
                img: OpenCV image
                midpoint: Midpoint to be tracked
                frame_no: Frame number
                '''
                #NOTE:
                #Use midpoint and contours for 2 frames i.e until we have established 2D-2D correspondencies
                #Followed by this use pure Visual Odometry for servoing

                # fil = self.imgfiles[0]
                # print("img.size()",img.size())
                #For the first frame
                if(frame_no == 0): #First frame
                        if self.method != 'OF':
                                # img = cv2.imread(fil)
                                #Converting OpenCV image into PIL format
                                #Original Image: img
                                r2d2_start = time.time()
                                kp, desc = extract_features_and_desc(img,trt=True)
                                print("Time taken for R2D2",time.time()-r2d2_start)                
                                frame1 = Frame(id = 0, img = img, kps = kp, desc = desc, fil = imgfile, pose = SE3(), seg_img = seg_img)
                                
                        
                        self.ref_data = append_to_list(self.ref_data, frame1)
                        self.initialized = False
                        self.parking_spot_points_found = False
                
                else: #NOTE: frame_no = imgidx
                        cur_img = img
                        imgidx = frame_no
                        cur_fil = imgfile
                        self.img_id = imgidx
                                
                        if self.method != 'OF':
                                # cur_img = cv2.imread(cur_fil)
                                # cur_kp, cur_desc = extract_features_and_desc(cur_img, cur_fil)
                                r2d2_start = time.time()
                                cur_kp, cur_desc = extract_features_and_desc(cur_img,trt= True)
                                print("Time taken for R2D2:",time.time()-r2d2_start)
                                frame1 = self.ref_data[-1]
                                ref_kp, ref_desc = self.ref_data[-1].get_kp_desc()
                                # self.vizualize_kps(None,cur_img,cur_kp, 0 )
                                frame2 = Frame(id = imgidx, img = cur_img, kps = cur_kp, desc = cur_desc, fil = cur_fil, pose = SE3(), seg_img = seg_img)
                                self.cur_data = frame2
                                # print(frame1.kps_index)
                                matches = get_matches(ref_kp[frame1.kps_index], ref_desc[frame1.kps_index], cur_kp, cur_desc, cur_img.shape)
                                try:
                                        ref_keypoints = ref_kp[frame1.kps_index[matches[:, 0]], : 2].astype(np.float32)
                                except Exception as e:
                                        
                                        print(e, matches.shape, frame1.kps_index.shape, ref_kp.shape)

                                # ref_keypoints = ref_kp[matches[:, 0], : 2].astype(np.float32)
                                cur_keypoints = cur_kp[matches[:, 1], : 2].astype(np.float32)
                        
                                framepair = FramePair(frame1, frame2, matches, ref_keypoints, cur_keypoints, matches)

                        
                        if self.initialized == False:
                            retval, framepair = self.computepose_2D_2D(framepair)
                        

                        else:
                            retval, framepair, common_pts, best_inliers = self.computepose_3D_2D(framepair)
                            scale = np.linalg.norm(framepair.pose.t)
                        


                        # if self.img_id > 145:
                        #     imdone

                        to_update = False
                        if retval==True:
                                
                                if self.initialized == False:
                                        self.triangulate_new_ref_points(framepair)
                                        global_scale = self.get_global_scale(framepair,midpoint)
                                        # if global_scale < 0.70 or global_scale > 4.00:
                                        if global_scale < 0.50 or global_scale > 4.00:
                                            # print("************************VERY POOR*******************", global_scale)
                                            return 
                                        
                                        framepair.frame1.kps_index = self.ref_kps_index[:, 0]
                                        framepair.frame2.kps_index = self.ref_kps_index[:, 1]
                                        
                                        # print("FOUND ABSOLUTE SCALE : ", global_scale, framepair.frame1.filename, framepair.frame2.filename, self.midpoint_3D * global_scale)
                                        scale = global_scale
                                        self.global_scale = scale
                                        # framepair.pose.t *= scale
                                        # self.good_points_3D *= scale
                                        common_pts = 250
                                        best_inliers = 100                        
                                        self.initialized=True
                                        to_update = True
                                

                                # if scale > 2.0 or common_pts < 200 :
                                if scale > 1.50 or common_pts < 200 :
                                        # print(scale, common_pts, best_inliers)
                                        # print("INSERTING KEYFRAME")
                                        no_of_triangulated = self.triangulate_new_ref_points(framepair)
                                        if no_of_triangulated < 50:
                                                # print("BAD TRIANGULATION ", no_of_triangulated)
                                                return 
                                        framepair.frame1.kps_index = self.ref_kps_index[:, 0]
                                        framepair.frame2.kps_index = self.ref_kps_index[:, 1]

                                        # scale = self.get_local_scale(framepair)     
                                        # print("LOCAL SCALE : ", scale)                            
                                        to_update = True
                                        # framepair.pose.t *= scale
                                        # self.good_points_3D *= scale

                                framepair.frame2.pose._pose = framepair.frame1.pose._pose @ framepair.pose._pose.copy()

                                self.pose_ctr += 1
                                self.global_poses[self.pose_ctr] = framepair.frame2.pose._pose
                                self.get_point_in_other_image(framepair, to_update)

                                # print("Current car pos : ", framepair.frame2.pose.t.T * self.global_scale)


                                if to_update:
                                        # print("--------------------- UPDATING REFERENCE FRAME TO %s with "%(framepair.frame2.filename), framepair.frame2.pose.t.T, common_pts)
                                        self.update_frames_data(framepair)
                        else:
                                framepair.frame2.pose._pose = framepair.frame1.pose._pose.copy()


                        