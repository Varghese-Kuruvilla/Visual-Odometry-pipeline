import numpy as np
from Utils.SE3_utils import SE3
class FramePair:
    def __init__(self, f1, f2, matches_no, left_kp, right_kp, frame1_idx = None,  cheirality_pts_ct=0, inlier_pts_ct=0, pose=SE3()):
        self.frame1 = f1
        self.frame2 = f2
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.cheirality_pts_ct = cheirality_pts_ct
        self.inlier_pts_ct = inlier_pts_ct
        self.pose = pose
        self.avg_optical_flow = 0
        self.matches_no = matches_no
        self.ess_mat = None
        self.frame_index = frame1_idx


    def getpose(self):
        # return self.pose
        pos = SE3()
        pos.t = self.pose.t.copy()
        pos.R = self.pose.R.copy()
        return pos

    def getkeypts(self):
        return (self.left_kp, self.right_kp)
    
    def getchecks(self):
        return (self.cheirality_pts_ct, self.inlier_pts_ct)

class Frame:
    def __init__(self, id, img, kps, desc, fil, pose = SE3(), seg_img = None, depth = np.zeros(1), image_arr = None, ):
        self.id = id
        self.image = img
        self.keypoints = kps
        self.descriptors = desc
        self.filename = fil
        self.depth = depth
        self.pose = pose
        self.image_arr = image_arr
        self.glob_pose = None
        self.tracked_kps = None
        self.global_pose = None
        self.kps_index = np.arange(len(kps))
        self.seg_img = seg_img

    
    def getitems(self):
        return (self.image, self.keypoints, self.descriptors, self.filename)

    def get_kp_desc(self):
        return (self.keypoints, self.descriptors)

    def get_image(self):
        return self.image

    def get_file(self):
        return self.filename
