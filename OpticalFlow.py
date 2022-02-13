import numpy as np
import cv2
from Camera import PinholeCamera
import math
from scipy.spatial.transform import Rotation as R
import os
import glob


def draw_patch_boundary(image, u_mask, v_mask, len, name):
    val = 1
    # print(u_mask, v_mask, len)
    out_img = cv2.rectangle(image.copy(), (u_mask - len // 2, v_mask - len // 2), (u_mask + len // 2, v_mask + len // 2), color=(0, 255, 0), thickness=3)
    cv2.imshow(name, out_img)

class ImagePatch:
    def __init__(self, img_patch, u_center, v_center, length):
        self.img_patch = img_patch
        self.u = u_center
        self.v = v_center
        self.len = length
    
    def get_patch(self):
        return self.img_patch.copy()

    def get_center(self):
        return (self.u, self.v)   

    def get_length(self):
        return self.len  

class Frame:
    def __init__(self, img, fil = '', depth = None):
        self.image = img
        self.filename = fil
        self.depth = depth
        self.pose = np.eye(4)
        self.camera = PinholeCamera(1280, 720, 664.3920764204493, 664.5147822695388, 620.5068279568037, 378.8579370468793)
        self.length = self.camera.height // 4
        self.ref_img_patch = ImagePatch(None, 0, 0, self.length)
        self.matched_img_patch = ImagePatch(None, 0, 0, self.length)

    def getitems(self):
        return (self.image)

    def get_image(self):
        return self.image.copy()

    def get_file(self):
        return self.filename

    def get_img_patch(self, which = 'ref'):
        if which == 'ref':
            return self.ref_img_patch
        elif which == 'match':
            return self.matched_img_patch

    def extract_ref_patch(self, u_mask, v_mask, len):

        # print(u_mask, v_mask, len)
        u_mask = int(u_mask)
        v_mask = int(v_mask)
        len = int(len)
        img_patch = self.image[v_mask - len // 2 : v_mask + len // 2, u_mask - len // 2 : u_mask + len // 2]
        draw_patch_boundary(self.image, u_mask, v_mask, len, "REF_PATCH")
        # print(u_mask, v_mask)
        self.ref_img_patch = ImagePatch(img_patch, u_mask, v_mask, len)

    def make_matched_patch(self, u_mask, v_mask, len):
        img_patch = self.image[v_mask - len // 2 : v_mask + len // 2, u_mask - len // 2 : u_mask + len // 2]
        draw_patch_boundary(self.image, u_mask, v_mask, len, "MATCHED_PATCH")
        self.matched_img_patch = ImagePatch(img_patch, u_mask, v_mask, len)



class FramePair:
    def __init__(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2

        self.displacement = (0.0,0.0)
        self.rel_pose = None
    
    def get_frames(self):
        return (self.frame1, self.frame2)

def append_to_list(l, ele, listlen = 2):
    l.append(ele)
    return l[-listlen:]


class OpticalFlowVO:
    def __init__(self):
        self.camera = PinholeCamera(1280, 720, 664.3920764204493, 664.5147822695388, 620.5068279568037, 378.8579370468793)

        self.cur_pos = np.asarray([0.0, 0.0])
        self.delu_prev = 0.0
        self.delv_prev = 0.0
        self.l_mask = self.camera.height // 4

        self.cam2ground_dist = 1.8
        self.cam_const = (self.cam2ground_dist / self.camera.fx, self.cam2ground_dist / self.camera.fy)

        self.state = 0

        self.ref_frames = []
        self.cur_frames = []

        self.skipped = 0

        self.total_distance = 0.0


    def extract_patch(self, frame):
        u_mask = (self.camera.width * 2) // 3 - self.l_mask // 2 + self.delu_prev
        v_mask = (self.camera.height * 3) // 4 - self.l_mask * 0 // 2 + self.delv_prev

        frame.extract_ref_patch(u_mask, v_mask, self.l_mask)
    
    def match_patch(self, framepair):
        
        frame1, frame2 = framepair.get_frames()
        img = frame2.get_image()
        patch = frame1.get_img_patch('ref')

        template = patch.get_patch()
        length = patch.get_length()
        center = patch.get_center()

        draw_patch_boundary(frame1.get_image(), center[0], center[1], length, "REF IMAGE WITH PATCH ")
        # cv2.imshow("CUR IMAGE  ", img)
        # cv2.waitKey(0)

        res = cv2.matchTemplate(img[0:, :],template,cv2.TM_CCOEFF_NORMED)

        # cv2.imshow("FRR1", img[300:, :])
        # cv2.imshow("FRR2", template)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        displaced_patch_center = (top_left[0] + length//2, top_left[1] + length//2)

        # print(displaced_patch_center, center)
        
        # self.delu, self.delv = (displaced_patch_center - center)
        framepair.displacement = (center[0] - displaced_patch_center[0], center[1] - displaced_patch_center[1])

        frame2.make_matched_patch(displaced_patch_center[0], displaced_patch_center[1], length)


    def compute_pose(self, framepair):

        frame1, frame2 = framepair.get_frames()

        delx = framepair.displacement[0] * self.cam_const[0]
        dely = framepair.displacement[1] * self.cam_const[1]

        Dcam = np.asarray([delx, dely, 0.0, 1.0])
        Dcam = np.asarray([delx, 0, 0, 1])

        # To ignore very small lateral drift        
        if abs(framepair.displacement[1]) < 3:
            theta = 0.0
        else:
            # theta = math.atan2(dely * 720 / 1280, delx) - 0.031239833430268277 - 0.03524494837293129
            # theta = math.atan2(dely * 720 / 1280, delx) + 0.03524494837293129 * 2
            theta = -(math.atan2(abs(dely), abs(delx)))
        # print(framepair.displacement)
        # theta = dely / delx
        # print(delx, dely, framepair.displacement, theta)

        temp_pose = np.eye(4)
        temp_pose[:3, :3] = np.asarray([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        temp_pose[:4, 3] = Dcam.copy()

        framepair.pose = temp_pose.copy()
    
    def update_frame(self, framepair):
        self.delu_prev, self.delv_prev = (framepair.displacement[0], framepair.displacement[1])
        self.ref_frames = append_to_list(self.ref_frames, framepair.frame2)

    def process_frame(self, img):

        frame = Frame(img)
        self.extract_patch(frame)

        if self.state == 0:
            self.ref_frames = append_to_list(self.ref_frames, frame)
            self.state = 1
        
        else:
            ref_frame = self.ref_frames[-1]
            self.framepair = FramePair(ref_frame, frame)

            self.match_patch(self.framepair)
            self.compute_pose(self.framepair)

            
            self.total_distance +=  np.linalg.norm(np.asarray(self.framepair.pose[:3, 3]))
            # print("TOTAL DIST : ", self.total_distance)

            if np.linalg.norm(np.asarray(self.framepair.displacement)) > 2 and np.linalg.norm(np.asarray(self.framepair.displacement)) < 350:
            # if np.linalg.norm(np.asarray(self.framepair.displacement)) > 1 and np.linalg.norm(np.asarray(self.framepair.displacement)) < 100 :
                self.framepair.frame2.pose = self.framepair.frame1.pose @ self.framepair.pose
                self.update_frame(self.framepair)
                self.skipped = 0

            else:
                # print("NOT UPDATING")
                self.framepair.frame2.pose = self.framepair.frame1.pose @ np.eye(4)
                # print("REJECTED", self.framepair.displacement)

            # print(self.framepair.frame2.pose[:3, 3], self.framepair.pose[:3, 3], R.from_dcm(self.framepair.frame2.pose[:3, :3]).as_euler('xyz', degrees = True))

if __name__ == '__main__':



    for traj in range(0,9):
        np.random.seed(7548)

        traj_no = '%02d'%(traj+0)
        ofvo = OpticalFlowVO()

        img_dir = "/home/olorin/Desktop/IISc/OdometryProject/ParkingSpot/data/feb24/data/opticalflow/all_sequences/%02d"%traj
        imgfiles = sorted(glob.glob(img_dir + '/*'))[0::2]
        for imgfile in imgfiles:
            image = cv2.imread(imgfile, 0)
            image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_LANCZOS4)
            ofvo.process_frame(image)
            cv2.waitKey(0)
        print(ofvo.framepair.frame2.pose[:3, 3], ofvo.total_distance)