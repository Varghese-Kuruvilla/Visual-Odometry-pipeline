import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from PIL import Image
from VisualOdometry_Mono import VisualOdometry
# from seg_utils import *
from park_utils import *
import time

idx = 0

numpyseeds = [4123]

H = np.asarray([[-5.980100603895147, -7.019381724070260, 328.403053534006403],[-4.248995325137237, -0.819454703998416, -984.512856940261827],[-0.019079413553497, -0.003101259290371, 1.000000000000000]])
cam_intr = np.asarray([[332.9648157406122, 0.0, 310.8472797033171], [0.0, 444.0950902369522, 252.76060777256825], [0.0, 0.0, 1.0]])
midpoints = [(168, 352), (175, 328), (183, 313), (204, 306), (187, 296), (203, 292), (229, 279), (235, 269), (235, 269)]

# weights_path = 'model_final_final_iisc_idd_16kweights.pth'
# auto_park_obj = auto_park_vision(weights_path)
points_ls = [[620,200,1],[620,-200,1],[1120,-200,1],[1120,200,1]]
points_2d_ls = world_2d(points_ls, H) #[967,427],[295,438],[438,309],[817,300]

#Run it for just one frame
seq_no = 0
print(seq_no + 5)
for numpyseed in numpyseeds:
  np.random.seed(numpyseed)
  images_directory = "/home/volta-2/VO/data/mar8seq/00/"
  vo = VisualOdometry(cam_intr, H, seq_no)
  start_no = 0
  for index, file_path in enumerate(sorted(glob.glob(images_directory + "left*"))):
      if index ==1 :
        continue
      # print(file_path)
      frame = cv2.imread(file_path)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LANCZOS4)
      frame_pil = Image.fromarray(frame)
      midpoint_2D = midpoints[seq_no]

      seg_img = cv2.imread("/home/volta-2/VO/data/mar8seq/seg_000000.png",0)
      # seg_img = auto_park_obj.forward_pass(frame_pil,img_path=None)
      start_time = time.time()
      vo.process_frame(frame, seg_img, midpoint_2D, index,  file_path)
      print("Time taken:",time.time() - start_time)
      if(vo.cur_data != None):  
        print(vo.cur_data.pose.t.T * vo.global_scale)

      # if(index == 0):
      #   break
      # cv2.waitKey(1)
  break 
print("Done")
# print(vo.cur_data.pose.t.T * vo.global_scale)
    
    # print(vo.cur_data.pose.t.T)
      