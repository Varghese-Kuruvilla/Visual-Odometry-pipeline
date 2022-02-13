import os
import numpy as np
import cv2
import glob
import sys
from tqdm import tqdm
from PIL import Image
from VisualOdometry_Stereo import VisualOdometry
# from seg_utils import *
from Utils.park_utils import *
import logging
import skimage.exposure
logger = logging.getLogger('module_stereo_runner')
logger.setLevel(logging.INFO)


midpoints = [(100,100)] #VO can also be used for tracking an arbitrary point in the scene. For example it can be used for 
#tracking a point on the road for parking an autonomous vehicle
numpyseeds = [8214]

#For debug
def breakpoint():
  inp = input("Waiting for input")


def display_image(name,frame):
  cv2.namedWindow(name,cv2.WINDOW_NORMAL)
  cv2.imshow(name,frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    sys.exit()


for numpyseed in numpyseeds:
  np.random.seed(numpyseed)

  
def vo_offline_data(cam_intr,img_path,output_filename):
  '''
  VO on saved rgbd images
  '''

  seq_no = 0
  global_pose = []
  vo = VisualOdometry(cam_intr,seq=0) #Seq_no: For multiple sequencees
  img_dir = images_directory = img_path

  strartno = 0
  left_image_files = sorted(glob.glob(img_dir + '/*.png')) #Choose every fourth frame
  dep_image_files = sorted(glob.glob(img_dir + '/*_depth.npy'))
  
  img_no = 0
  start_time = 0.0

  for index, (left_image_file, dep_image_file) in tqdm(enumerate(zip(left_image_files, dep_image_files))):
    if index == 1:
      start_time = time.time()
    img_no += 1
    try:
      left_frame = cv2.imread(left_image_file)
      dep_img = np.load(dep_image_file) 
    except:
      print("failed to load the image or depth file:",index)

    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    frame_pose = vo.process_frame(left_frame, dep_img, midpoints[seq_no], index)
    print("Time taken:"+str(time.time() - start_time))
    print("frame_pose.t.T" + str(frame_pose.t.T))
    global_pose.append(frame_pose.pose)
  print("Average time per frame:",(time.time() - start_time)/img_no)
  np.save(output_filename,np.asarray(global_pose))
