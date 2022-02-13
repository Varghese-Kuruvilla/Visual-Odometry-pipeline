'''
Python script to plot trajectories 
'''
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import glob
import sys

#Global variables
gt_trans = []
custom_trans = []
#Debug utils
def breakpoint():
    inp = input("Waiting for input...")

def display_image(winname,frame):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(1)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit()

def read_kitti_traj(gt_txt_file):
    '''
    Read Kitti GT from the text file
    '''
    global gt_trans
    file = open(gt_txt_file,'r')
    for line in file:
        line_split = line.split(' ')
        # print("line_split",line_split)
        gt_trans.append([float(line_split[3]), float(line_split[7]), float(line_split[11])])

    gt_trans = np.asarray(gt_trans)


# def view_video():
#     img_list = sorted(glob.glob('/home/varghese/ARTPARK/UGV/Data/V.O_testing/03/image_2/*.png'))
#     for img_file in img_list:
#         img = cv2.imread(img_file)
#         display_image("image",img)



def plot_custom_traj(npy_file):
    custom_traj = np.load(npy_file)
    custom_traj = custom_traj[:,:3,3]
    fig, ax = plt.subplots()
    ax.plot(gt_trans[:,0], gt_trans[:,2], '-b', label='GT')
    ax.plot(custom_traj[:,0], custom_traj[:,2], '-r', label='Custom traj')
    ax.set_xlabel('Result of VO')
    ax.set_ylabel('Ground truth trajectory')
    ax.legend()
    ax.axis('equal')
    plt.show()



if __name__ == '__main__':
    #Reads parameters from the yaml file
    with open("../config/vo_params.yaml") as f:
        vo_params = yaml.load(f, Loader=yaml.FullLoader)
    
    gt_txt_file = vo_params['gt_txt_file_path']
    traj_npy_file = vo_params['poses_file_path']
    read_kitti_traj(gt_txt_file)
    #view_video()
    plot_custom_traj(traj_npy_file)
