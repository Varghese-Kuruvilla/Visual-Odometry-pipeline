import numpy as np
import yaml 

#Utils
def breakpoint():
    inp = input('Waiting for input...')

def prepare_data(file_name):
    data = np.load(file_name)
    with open(file_name + '.txt',mode='w') as f:
        for pose in data:
            pose = pose.reshape(1,16)
            np.savetxt(f,pose)
    
    print("Data processed")


def prepare_kitti_gt_data(gt_file):
    f_read = open(gt_file,"r")
    data = f_read.read().split('\n')
    op_filename =  gt_file.split('.txt')[0] + '_modified.txt'
    with open(op_filename ,'w') as f_write:
        for line in data:
            line = line + " " + "0.00" + " " + "0.00" + " " + "0.00" + " " + "1.00" + "\n"
            f_write.write(line)
    
    print("Data processed")

    


        








if __name__ == '__main__':
     #Reads parameters from the yaml file
    with open("../config/vo_params.yaml") as f:
        vo_params = yaml.load(f, Loader=yaml.FullLoader)
    
    
    file_name = vo_params['poses_file_path']
    gt_file = vo_params['gt_txt_file_path']
    prepare_kitti_gt_data(gt_file)
    prepare_data(file_name)