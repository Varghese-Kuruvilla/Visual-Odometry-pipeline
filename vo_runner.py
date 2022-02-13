import yaml
import vo_stereo_runner
import numpy as np
#For debug
def breakpoint():
    inp = input("Waiting for input...")

def read_yaml_file():
    #Reads parameters from the yaml file
    with open("config/vo_params.yaml") as f:
        vo_params = yaml.load(f, Loader=yaml.FullLoader)
    
    vo_method = vo_params['vo_method']
    cam_intr = vo_params['camera_intrinsic_matrix']
    cam_intr = np.reshape(np.asarray(cam_intr),(3,3))
    img_path = vo_params['image_path']
    output_filename = vo_params['output_filename']
    vo_stereo_runner.vo_offline_data(cam_intr,img_path,output_filename)
    # if(vo_method == "rgbd"):
        # cam_intr
        














if __name__ == '__main__':
    read_yaml_file()
