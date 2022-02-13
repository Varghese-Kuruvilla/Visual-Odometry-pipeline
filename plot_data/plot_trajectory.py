import matplotlib.pyplot as plt
import numpy as np



def plot_traj():
    '''
    Plot the trajectory given by visual Odometry
    '''
    data = np.load('./global_pose_fastr2d2.npy')
    print("data[:,:,0]",data[0:10,:,0])
    vehicle_heading = data[:,:,0]
    print("vehicle_heading:",vehicle_heading)
    vehicle_lateral = data[:,:,2]
    print("vehicle_lateral:",vehicle_lateral)
    plt.plot(vehicle_lateral,vehicle_heading,'r+')
    plt.xlabel('Vehicle heading')
    plt.xlabel('Vehicle lateral')
    plt.show()


if __name__ == '__main__':
    plot_traj()