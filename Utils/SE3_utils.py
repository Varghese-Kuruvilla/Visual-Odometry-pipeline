# Taken from  DF-VO : https://github.com/Huangying-Zhan/DF-VO

import numpy as np

class SE3():
    """SE3 object consists rotation and translation components
    Attributes:
        pose (4x4 numpy array): camera pose
        inv_pose (4x4 numpy array): inverse camera pose
        R (3x3 numpy array): Rotation component
        t (3x1 numpy array): translation component,
    """
    def __init__(self, np_arr=None):
        if np_arr is None:
            self._pose = np.eye(4)
        else:
            self._pose = np_arr

    @property
    def pose(self):
        """ pose (4x4 numpy array): camera pose """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def inv_pose(self):
        """ inv_pose (4x4 numpy array): inverse camera pose """
        return np.linalg.inv(self._pose)

    @inv_pose.setter
    def inv_pose(self, value):
        self._pose = np.linalg.inv(value)

    @property
    def R(self):
        return self._pose[:3, :3]

    @R.setter
    def R(self, value):
        self._pose[:3, :3] = value

    @property
    def t(self):
        return self._pose[:3, 3:]

    @t.setter
    def t(self, value):
        self._pose[:3, 3:] = value
