import glob
import numpy as np
import cv2 
import os
import time
from statistics import mean
import pickle

def world_2d(points_ls, H):
    '''
    Convert a list of 3d points to corresponding image points using homography
    '''
    points_2d_ls = []
    for point in points_ls:
        point_3d = np.asarray(point)
        point_3d = point_3d.reshape(3,1)
        # print("point_3d",point_3d)
        point_2d = np.dot(H,point_3d)
        point_2d = point_2d // point_2d[2,0]
        points_2d_ls.append(point_2d)
        # print("point_2d",point_2d)
    return points_2d_ls

def project_points(img,points_2d_ls):
    '''
    Draw a bounding rectangle on the image using the computed homography
    '''
    # cv2_imshow(img)
    # for point_2d in points_2d_ls:
    #     cv2.circle(img,(point_2d[0],point_2d[1]), 5, (0,0,255), -1)

    #For visualization
    N = len(points_2d_ls)
    for i in range(0,N):
        cv2.line(img,(points_2d_ls[i%N][0],points_2d_ls[i%N][1]),\
                 (points_2d_ls[(i+1)%N][0],points_2d_ls[(i+1)%N][1]),\
                 (0,0,255),4)
    #Image with points drawn on it
    cv2.imshow('ProjPoints', img)

def pot_parking_spot(orig_img,inf_img,points_2d_ls):
    '''
    Function to detect potential parking spot
    '''
    #TODO: Optimize the code
    xcoords_ls = []
    ycoords_ls = []
    for point in points_2d_ls:
        xcoords_ls.append(point[0][0])
        ycoords_ls.append(point[1][0])
    
    #Line equations of the top and bottom line
    coeff_top = np.polyfit(xcoords_ls[0:2],ycoords_ls[0:2],1)
    line_top = np.poly1d(coeff_top)

    coeff_bottom = np.polyfit(xcoords_ls[2:4],ycoords_ls[2:4],1)
    line_bottom = np.poly1d(coeff_bottom)
    # print("line_bottom",line_bottom)
    # print("line_top",line_top)
    flag_top = 0
    flag_bottom = 0
    #Points for potential parking spot
    pt_tl = []
    pt_tr = []
    pt_bl = []
    pt_br = []
    for x in range(0,inf_img.shape[1]):
        if(inf_img[int(line_top(x)),x] == 255 and flag_top == 0):
            pt_tl = [int(line_top(x)),x]
            pt_tr = [int(line_top(x+200)),int(x+200)]
            # cv2.circle(orig_img,(pt_tl[1],pt_tl[0]), 5, (0,0,255), -1)
            # cv2.circle(orig_img,(pt_tr[1],pt_tr[0]), 5, (0,0,255), -1)
            # cv2_imshow(orig_img)
            flag_top = 1

        if(inf_img[int(line_bottom(x)),x] == 255 and flag_bottom == 0):
            pt_bl = [int(line_bottom(x)),x]
            pt_br = [int(line_bottom(x+200)),int(x+200)]
            # cv2.circle(orig_img,(pt_bl[1],pt_bl[0]), 5, (0,0,255), -1)
            # cv2.circle(orig_img,(pt_br[1],pt_br[0]), 5, (0,0,255), -1)
            # cv2_imshow(orig_img)
            flag_bottom = 1
        
        if(flag_top == 1 and flag_bottom ==1):
            # cv2_imshow(orig_img)
            break

    if(flag_top == 1 and flag_bottom == 1):
        cv2.line(orig_img,(pt_tl[1],pt_tl[0]),(pt_tr[1],pt_tr[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_tr[1],pt_tr[0]),(pt_br[1],pt_br[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_br[1],pt_br[0]),(pt_bl[1],pt_bl[0]),(0,0,255),2)
        cv2.line(orig_img,(pt_bl[1],pt_bl[0]),(pt_tl[1],pt_tl[0]),(0,0,255),2)
    
    pt_ls = [pt_bl,pt_br,pt_tr,pt_tl]
    return orig_img,pt_ls
    
    
def chk_cnts(pt_ls):
    '''
    Check if contour pt_ls contains invalid entries
    '''
    for pt in pt_ls:
        if not pt:
            return False
    
    return True
def ret_line_eq(pt1,pt2):
    '''
    Returns m1,c1 given 2 points
    '''
    points = [pt1,pt2]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m1, c1 = np.linalg.lstsq(A, y_coords)[0] #TODO: Improve this, it computes the least square solution
    return m1,c1


def find_midpoint(pt_ls):
    '''
    Find the midpoint given 4 corners of the contour
    '''
    m1,c1 = ret_line_eq(pt_ls[0],pt_ls[2])
    m2,c2 = ret_line_eq(pt_ls[1],pt_ls[3])

    #Solve the 2 eqns to obtain the midpoint
    A = np.array([[-m1,1],
                 [-m2,1]],dtype=np.float64)
    B = np.array([c1,c2])
    midpoint = np.linalg.inv(A).dot(B)
    #Inverse homography matrix
    inv_h = np.linalg.inv(H)
    #Conversion from Numpy coordinates to OpenCV coordinates
    #TODO: Optimize this computation
    midpoint_homography = np.copy(midpoint).reshape(2,1)
    midpoint_homography[0,0] = midpoint[1]
    midpoint_homography[1,0] = midpoint[0]
    midpoint_homography = np.append(midpoint_homography,1).reshape(-1,1)
    # midpoint_homography.append(midpoint[0])
    # midpoint_homography.append(1)
    world_midpoint = np.dot(inv_h,midpoint_homography)
    world_midpoint = world_midpoint / world_midpoint[2,0]
    return world_midpoint , midpoint
