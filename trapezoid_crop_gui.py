import os
import cv2
import sys
import numpy as np
from math import sqrt
import os.path as osp
import argparse
path = osp.dirname(osp.abspath(__file__))

from MousePts import MousePts

def get_euclidian_distance(pt1, pt2):
	return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def apply_transform(point_list,M):
	out_point_list= (cv2.perspectiveTransform(np.array(point_list),M)).astype(int)
	return out_point_list
		
def trapezoidHandCrop(image, ht=None, wd=None):
    """
    Function that asks user to click on four corner points of trapezoidal shape and 
    generates rectangular transformed image of selected trapezoid. ht and wd are height and 
    width of final crop image, by default it will be calculated based on input points.
    """
    image_copy = image.copy()
    rows, cols = image.shape[:2]
    
    if 1:
        src_points = MousePts().select_roi(image_copy)
        src_points = np.float32(src_points)
        np.savetxt('pts.txt', src_points)
    else:
        src_points = np.loadtxt('pts.txt')
    print('src_points:',src_points)
    
    pt1, pt2, pt3, pt4 = src_points
    w1 = get_euclidian_distance(pt2, pt1)
    h1 = get_euclidian_distance(pt2, pt3)
    
    w1 = wd if wd is not None else w1 #assign crop width as user's input
    h1 = ht if ht is not None else h1 #assign crop height as user's input
    x1, y1 = 0, 0
    dst_points = np.float32([[x1,y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    crop_image = cv2.warpPerspective(image_copy, M, (int(w1), int(h1)))
    
    pointsOut = convert_pts(src_points, M)
    
    print('pointsOut: ', pointsOut)
    
    return crop_image
    
def convert_pts(boxpoints, M):
    boxpoints = np.float32(boxpoints)
    warp_boxes = []
    for b in boxpoints:
        b = np.array(b).reshape(1, 1, 2)
        w_b = apply_transform(b, M)
        w_box_pt = list(w_b[0][0])
        warp_boxes.append(w_box_pt)
    return warp_boxes

def Get_warped_image(img, M=None):
    rows, cols = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (cols, rows)) 
    return warped

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--path"  , default='crops/18.png', help="path of image")
    args = parser.parse_args()
    imgPath = args.path
    
    image = cv2.imread(imgPath)
    crop_img = trapezoidHandCrop(image)

    path='crops'
    if not os.path.exists(path):
        os.makedirs(path)
        
    basename = os.path.basename(imgPath)
    #import pdb;pdb.set_trace()

    filename = os.path.splitext(basename)[0]+'.png'
    cv2.imwrite(os.path.join(path, filename),crop_img)

    cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
    cv2.imshow('Cropped', crop_img)
    cv2.waitKey(0)

    
    
