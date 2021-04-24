import os
import cv2
import sys
import numpy as np
from math import sqrt
import os.path as osp
path = osp.dirname(osp.abspath(__file__))
try:
	from .MousePts import MousePts
	from .img_utils import apply_transform, get_euclidian_distance
except:
	from MousePts import MousePts
	from img_utils import apply_transform, get_euclidian_distance

class Caliberate():
	def __init__(self, recaliberate=True):
		self.imageWindowName   = 'image'
		self.birdeyeWindowName = 'BirdEye'
		self.cropWindowName    = 'CropImage'
		self.pointssaveFile    = osp.join(path, 'data', 'points.txt')
		self.WarpMatrixFile    = osp.join(path, 'data', 'warpmatrix.txt')
		self.ReferenceptsFile  = osp.join(path, 'data', 'refpts.txt')
		self.ROIptsFile        = osp.join(path, 'data', 'roipts.txt')
		self.InvWarpMatrixFile = osp.join(path, 'data', 'inv_warpmatrix.txt')
		self.recaliberate      = recaliberate
		self.x_offset, self.y_offset = 0, 0
		
		if not self.recaliberate:
			self.src_points = np.float32(np.loadtxt(self.pointssaveFile))
			self.M = np.float32(np.loadtxt(self.WarpMatrixFile))
			self.M_inv = np.float32(np.loadtxt(self.InvWarpMatrixFile))
			self.transformed_two_pts = np.float32(np.loadtxt(self.ReferenceptsFile))
			self.roipts = np.float32(np.loadtxt(self.ROIptsFile))
			
	def CaliberateImage(self, image, num_pts=4):
		image_copy = image.copy()
		rows, cols = image.shape[:2]
		cv2.namedWindow(self.imageWindowName, cv2.WINDOW_NORMAL)
		cv2.imshow(self.imageWindowName, image)
		cv2.waitKey(30)
		
		src_points, __ = MousePts(image).getpt(num_pts)
		src_points = np.float32(src_points)
		np.savetxt(self.pointssaveFile, src_points)
		
		temp_points = np.array(src_points).astype(int).tolist()
		cv2.polylines(img=image, pts=np.array([temp_points]), isClosed=True, color=(0,0,255), thickness=2)
		pt1, pt2, pt3, pt4 = src_points
		w1 = get_euclidian_distance(pt2, pt1)
		h1 = get_euclidian_distance(pt2, pt3)
		
		x1, y1 = pt1[0] + self.x_offset, pt1[1] + self.y_offset
		dst_points = np.float32([[x1,y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
		
		
		M = cv2.getPerspectiveTransform(src_points, dst_points)
		M_inv= cv2.getPerspectiveTransform(dst_points, src_points)
		np.savetxt(self.WarpMatrixFile, M)
		np.savetxt(self.InvWarpMatrixFile, M_inv)
		
		warped_image = cv2.warpPerspective(image_copy, M, (cols, rows))
		#Transforming points 
		pointsOut = self.convert_pts(src_points, M)
		print('pointsOut: ', pointsOut)
		cv2.polylines(img=warped_image, pts=np.array([pointsOut]), isClosed=True, color=(0,0,255), thickness=2)

		
		#Transforming a point
		two_pts, __ = MousePts(image).getpt(2)
		transformed_two_pts = self.convert_pts(two_pts, M)
		np.savetxt(self.ReferenceptsFile, transformed_two_pts)
		
		temp_points = np.array(transformed_two_pts).astype(int).tolist()
		cv2.polylines(img=warped_image, pts=np.array([temp_points]), isClosed=True, color=(0,0,255), thickness=2)
		
		p1, p2 = MousePts(warped_image).selectRect(warped_image,'image')
		roipts = [p1, p2]
		print('roipts: ', roipts)
		x1, y1 = roipts[0]
		x2, y2 = roipts[1]
		roipts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
		np.savetxt(self.ROIptsFile, roipts)
		
		startpoint = roipts[0]
		endpoint = roipts[2]
		inv_roipts = self.convert_pts(roipts, M_inv)
		
		crop_img = warped_image[int(startpoint[1]):int(endpoint[1]), int(startpoint[0]):int(endpoint[0])]

		# cv2.polylines(img=warped_image, pts=np.array([transformed_two_pts]), isClosed=True, color=(0,0,255), thickness=2)
		cv2.polylines(img=image, pts=np.array([inv_roipts]), isClosed=True, color=(0,0,255), thickness=2)
		
		cv2.namedWindow(self.birdeyeWindowName, cv2.WINDOW_NORMAL)
		cv2.imshow(self.birdeyeWindowName, warped_image)
		cv2.waitKey(0)

		cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
		cv2.imshow('Cropped', crop_img)
		cv2.waitKey(0)

		cv2.namedWindow('img',cv2.WINDOW_NORMAL)
		cv2.imshow('img',image)
		cv2.waitKey(0)
		
	def GetCaliberatedPoints(self):
		return self.M, self.M_inv, self.transformed_two_pts, self.roipts

	def convert_pts(self, boxpoints, M):
		boxpoints = np.float32(boxpoints)
		warp_boxes = []
		for b in boxpoints:
			b = np.array(b).reshape(1, 1, 2)
			w_b = apply_transform(b, M)
			w_box_pt = list(w_b[0][0])
			warp_boxes.append(w_box_pt)
		return warp_boxes

	def Get_warped_image(self, img, M=None, ratio=None):
		rows, cols = img.shape[:2]
		# warped = cv2.warpPerspective(img, self.matrix, (ratio[0]*cols,ratio[1]*rows))
		warped = cv2.warpPerspective(img, M, (cols, rows)) 
		return warped

if __name__=='__main__':
    imgPath = '0_082924_8.png'
    if 1:
        Cal_Obj = Caliberate(recaliberate=True)
        image = cv2.imread(imgPath)
        Cal_Obj.CaliberateImage(image)
	
    if 1:
        Cal_Obj = Caliberate(recaliberate=False)
        image = cv2.imread(imgPath)
        M, M_inv, transformed_two_pts, roipts = Cal_Obj.GetCaliberatedPoints()
        warped = Cal_Obj.Get_warped_image(image, M=M)

        temp_points = np.array(transformed_two_pts).astype(int).tolist()
        cv2.polylines(img=warped, pts=np.array([temp_points]), isClosed=True, color=(0,0,255), thickness=2)
        startpoint = roipts[0]
        endpoint = roipts[2]
        inv_roipts = Cal_Obj.convert_pts(roipts, M_inv)
        crop_img = warped[int(startpoint[1]):int(endpoint[1]), int(startpoint[0]):int(endpoint[0])]
        cv2.polylines(img=image, pts=np.array([inv_roipts]), isClosed=True, color=(0,0,255), thickness=2)
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img',image)
        cv2.waitKey(10)

        cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
        cv2.imshow('warped', warped)
        cv2.waitKey(10)

        cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
        cv2.imshow('Cropped', crop_img)
        cv2.waitKey(0)
        cv2.imwrite(os.path.splitext(os.path.basename(imgPath))[0] + '_crop.png',crop_img)

	
