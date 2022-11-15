import cv2
import numpy as np

class Helpers:
	

	def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
		dim = None
		(h, w) = image.shape[:2]
		if width is None and height is None:
			return image
		if width is None:
			r = height / float(h)
			dim = (int(w * r), height)
		else:
			r = width / float(w)
			dim = (width, int(h * r))
		resized = cv2.resize(image, dim, interpolation=inter)
		
		return resized
	
	def checkContours(cnts)
		if len(cnts) == 2:
			cnts = cnts[0]
		elif len(cnts) == 3:
			cnts = cnts[1]
		else:
			raise Exception_Length('The length of the contour must be 2 or 3.')

		return cnts


	def orders(pts):
		rect = np.zeros((4, 2), dtype = "float32")
		s = pts.sum(axis = 1)

		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]

		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]
		
		return rect
	
	def transformation(image)

		rect = Helpers.orders(pts)
		(tl, tr, br, bl) = rect

		width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		max_width = max(int(widthA), int(widthB))

		height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		height_b= np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		max_height = max(int(heightA), int(heightB))

		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

		return warped

