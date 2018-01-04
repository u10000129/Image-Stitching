import cv2
import math
import copy
import numpy as np 

class matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, i1, i2, direction=None):
        imageSet1 = self.getSURFFeatures(i1)
        imageSet2 = self.getSURFFeatures(i2)
        print ("Direction : ", direction)
        matches = self.flann.knnMatch(
            imageSet2['des'],
            imageSet1['des'],
            k=2
        )
        good = []
        for i , (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (__, i) in good]
            )
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i, __) in good]
            )
            
            # img1 = copy.deepcopy(i1)
            # img2 = copy.deepcopy(i2)
            # for i in range(matchedPointsCurrent.shape[0]):
                # cv2.circle(img2, (math.floor(matchedPointsCurrent[i][0]), math.floor(matchedPointsCurrent[i][1])), 2, (0,0,255), -1)
                # cv2.circle(img1, (math.floor(matchedPointsPrev[i][0]), math.floor(matchedPointsPrev[i][1])), 2, (0,0,255), -1)
            # cv2.imshow("img2", img2)
            # cv2.imshow("img1", img1)
            # cv2.waitKey()

            H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSURFFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp':kp, 'des':des}