import numpy as np
import cv2
import sys
from matchers import matchers
import time
import math
import seamCarving as sc
import imutils
import utils
import poissonblending as p_b
import geotransform as gt

class Stitch:
    def __init__(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
        # print (filenames)
        self.images = [cv2.imread(each) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [],None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def prepare_lists(self):
        # print ("Number of images : %d"%self.count)
        self.centerIdx = self.count//2 
        # print ("Center index image : %d"%self.centerIdx)
        self.center_im = self.images[self.centerIdx]
        for i in range(self.count):
            if(i<=self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        # print ("Image lists prepared")

    def leftshift(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            xh = np.linalg.inv(H)
            rt = np.dot(xh, np.array([a.shape[1], 0, 1])) # right top
            rt = rt/rt[-1]
            lb = np.dot(xh, np.array([0, a.shape[0], 1])) # left bottom
            lb = lb/lb[-1]
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1])) # right bottom
            ds = ds/ds[-1]
            f1 = np.dot(xh, np.array([0,0,1])) # left top
            f1 = f1/f1[-1]
            offsetx = min(lb[0], f1[0], 0)
            offsety = min(rt[1], f1[1], 0)
            if(offsetx < 0):
                xh[0][-1] += abs(offsetx)
                offsetx = math.ceil(abs(offsetx))
            if(offsety < 0):
                xh[1][-1] += abs(offsety)
                offsety = math.ceil(abs(offsety))
            d_y = max(b.shape[0], math.ceil(lb[1]), math.ceil(ds[1]))
            dsize = (offsetx + b.shape[1], offsety + d_y)
            # tmp = cv2.warpPerspective(a, xh, dsize, flags = cv2.INTER_CUBIC)
            tmp = gt.perspectivetrans(a, xh, dsize)
            
            # Poisson-blending
            mask = cv2.warpPerspective(np.ones(a.shape, dtype=np.uint8)*255, xh, dsize, flags = cv2.INTER_CUBIC)
            mask[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = np.zeros(b.shape, dtype=np.uint8)
            blend_img = np.zeros(tmp.shape, dtype=np.uint8)
            blend_img[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
            tmp = p_b.blend(blend_img, tmp, mask)
            
            # directly stitch
            # tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            rt = np.dot(H, np.array([each.shape[1], 0, 1])) # right top
            rt = rt/rt[-1]
            lb = np.dot(H, np.array([0, each.shape[0], 1])) # left bottom
            lb = lb/lb[-1]
            f1 = np.dot(H, np.array([0,0,1])) # left top
            f1 = f1/f1[-1]
            ds = np.dot(H, np.array([each.shape[1], each.shape[0], 1])) # right bottom
            ds = ds/ds[-1]
            dsize = (int(max(rt[0], ds[0])), int(max(lb[1], ds[1], self.leftImage.shape[0])))
            # tmp = cv2.warpPerspective(each, H, dsize, flags = cv2.INTER_CUBIC)
            tmp = gt.perspectivetrans(each, H, dsize)
            
            # Poisson-blending
            cv2.imwrite('left.jpg', self.leftImage)
            c = self.leftImage.shape[1]-1
            for i in range(0, self.leftImage.shape[0]-1):
                if not np.array_equal(self.leftImage[i, c], np.array([0, 0, 0])):
                    if np.array_equal(self.leftImage[i-1, c], np.array([0, 0, 0])):
                        r1 = i
                if not np.array_equal(self.leftImage[i, c], np.array([0, 0, 0])):
                    if np.array_equal(self.leftImage[i+1, c], np.array([0, 0, 0])):
                        r2 = i+1
            mask = cv2.warpPerspective(np.ones(each.shape, dtype=np.uint8)*255, H, dsize, flags = cv2.INTER_CUBIC)
            mask[r1:r2, :self.leftImage.shape[1]] = np.zeros((r2-r1, self.leftImage.shape[1], 3), dtype=np.uint8)
            blend_img = np.zeros(tmp.shape, dtype=np.uint8)
            blend_img[:self.leftImage.shape[0], :self.leftImage.shape[1]] = self.leftImage
            tmp = p_b.blend(blend_img, tmp, mask)
            
            # directly stitch
            # tmp = self.mix_and_match(self.leftImage, tmp)
            self.leftImage = tmp

    def mix_and_match(self, leftImage, warpedImage):
        t = time.time()
        # gray = cv2.cvtColor(leftImage, cv2.COLOR_RGB2GRAY)
        i1y, i1x = leftImage.shape[:2]

        for i in range(0, i1x):
            if((time.time() - t)>60):
                print ("time limit exceeded")
                break
            for j in range(0, i1y):
                if((time.time() - t)>60):
                    print ("time limit exceeded")
                    break
                if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                    warpedImage[j,i] = [0, 0, 0]
                else:
                    if(np.array_equal(warpedImage[j,i],[0,0,0])):
                        warpedImage[j,i] = leftImage[j,i]
                    else:
                        if not np.array_equal(leftImage[j,i], [0,0,0]):
                            warpedImage[j, i] = leftImage[j,i]
        return warpedImage

    def trim_left(self):
        pass

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()


if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files2.txt"
    finally:
        print ("Parameters : ", args)
    s = Stitch(args)
    s.leftshift()
    # s.showImage('left')
    s.rightshift()
    print ("done")
    result = s.leftImage
    # result = cv2.imread("stitching-result.jpg")
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = imutils.resize(result, width=400)
    result = sc.preprocess(result)
    result = sc.fill(result, iter=500)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test-t.jpg", result)
    print ("image written")
    cv2.destroyAllWindows()