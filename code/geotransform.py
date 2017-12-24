import numpy as np
import itertools
import random

def homograpghy(src_pts, dst_pts, threshold):

    prob = 0.5
    prob = 1 - np.power(prob, 4)
    confident_level = 0.95
    iter_cnts = int(np.ceil(np.log(1 - confident_level) / np.log(prob)))
    
    homo, mask = __homoransac(src_pts, dst_pts, iter_cnts, threshold)
    
    masked_srcpts = []
    masked_dstpts = []
    
    for i in range(len(src_pts)):
        if mask[i] > 0:
            masked_srcpts.append(src_pts[i])
            masked_dstpts.append(dst_pts[i])

    #homo = __solvehomo(masked_srcpts, masked_dstpts)    
    
    return homo, mask
    

def __homoransac(src_pts, dst_pts, iter_cnts, threshold):
    '''
    '''
    pts_cnt = len(src_pts)
    
    max_subset_size = -1
    best_homo = None
    best_mask = None
    
    for i in range(iter_cnts):
        hypo_indicies = __randindicies(0, pts_cnt, 4)
        
        ransac_srcpts = [src_pts[i] for i in hypo_indicies]
        ransac_dstpts = [dst_pts[i] for i in hypo_indicies]
        
        if __isgeoconstraint(ransac_srcpts, ransac_dstpts):
            homo = __solvehomo(ransac_srcpts, ransac_dstpts)
            subset_size, mask = __homoquality(homo, src_pts, dst_pts, threshold)
            
            #print(subset_size)
            if subset_size > max_subset_size:
                max_subset_size = subset_size
                best_mask = mask
                best_homo = homo
    
    return best_homo, best_mask
    
    
def __isgeoconstraint(src_pts, dst_pts):
    test_size = len(src_pts)
    
    indicies_list = list(itertools.combinations(np.arange(test_size), 3))
    
    neg = 0
    
    for indicies in indicies_list:
        src = [[src_pts[i][0], src_pts[i][1], 1.0] for i in indicies]
        dst = [[dst_pts[i][0], dst_pts[i][1], 1.0] for i in indicies]
        
        if (np.linalg.det(src) * np.linalg.det(dst) < 0):
            neg = neg + 1
    
    if neg != 0 and neg != len(indicies_list):
        return False
    return True

def __solvehomo(src_pts, dst_pts):
    counts = len(src_pts)
    
    eqs = np.zeros((counts * 2, 9))
    
    for i in range(0, len(eqs), 2):
        j = int(i / 2)
        x, y = src_pts[j]
        xbar, ybar = dst_pts[j]
        eqs[i][0] = x
        eqs[i][1] = y
        eqs[i][2] = 1
        eqs[i][3] = 0
        eqs[i][4] = 0
        eqs[i][5] = 0
        eqs[i][6] = -1 * xbar * x
        eqs[i][7] = -1 * xbar * y
        eqs[i][8] = -1 * xbar
        eqs[i+1][0] = 0
        eqs[i+1][1] = 0
        eqs[i+1][2] = 0
        eqs[i+1][3] = x
        eqs[i+1][4] = y
        eqs[i+1][5] = 1
        eqs[i+1][6] = -1 * ybar * x
        eqs[i+1][7] = -1 * ybar * y
        eqs[i+1][8] = -1 * ybar
    
    #print(eqs)
    u, s, v = np.linalg.svd(eqs)

    #v = v.T
    #print(s, s.shape, u.shape, eqs.shape)
    homo = v[-1,:] / v[-1,-1]
    homo = homo.reshape(3, 3)
    
    return homo

def __homoquality(homo, src_pts, dst_pts, threshold):
    counts = len(src_pts)
    
    mask = np.zeros(counts, np.uint8)
    
    for i in range(counts):
        x, y = src_pts[i]
        dstx, dsty = dst_pts[i]
        tx, ty, k = np.dot(homo, [x, y, 1])
        #tx = tx / k
        #ty = ty / k
        dist = np.linalg.norm([dstx - tx, dsty - ty], 2)
        if (dist < threshold):
            mask[i] = 1
            

    return np.sum(mask), mask
        
        
def __randindicies(start, stop, count):
    '''
    
    '''
    start = int(start)
    stop = int (stop)
    count = int(count)
    
    if stop - start < count:
        return None
    elif stop - start == count:
        return [i for i in range(start, stop)]
    
    indicies = []
    
    while count > 0:
        while True:
            index = random.randint(start, stop-1)
            if (index not in indicies):
                indicies.append(index)
                break;
        count = count - 1;
    
    return indicies;
      
    
 