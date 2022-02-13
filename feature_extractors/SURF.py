import cv2
# import orb_matcher
# import orb
import numpy as np
import time

# sift = cv2.xfeatures2d.SIFT_create(6000)
sift = cv2.xfeatures2d.SURF_create()
bf = cv2.BFMatcher()


def extract_features_and_desc(image):

    t1 = time.time()
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(image,None)
    # print(kp)
    kp = np.asarray([[p.pt[0], p.pt[1]] for p in kp])
    # print(kp.shape)
    return kp, desc

def get_matches(ref_kp, ref_desc, cur_kp, cur_desc, img_shape, pix_rad = 100, flag = 2):
    if flag == 2:
        matchesx = bf.knnMatch(ref_desc,cur_desc, k=2)
        good = []
        for m,n in matchesx:
            if m.distance < 0.85*n.distance:
                good.append(m)
        # print(good)
        matches_mtr = [[m.queryIdx, m.trainIdx] for m in good]
        return (np.asarray(matches_mtr))

