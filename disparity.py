UBIT = 'emmanueljohnson'
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(sum([ord(c) for c in UBIT]))

#References
#https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html

K = 2
RATIO = 0.75
N_MATCHES = 10
MATCH_COLOR = (0,0,255)
R_COLORS = [(213, 168, 3), (44, 195, 34), (250, 238, 196), (122, 48, 239), (146,17, 94), 
            (106, 250, 1), (82, 103, 1), (3, 97, 239), (15, 88, 241), (191, 29, 97)]
RTHRESH = 4.0

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name,image):
    cv2.imwrite(name,image) 

#Extract the keypoints using SIFT
def get_key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp,ft = sift.detectAndCompute(img, None)
    return kp,ft

#Draw the keypoints and save the result as image
def draw_key_points(img,keypoints,name):
    result = cv2.drawKeypoints(img,keypoints,None)
    save_image(name,result)

#Find the good matches among all the available matches
def get_matches(allMatches):
    matches = []
    ms = []
    for m1,m2 in allMatches:
        change = m2.distance * RATIO
        if m1.distance < change:
            matches.append((m1.trainIdx, m1.queryIdx))
            ms.append(m1)
    return matches, ms

#Get random values for a specified range
def get_random_index(r):
    rPicks = np.random.randint(0, r, N_MATCHES)
    return rPicks

#Get only one random value from the specified range
def get_one_random_index(r):
    rPicks = np.random.randint(0, r, 1)
    return rPicks

#Draw epiline for the given image using the given points
def draw_epilines(img1,img2,epilines,pts1,pts2,colors):
    w = img1.shape[1]
    for i, (r, pt1, pt2) in enumerate(zip(epilines,pts1,pts2)):
        x,y = [0, int(-r[2]/r[1])]
        x1,y1 = [w, int(-(r[2]+r[0]*w)/r[1])]
        img1 = cv2.line(img1, (x, y), (x1, y1), colors[i], 1)
        img1 = cv2.circle(img1,(int(pt1[0]),int(pt1[1])),4,colors[i],-1)
    return img1

#Get n random masks
def get_random_matches(masks, limit, max):
    result = np.zeros(max).tolist()
    c = 0
    while True:
        index = get_one_random_index(max)
        if masks[index[0]] == 1:
            result[index[0]] = 1
            c+=1
        if c == limit:
            break
    return result

def main():
    left = cv2.imread('tsucuba_left.png')
    gleft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    right = cv2.imread('tsucuba_right.png')
    gright = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    #Get keypoints of image1 and image2
    kpLeft, ftLeft = get_key_points(left)
    kpRight, ftRight = get_key_points(right)

    #Draw the keypoints of image1 and image2
    print('performing task 2.1')
    draw_key_points(gleft, kpLeft, 'task2_sift1.jpg')
    draw_key_points(gright, kpRight, 'task2_sift2.jpg')

    #Use BFMatcher to detect all the matches
    matcher = cv2.BFMatcher()
    allMatches = matcher.knnMatch(ftLeft, ftRight, K)

    #Get the matches whose distance is less than
    #the specified ratio
    matches, ms = get_matches(allMatches)

    #using the good matches draw the matches and save the image
    print('performing task 2.2')
    matchimage = cv2.drawMatches(left, kpLeft, right, kpRight, ms, None, matchColor=MATCH_COLOR, flags=2)
    cv2.imwrite('task2_matches_knn.jpg', matchimage)

    keys1, keys2, pts1, pts2 = list(), list(), list(), list()

    for k in kpLeft:
        keys1.append(k.pt)

    for k in kpRight:
        keys2.append(k.pt)

    for m in matches:
        pts1.append(keys1[m[1]])
        pts2.append(keys2[m[0]])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    #Calculate the fundamental matrix using the
    #keypoints and using the ransac algorithm
    #Make sure atleast 8 matches are present because
    #fundamental matrix is a 8 point algorithm
    print('performing task 2.3')
    if len(matches) > 8:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, RTHRESH)

    print('\n')
    print('Fundamental Matrix:')
    print(F)
    print('\n')
    
    tflist = mask.ravel().tolist()    
    npts1,npts2 = list(),list()

    rmatches = get_random_matches(tflist, N_MATCHES, len(tflist))
    rmatches = np.array(rmatches)

    for p1, p2, tf in zip(pts1, pts2, rmatches):
        if tf == 1:
            npts1.append(p1)
            npts2.append(p2)

    rpicksPts1 = np.array(npts1)
    rpicksPts2 = np.array(npts2)
    
    #Draw the epilines for both the images
    leftEpilines = cv2.computeCorrespondEpilines(rpicksPts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    leftImage = draw_epilines(left, right, leftEpilines, rpicksPts1, rpicksPts2, R_COLORS)

    rightEpilines = cv2.computeCorrespondEpilines(rpicksPts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    rightImage = draw_epilines(right, left, rightEpilines, rpicksPts2, rpicksPts1, R_COLORS)

    print('performing task 2.4')
    save_image('task2_epi_left.jpg', leftImage)
    save_image('task2_epi_right.jpg', rightImage)

    print('performing task 2.5')

    #StereoBM settings
    dset = {
        'ndisp': 112,
        'bs': 17,
        'mindisp': 16,
        'maxdiff': 0,
        'uniqratio':10,
        'srange':32,
        'swindowsize':200
    }
    
    #Compute disparity map using the StereoBM function
    stereo = cv2.StereoBM_create(numDisparities=dset['ndisp'], blockSize=dset['bs'])
    stereo.setMinDisparity(dset['mindisp'])
    stereo.setDisp12MaxDiff(dset['maxdiff'])
    stereo.setUniquenessRatio(dset['uniqratio'])
    stereo.setSpeckleRange(dset['srange'])
    stereo.setSpeckleWindowSize(dset['swindowsize'])
    disparity = np.array(stereo.compute(gleft, gright), dtype='float32')
    plt.imsave('task2_disparity.jpg', ((disparity/float(dset['mindisp'])) - dset['mindisp'])/dset['ndisp'], cmap=plt.cm.gray)

if __name__ == '__main__':
    main()
