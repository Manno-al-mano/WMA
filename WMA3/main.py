import cv2
import numpy as np

def resize(img, s):
    image = img.copy()
    h = int(image.shape[0] / s)
    w = int(image.shape[1] / s)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image
def findBestImage(fun,image):
    image1 = cv2.imread('../data/mp3/p1.jpg')
    image2 = cv2.imread('../data/mp3/p2.jpg')
    image3= cv2.imread('../data/mp3/p3.jpg')
    match1=fun(image1,image)
    match2=fun(image2,image)
    match3=fun(image3,image)
    if len(match1)>=len(match2) and len(match1)>=len(match3) :
        return image1
    elif len(match2)>=len(match1) and len(match2)>=len(match3) :
        return image2
    else:
        return image3


def siftMatches(image, image2):

    gimg1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gimg1 = cv2.medianBlur(gimg1, ksize=5)
    gimg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.medianBlur(gimg2, ksize=5)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    siftobject = cv2.SIFT_create()
    keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)



    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
  #  return bestMatches
    return matches
def orbMatches(image, image2):

    gimg1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    orbtobject = cv2.ORB_create()
    keypoints_1, descriptors_1 = orbtobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = orbtobject.detectAndCompute(gimg2, None)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    #  return bestMatches
    return matches
def sift(image2):
    image =findBestImage(siftMatches ,image2)
    # image = cv2.imread('../data/mp3/p1.jpg')
    image = resize(image, 4)

    gimg1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gimg1 = cv2.medianBlur(gimg1, ksize=1)
    gimg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.medianBlur(gimg2, ksize=1)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    siftobject = cv2.SIFT_create(nfeatures=10000, nOctaveLayers=8)
    keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)


    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = gimg1.shape[:2]

    obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(obj_corners, H)
    framed = cv2.polylines(image2, [np.int32(dst_corners)], True, (0, 255, 0), 5)
    return framed

def orb(image2):
    image = findBestImage(orbMatches,image2)
    #image = cv2.imread('../data/mp3/p3.jpg')
    image = resize(image, 4)

    gimg1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gimg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    orbObject = cv2.ORB_create(nfeatures=5000,nlevels=16)
    keypoints_1, descriptors_1 = orbObject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = orbObject.detectAndCompute(gimg2, None)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = gimg1.shape[:2]

    obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(obj_corners, H)
    framed = cv2.polylines(image2, [np.int32(dst_corners)], True, (0, 255, 0), 5)
    return framed
def film(fun, video,out):
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter(
        out , cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 1
    while True:
        success, frame = video.read()
        if not success:
            break
        print('klatka {} z {}'.format(counter, total_frames))

        frame = fun(frame)
        result.write(frame)
        counter = counter + 1
    video.release()
    result.release()


def main():

    video = cv2.VideoCapture('../data/mp3/wideo.mp4')
    film(sift,video,'../out/SIFT.avi')
    #film(orb,video,'../out/ORB.avi')




if __name__ == '__main__':
    main()
