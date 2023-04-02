import cv2
import numpy as np

def main():
    image = loadImage()
    show(image)
    videoMark('data/kostka.mp4','out/result2.avi')

def loadImage():
    return cv2.imread("data/kostka.png")

def hsvify(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def mask(image):
    # Określenie zakresu kolorów maski
    lower_color = np.array([0, 250, 100])
    upper_color = np.array([5, 255, 255])
    #Maska
    maska = cv2.inRange(hsvify(image), lower_color, upper_color)

    return maska


def maskClear(image):
    kernel = np.ones((4, 4), np.uint8)
    maska = mask(image)
    maska_bez = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel)
    maska_zam = cv2.morphologyEx(maska_bez, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.medianBlur(maska_zam, ksize=5)
    return denoised
def mark(image):
    #tworzy maskę
    maska = maskClear(image)
    #wylicza współrzędne środka
    M = cv2.moments(maska,1)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    #dodaje obrazek
    image_marker = image
    #rysuje marker
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=( 0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    return image_marker


def show(image):
    cv2.imshow('Obrazek Maska Czysta',maskClear(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Obrazek z Markerem',mark(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



###
###
###

def videoMark(vid, res):
    video = cv2.VideoCapture()
    video.open(vid)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(
        res, cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    counter = 1

    while True:
        success, frame_rgb = video.read()
        if not success:
            break
        print('klatka {} z {}'.format(counter, total_frames))


        frame_rgb=mark(frame_rgb)
        result.write(frame_rgb)
        counter = counter + 1

    video.release()
    result.release()


if __name__ == '__main__':
    main()

