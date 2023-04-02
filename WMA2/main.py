import cv2
import numpy as np
from screeninfo import get_monitors
import os


def findCircles(image):
    gimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bimg = cv2.blur(gimg, (5, 5))
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1.1, 10, param1=80, param2=40, minRadius=20, maxRadius=40)
    circles = np.uint16(np.around(circles))
    return circles


def findBig5(image):
    circles = findCircles(image)
    max = 20
    for i in circles[0, :]:
        max = i[2] if i[2] > max else max
    return max * max * np.pi


def task1(image):
    image_c = image.copy()
    circles = findCircles(image_c)
    max = 20
    for i in circles[0, :]:
        max = i[2] if i[2] > max else max
    for i in circles[0, :]:
        if i[2] > max - 4:
            cv2.circle(image_c, (i[0], i[1]), i[2], (255, 0, 0), 2)
        else:
            cv2.circle(image_c, (i[0], i[1]), i[2], (0, 255, 255), 2)
    cv2.imshow('detected circ', image_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task2(image):
    image_c = image.copy()
    circles = findCircles(image_c)
    sumSurface = 0
    countBig = 0
    countSmall = 0
    max = 20
    for i in circles[0, :]:
        max = i[2] if i[2] > max else max

    for i in circles[0, :]:
        if i[2] > max - 4:
            countBig += 1

        else:
            countSmall += 1
        sumSurface += i[2] * i[2] * np.pi
    print("Monety zajmują", sumSurface, "powierzchni, jest", countBig, "monet o nominale 5zł i", countSmall,
          "monet o nominale 5gr")



def task3(image):
    image_c = cv2.medianBlur(image, 5)
    image_gray = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 350, 620, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=40, maxLineGap=5)

    leftX = lines[0][0][0]
    rightX = lines[0][0][0]
    downY = lines[0][0][0]
    upY = lines[0][0][0]
    for line in lines:
        leftX = line[0][0] if leftX > line[0][0] else leftX
        rightX = line[0][0] if rightX < line[0][0] else rightX
        downY = line[0][1] if downY < line[0][1] else downY
        upY = line[0][1] if upY > line[0][1] else upY
    image_c = image.copy()

    cv2.line(image_c, (leftX, downY), (rightX, downY), (0, 255, 0), 2)
    cv2.line(image_c, (leftX, downY), (leftX, upY), (0, 255, 0), 2)
    cv2.line(image_c, (rightX, upY), (rightX, downY), (0, 255, 0), 2)
    cv2.line(image_c, (rightX, upY), (leftX, upY), (0, 255, 0), 2)
    cv2.imshow('detected circ', image_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    w = rightX - leftX
    l = downY - upY
    surface = w * l
    coinS = findBig5(image)
    print("Powierzchnia tacy wynosi:", surface)
    print("Powierzchnia monety wynosi:", coinS)
    print("moneta jest", surface / coinS, "razy mniejsza od tacy i o", surface - coinS, "mniejsza")
    cv2.waitKey(0)


def task4(image):
    circles = findCircles(image)
    image_c = cv2.medianBlur(image, 5)
    image_gray = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 350, 620, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=40, maxLineGap=5)

    leftX = lines[0][0][0]
    rightX = lines[0][0][0]
    downY = lines[0][0][0]
    upY = lines[0][0][0]
    for line in lines:
        leftX = line[0][0] if leftX > line[0][0] else leftX
        rightX = line[0][0] if rightX < line[0][0] else rightX
        downY = line[0][1] if downY < line[0][1] else downY
        upY = line[0][1] if upY > line[0][1] else upY
    max = 20
    sumTaca = 0.
    sumTable = 0.
    for i in circles[0, :]:
        max = i[2] if i[2] > max else max
    for i in circles[0, :]:

        if i[2] > max - 4:
            if leftX < i[0] < rightX and upY < i[1] < downY:
                sumTaca += 5
            else:
                sumTable += 5
        else:
            if leftX < i[0] < rightX and upY < i[1] < downY:
                sumTaca += 0.05
            else:
                sumTable += 0.05

    print("na tacy leży", int(sumTaca), "złoty i ", int(sumTaca % 1 * 100), "groszy.")
    print("poza tacą leży", int(sumTable), "złoty i ", int(sumTable % 1 * 100), "groszy.")



def tasks(image):
    task1(image)
    task2(image)
    task3(image)
    task4(image)


def main():
    directory = '../data/mp2/'
    for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            tasks(image)
            print("----------------------------------------------------- -------------------------------------------------------------------------------")
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
