import cv2
import numpy as np
from train_license_province import *
from train_license_digits import *
from train_license_letters import *

def lpr(filename):
  img = cv2.imread(filename)
  # cv2.imshow("image", img)
  # cv2.waitKey(0)
  gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
  # cv2.imshow("GaussianBlur_img", GaussianBlur_img)
  Sobel_img = cv2.Sobel(GaussianBlur_img, -1, 1, 0, ksize=3)
  # cv2.imshow("Sobel_img", Sobel_img)
  ret, binary_img = cv2.threshold(Sobel_img, 127, 255, cv2.THRESH_BINARY)
  # cv2.imshow("binary_img", binary_img)
  kernel = np.ones((1, 15), np.uint8)

  close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
  open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
  # cv2.imshow("close_img", close_img)
  # cv2.imshow("open_img", open_img)
  # kernel2 = np.ones((10, 10), np.uint8)
  # open_img2 = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel2)

  element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  dilation_img = cv2.dilate(open_img, element, iterations=3)
  _,contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
  # cv2.imshow("lpr", img)
  # cv2.waitKey(0)
  rectangles = []
  for c in contours:
    x = []
    y = []
    for point in c:
      y.append(point[0][0])
      x.append(point[0][1])
    r = [min(y), min(x), max(y), max(x)]
    rectangles.append(r)
  dist_r = []
  max_mean = 0
  for r in rectangles:
    block = img[r[1]:r[3], r[0]:r[2]]
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    low = np.array([100, 60, 60])
    up = np.array([140, 255, 255])
    result = cv2.inRange(hsv, low, up)

    mean = cv2.mean(result)
    if mean[0] > max_mean:
      max_mean = mean[0]
      dist_r = r

  img1 = np.zeros([abs(dist_r[1]-dist_r[3]),abs(dist_r[0]+3- dist_r[2]+3),3], img.dtype)
  # cv2.rectangle(img, (dist_r[0]+3, dist_r[1]), (dist_r[2]-3, dist_r[3]), (0, 255, 0), 2)
  x = -1
  for i in range(dist_r[0]+3, dist_r[2]-3):
      x = x + 1
      y = - 1
      for j in range(dist_r[1], dist_r[3]):
          y = y + 1
          img1[y][x] = img[j][i]
  # cv2.imshow("lpr", img1)
  # cv2.waitKey(0)
  return img1

def seg(img,c_img):
    black = 0
    seg_l = []
    seg_line = []
    for i in range(img.shape[1] - 3):
        black = 0
        for j in range(img.shape[0]):
            if img[j][i] == 255:
                black = black + 1
        if black < img.shape[0] * 0.15:
            seg_l.append(i)
    seg_line.append(seg_l[0])
    y = 0
    seg_line1 =[]
    for i in range(1, len(seg_l)):
        if seg_l[i] - seg_l[i-1] >= 3:
            seg_line.append(seg_l[i])
            seg_line1.append(seg_l[i])
        if seg_l[i] - seg_l[i-1] == 1:
            y = y + 1
        if y >= 5:
            if seg_line1!=[]:
                seg_line1[len(seg_line1) - 1] = seg_l[i]
    seg_line = seg_line + seg_line1
    for i in range(len(seg_line) - 1):
        for j in range(i, len(seg_line) - 1):
            if seg_line[j] < seg_line[i]:
                mid = seg_line[i]
                seg_line[i] = seg_line[j]
                seg_line[j] = mid

    segment = []
    for i in range(1,len(seg_line) - 2 , 2):
        segment.append(int((seg_line[i + 1] + seg_line[i])//2))

    REC = [[0,5,img.shape[0], segment[0]]]
    for i in range(len(segment)):
        for j in range(img.shape[0]):
            img[j][segment[i]]= 255
    for i in range(1, len(segment)):
        REC.append([0, segment[i-1],img.shape[0], segment[i]])
    REC.append([0, segment[len(segment)-1],img.shape[0], img.shape[1] -5])

    img_seg = []
    for ij in REC:
        img1 = np.zeros([abs(ij[0] - ij[2]), abs(ij[1] - ij[3])], img.dtype)
        cv2.rectangle(c_img, (ij[1], ij[0]), (ij[3], ij[2]), (0, 255, 0), 2)
        x = -1
        for i in range(ij[1], ij[3]):
            x = x + 1
            y = - 1
            for j in range(ij[0], ij[2]):
                y = y + 1
                img1[y][x] = img[j][i]
        img_seg.append(img1)
    # cv2.imshow('1', c_img)
    x = 0
    img_seg_name = []
    for i in img_seg:
        x = x + 1
        i = cv2.resize(i, (32, 40))
        cv2.imwrite( '%d'%x + '.bmp', i)

        img_seg_name.append('%d'%x + '.bmp')
    print('seg = %d'%x)
    # cv2.waitKey(0)
    return img_seg_name

if __name__ == '__main__':
    # cv2.imread("lpr", img)
    img = lpr("1.png")
    # cv2.imshow("lpr", img)
    # cv2.waitKey(0)
    cv2.imwrite('lpr.png', img)
    img1 = cv2.imread('lpr.png',0)
    th, img1 = cv2.threshold(img1,110,255,cv2.THRESH_BINARY)
    # cv2.imshow("lpr1", img1)
    seg_image = seg(img1,img)
    x1 = tf.Graph()
    province = province_predict(seg_image[0],x1)
    print("province:", province)
    x2 = tf.Graph()
    letters_predict([seg_image[1]],x2)
    x3 = tf.Graph()
    digits = digits_predict(seg_image[2:],x3)
    # print("province:", province)
    # cv2.waitKey(0)
    # print(col_histogram)
    # fig = plt.figure()
    # plt.hist( col_histogram )
    # plt.show()
