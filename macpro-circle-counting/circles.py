import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

img = cv2.imread('mp.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input')
plt.xticks([])
plt.yticks([])

circle1 = cv2.HoughCircles(
    gray, 
    cv2.HOUGH_GRADIENT, 
    1, 
    5, 
    param1=100, 
    param2=30, 
    minRadius=20, 
    maxRadius=25
)

circles = np.uint16(np.around(circle1[0, :, :]))  # 提取为二维并四舍五入
circle_count = 0
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 4)  # 画圆心
    circle_count += 1

print('Result: ', circle_count)

plt.subplot(122), plt.imshow(img)
plt.title('Output'),
plt.xticks([]),
plt.yticks([])
plt.show()