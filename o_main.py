import cv2
import numpy as np

res = 1000
treshhold = 40

img = cv2.imread("scan.jpg")
height, width = img.shape[:2]
crop_img = img[0:height, int(width/10):int(width/7)]
crop_img = img

lower = np.array([10, 200, 20])
upper = np.array([240, 255, 150])

blur = cv2.GaussianBlur(crop_img, (5,5), 0)

mask = cv2.inRange(blur, lower, upper)
crop_img = cv2.bitwise_and(crop_img, crop_img, mask = mask)

gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

tresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

contours, hierarchy =  cv2.findContours(tresh ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

ares = []

for i in contours:
    if cv2.contourArea(i) > 10000:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)       
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0),5)
        ares.append([box[0][1].tolist()-treshhold, box[2][1].tolist()+treshhold])

impart = []
n_h = 0
for a in ares:
    cut = img[a[0]:a[1], :width]
    impart.append([cut,cut.shape[0], n_h])
    n_h += cut.shape[0]

new_img = np.zeros((n_h, width,3), np.uint8)

for part in impart:
    new_img[part[2]:part[2]+part[1], :part[0].shape[1], :3] = part[0]

ratio = new_img.shape[0]/new_img.shape[1]
new_img = cv2.resize(new_img, (res, int(res*ratio)))

cv2.imshow("tse", new_img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
elif cv2.waitKey(0) & 0xFF == ord('s'):
    cv2.imwrite("doc.jpg", new_img)