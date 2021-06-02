import cv2 
import numpy as np

class document:
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.height, self.width = self.img.shape[:2]
        return

    def FindMarked(self, lower, upper, tresh, render=True):
        e_img = cv2.GaussianBlur(self.img, (5,5), 0)
        mask = cv2.inRange(e_img, lower, upper)
        e_img = cv2.bitwise_and(e_img, e_img, mask = mask)

        e_img = cv2.cvtColor(e_img, cv2.COLOR_RGB2GRAY)
        e_img = cv2.GaussianBlur(e_img, (5,5), 0)
        e_img = cv2.adaptiveThreshold(e_img, 255, 1, 1, 11, 2)

        self.contours, hierarchy =  cv2.findContours(e_img ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        areas = []

        for c in self.contours:
            if cv2.contourArea(c) > 10000:
                box = cv2.boxPoints(cv2.minAreaRect(c))
                box = np.int0(box)
                areas.append([box[0][1].tolist()-tresh, box[2][1].tolist()+tresh])
                if render:
                    cv2.drawContours(self.img, [box], 0, (0, 255, 0),5)
        
        self.marked_img = np.zeros((self.height, self.width, 3), np.uint8)
        self.new_height = 0
        for a in areas:
            cut = self.img[a[0]:a[1], :self.width]
            self.marked_img[self.new_height:self.new_height+cut.shape[0], :self.width, :3] = cut
            self.new_height += cut.shape[0]
        
        self.marked_img = self.marked_img[:self.new_height, :self.width, :3]

        return self.marked_img

    def resize(self, res, img = False):
        if not type(img) == np.ndarray:
            img = self.img
            print("test")
        ratio = img.shape[0]/img.shape[1]
        res_img = cv2.resize(img, (res, int(res*ratio)))
        return res_img
    
    def save(self):
        cv2.imwrite("doc.jpg" ,self.marked_img)