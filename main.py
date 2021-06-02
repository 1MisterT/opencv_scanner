from scanner import document
import numpy as np
import cv2

res = 1000
treshhold = 40

lower = np.array([10, 200, 20])
upper = np.array([240, 255, 150])

doc = document("scan.jpg")

img = doc.FindMarked(lower, upper, 40)

img = doc.resize(1000, img)

cv2.imshow("Scanned Image", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
elif cv2.waitKey(0) & 0xFF == ord('s'):
    doc.save()