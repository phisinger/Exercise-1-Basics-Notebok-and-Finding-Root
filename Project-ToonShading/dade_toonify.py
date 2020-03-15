import cv2
import numpy as np

# Maybe a console inut would be useful to select the picture 
img = cv2.imread("teapot_gray.jpg")

# 1) Edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 13)
# Edge detection with canny
edges = cv2.Canny(gray, 50, 80)

# make the edge lines thicker
kernel = np.ones((2,2),np.uint8)
dilation = edges.copy()
dilation = cv2.dilate(dilation, kernel, 1)
# an invertation is necessairy to have black edges and not white ones
dilation = (255-dilation)

# 2) Color stuff

# smooth the colors within one color area
color = img.copy()
# It looks better without the bilateralFilter
#color = cv2.bilateralFilter(color, 7, 200, 200)
for _ in range(2):
    color = cv2.medianBlur(color, 7)

# quantise colors:
Z = color.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 13
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
color_res = res.reshape((color.shape))

# 3) Combining the color images and the edges
cartoon = cv2.bitwise_and(color_res, color_res, mask=dilation)



cv2.imshow("Image", img)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()