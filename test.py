import matplotlib.patches as patches
import cv2 as cv

image = cv.imread("data/results/Figure_4.png", cv.IMREAD_GRAYSCALE)

cv.imshow("Original", image)
cv.waitKey(0)

zoom_x = 120
zoom_y = 140
zoom_size = 40


cv.destroyAllWindows()