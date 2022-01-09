import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy



file_path = "fruit-inspection-images/first task/"
NIR_prefix = "C0_"
color_prefix = "C1_"
file_list = {"NIR": [], "color": []}

# def flood_fill(image, seed):
#
def FillHole(image):
    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # choose a seed point, seed point must be a point in background
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # get im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    # im_floodfill_inv is the inversion of im_floodfill
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # get output image
    im_out = image | im_floodfill_inv
    return im_out, im_floodfill, im_floodfill_inv

for i in range(1, 4):
    file_list["NIR"].append(NIR_prefix + str(i).zfill(6) + ".png")
    file_list["color"].append(color_prefix + str(i).zfill(6) + ".png")

for i in range(0, 3):
    NIR_image_path = file_path + file_list["NIR"][i]
    color_image_path = file_path + file_list["color"][i]
    NIR_image = cv2.imread(NIR_image_path, 0)
    color_image = cv2.imread(color_image_path)
    NIR_ret, NIR_output = cv2.threshold(NIR_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    copy_color_output = copy.deepcopy(color_image)
    flood_filled_NIR_output = FillHole(NIR_output)
    contours, hierarchy = cv2.findContours(
        flood_filled_NIR_output[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Ksize = 3
    minVal = 20
    maxVal = 40
    L2g = True
    canny = cv2.Canny(NIR_output, 50, 100, apertureSize=Ksize, L2gradient=L2g)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)
    plt.subplot(3, 2, 1)
    plt.imshow(NIR_image, cmap='gray', vmin=0, vmax=255)
    plt.title("input NIR image")
    plt.subplot(3, 2, 2)
    plt.imshow(NIR_output, cmap='gray', vmin=0, vmax=255)
    plt.title("threshold NIR image")
    plt.subplot(3, 2, 3)
    plt.imshow(flood_filled_NIR_output[0], cmap='gray', vmin=0, vmax=255)
    plt.title("Flood filled output NIR image")
    plt.subplot(3, 2, 4)
    plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
    plt.title("Canny edge")
    plt.subplot(3, 2, 5)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title("outline fruit")
    plt.show()


