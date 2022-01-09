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

# find_defects:
# 1.remove black background:
#   every pixel which is out of the fruit contour should be considered as a background pixel
#   these pixel will be replaced by green
#   the image will be 4 part as:
#   -------------------------------
#   |              |               |
#   |              |               |
#   |     part 1   |     part 2    |
#   |              |               |
#   |              |               |
#   |              |               |
#   -------------------------------
#   |              |               |
#   |              |               |
#   |              |               |
#   |   part 3     |     part 4    |
#   |              |               |
#   |              |               |
#   -------------------------------
#  in each part, pixel out of contour will handled
# 2.find the most dark part
# 3.return a new image
# def find_defects(image_with_contour, edge_image, NIR_image, threshold):
#     h, w = edge_image.shape
#     left_part_x = (0, w / 2)
#     right_part_x = (w / 2 + 1, w)
#     bottom_part_y = (0, h / 2)
#     top_part_y = (h / 2 + 1, h)
#     # in each part of the image, there suppose to be only one pixel is green and this pixel is on contour
#     for x in range(0, w):
#         for y in range(0, h):
#             # handle first part of image
#             if x in range(0, w / 2) and y in range(h / 2 + 1, h):


    # this part, if a pixel is green , each pixel on its left is background

for i in range(1, 4):
    file_list["NIR"].append(NIR_prefix + str(i).zfill(6) + ".png")
    file_list["color"].append(color_prefix + str(i).zfill(6) + ".png")

for i in range(0, 3):
    NIR_image_path = file_path + file_list["NIR"][i]
    color_image_path = file_path + file_list["color"][i]
    NIR_image = cv2.imread(NIR_image_path, 0)
    color_image = cv2.imread(color_image_path)
    original_image = copy.deepcopy(color_image)
    NIR_ret, NIR_output = cv2.threshold(NIR_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    copy_NIR_output = NIR_output.copy()
    flood_filled_NIR_output = FillHole(NIR_output)
    # contour for the whole fruit

    contours, hierarchy = cv2.findContours(
        NIR_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)
    # for i in range(0, len(edges)):
    #     cv2.drawContours(color_image, edges, i, (0, 255, 0), 1)
    # for cont in range(0, len(edges) - 1):
    #     (x, y), radius = cv2.minEnclosingCircle(edges[cont])
    #     cv2.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    Ksize = 3
    minVal = 20
    maxVal = 40
    L2g = True
    gaussian_NIR_image = cv2.GaussianBlur(NIR_image, (0, 0), sigmaX=1.5)
    canny = cv2.Canny(gaussian_NIR_image, 50, 100, apertureSize=Ksize, L2gradient=L2g)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(NIR_image)
    cv2.circle(original_image, minLoc, 20, (0, 255, 0), 2)
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
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title("outline fruit")
    plt.subplot(3, 2, 5)
    plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
    plt.title("Canny edge")
    plt.subplot(3, 2, 6)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("defects")
    plt.show()


