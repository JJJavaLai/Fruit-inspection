import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from skimage.measure import regionprops_table
import pandas as pd
import copy
file_path = "fruit-inspection-images/first task/"
NIR_prefix = "C0_"
color_prefix = "C1_"
file_list = {"NIR": [], "color": []}

for i in range(1, 4):
    file_list["NIR"].append(NIR_prefix + str(i).zfill(6) + ".png")
    file_list["color"].append(color_prefix + str(i).zfill(6) + ".png")


properties = ['area', 'perimeter', 'bbox', 'bbox_area', 'label'
              #  'convex_area', 'major_axis_length', 'minor_axis_length', 'eccentricity',
              ]

def compute_centroid(img,blob_area):
    """
        given an image with a blob and the area of the blob it computes the centroid of the blob.
    """
    x_accumulator = y_accumulator=0
    for index_i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[index_i,j]>= 60:
                x_accumulator += index_i
                y_accumulator += j

    return int(np.rint(y_accumulator/blob_area)),int(np.rint(x_accumulator/blob_area))


for i in range(0, 3):
    NIR_image_path = file_path + file_list["NIR"][i]
    color_image_path = file_path + file_list["color"][i]
    NIR_image = cv2.imread(NIR_image_path, 0)
    color_image = cv2.imread(color_image_path)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    copy_color_image = copy.deepcopy(color_image)
    labels = measure.label(gray_image)

    df = pd.DataFrame(regionprops_table(labels, properties=properties))
    df = df[df["area"] >= 100]
    df = df[df["area"] <= 500]
    df = df.sort_values(["area"], ascending=[False])

    best = df.iloc[0]['label']
    area = df.iloc[0]["area"]

    labels[labels == best] = 255
    labels[labels != best] = 0
    plt.subplot(1, 1, 1)
    plt.imshow(labels, cmap='gray', vmin=0, vmax=255)
    plt.title("labels")
    plt.show()
    for area in df['area']:
        (x_tab, y_tab) = compute_centroid(labels, area)
        copy_color_image = cv2.circle(copy_color_image, (x_tab, y_tab), 50, (0, 255, 0), 5)
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(copy_color_image, cv2.COLOR_BGR2RGB))
        plt.title("blobs")
        plt.show()
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    Ksize = 3
    minVal = 20
    maxVal = 40
    L2g = True
    canny = cv2.Canny(gray_image, 50, 100, apertureSize=Ksize, L2gradient=L2g)
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title("input color_image")
    plt.subplot(3, 2, 2)
    plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
    plt.title("gray image")
    plt.subplot(3, 2, 3)
    plt.imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    plt.title("blurred image")
    plt.subplot(3, 2, 4)
    plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
    plt.title("Canny edge")
    plt.subplot(3, 2, 5)
    plt.imshow(cv2.cvtColor(copy_color_image, cv2.COLOR_BGR2RGB))
    plt.title("blobs")
    plt.show()
