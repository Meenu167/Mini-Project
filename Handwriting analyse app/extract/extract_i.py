import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import glob
import cv2
import math



def find_centroid(contour):
    x_sum = 0
    y_sum = 0
    num_points = len(contour)

    for point in contour:
        x_sum += point[0][0]
        y_sum += point[0][1]

    cx_alt = int(x_sum / num_points)
    cy_alt = int(y_sum / num_points)
    
    return [cx_alt,cy_alt]

# Slant of the handwritten image
# Input: BGR Image
# Output: Slant angle of the image

#
# def slant_angle(img):
#     img = cv2.bilateralFilter(img, 5, 700, 700)
#
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     centroids = []
#     # Apply image preprocessing techniques
#     # Example: Adaptive thresholding
#     _, thresholded = cv2.threshold(
#         gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(
#         thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         # Calculate the moments of the contour
#         a = find_centroid(contour)
#         centroids.append(a)
#
#     stem = centroids[0]
#
#     stem_contour = contours[0]
#     # min_y = centroids[0][1]
#     for i in range(len(centroids)):
#         if centroids[i][1] > stem[1]:
#             stem = centroids[i]
#             stem_contour = contours[i]
#
#     # print(stem_contour)
#     if True:
#         # Fit a line through the contour using linear regression
#         [vx, vy, x, y] = cv2.fitLine(stem_contour, cv2.DIST_L2, 0, 0.01, 0.01)
#
#     # Calculate the inclination angle of the line
#         angle = np.arctan2(vy, vx) * 180 / np.pi
#
#         if angle < 0:
#             angle += 180
#
#         return 180 - angle[0]

import cv2
import numpy as np


def slant_angle(img):
    # Check if the image is empty or None
    if img is None or img.size == 0:
        raise ValueError("Input image is empty. Cannot process slant angle.")

    # Apply bilateral filter
    img = cv2.bilateralFilter(img, 5, 700, 700)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centroids = []

    # Apply adaptive thresholding with error handling
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in the image. Cannot calculate slant angle.")

    # Find centroids of each contour
    for contour in contours:
        a = find_centroid(contour)  # Ensure find_centroid function is defined
        centroids.append(a)

    # Initialize stem as the first centroid
    stem = centroids[0]
    stem_contour = contours[0]

    # Find the lowest point by y-coordinate
    for i in range(len(centroids)):
        if centroids[i][1] > stem[1]:
            stem = centroids[i]
            stem_contour = contours[i]

    # Fit a line to the contour with error handling
    if stem_contour.size > 0:
        [vx, vy, x, y] = cv2.fitLine(stem_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy, vx) * 180 / np.pi  # Calculate inclination angle

        # Normalize the angle to be within [0, 180] degrees
        if angle < 0:
            angle += 180
        return 180 - angle[0]
    else:
        raise ValueError("Stem contour is empty. Cannot calculate slant angle.")


# Position of the dot in the handwritten image with respect to stem
# Input: BGR Image
# Output: Position ['left','right','balanced']
def pos_of_dot(image):
    b_f_img = cv2.bilateralFilter(image, 5, 700, 700)
    angle = slant_angle(image)
    gray = cv2.cvtColor(b_f_img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        # Calculate the moments of the 
        a = find_centroid(contour)
        centroids.append(a)

        dot = centroids[0]
        stem = centroids[0]

        dot_contour = contours[0]
        stem_contour = contours[0]
    # min_y = centroids[0][1]
        for i in range(len(centroids)):
            if centroids[i][1] < dot[1]:
                dot = centroids[i]
                dot_contour = contours[i]
        for i in range(len(centroids)):
            if centroids[i][1] > stem[1]:
                stem = centroids[i]
                stem_contour = contours[i]
    dot[1] = thresholded.shape[1]-dot[1]
    stem[1] = thresholded.shape[1]-stem[1]

    slope = (dot[1] - stem[1]) / (dot[0] - stem[0])
    angle_rad = math.atan(slope)
    angle_dot = math.degrees(angle_rad)
    if angle_dot < 0:
        angle_dot += 180

    if angle_dot < angle-10:
        return "Right"
    elif angle_dot >= angle-10 and angle_dot <= angle+10:
        return "Balanced"
    else:
        return "Left"



# Height of the dot in the handwritten image with respect to stem
# Input: BGR Image
# Output: Height of the dot (Euclidean distance between dot and top of stem)
def height_of_dot(image):
    image = cv2.resize(image, (228, 228))
    b_f_img = cv2.bilateralFilter(image, 5, 700, 700)
    angle = slant_angle(image)
    gray = cv2.cvtColor(b_f_img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        # Calculate the moments of the contour
    # Calculate the centroid coordinates
        a=find_centroid(contour)
        centroids.append(a)

        dot = centroids[0]
        stem = centroids[0]

        dot_contour = contours[0]
        stem_contour = contours[0]
    # min_y = centroids[0][1]
        for i in range(len(centroids)):
            if centroids[i][1] < dot[1]:
                dot_contour = contours[i]
        for i in range(len(centroids)):
            if centroids[i][1] > stem[1]:
                stem_contour = contours[i]
    
    top_point = tuple(stem_contour[stem_contour[:, :, 1].argmin()][0])
    centroid = find_centroid(dot_contour)
    return math.sqrt((centroid[0] - top_point[0]) ** 2 + (centroid[1] - top_point[1]) ** 2)