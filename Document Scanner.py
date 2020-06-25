import cv2
import numpy as np
import os
import stacked_images as si


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 100, 100)
    kernel = np.ones((5, 5))
    canny_dilation = cv2.dilate(canny, kernel, iterations=2)
    canny_erode = cv2.erode(canny_dilation, kernel, iterations=1)
    return canny_erode


def find_contour(img):
    points = np.array([])
    max_area = 0
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cvt in contours:
        area = cv2.contourArea(cvt)
        if area > 3000:
            perimeter = cv2.arcLength(cvt, True)
            approx = cv2.approxPolyDP(cvt, 0.02*perimeter, True)
            # find max counter with rectangle in shape
            if area > max_area and len(approx) == 4:
                points = approx
                max_area = area
    return points


def warp(points):

    x1, y1 = points[0][0][0], points[0][0][1]
    x2, y2 = points[1][0][0], points[1][0][1]
    x3, y3 = points[2][0][0], points[2][0][1]
    x4, y4 = points[3][0][0], points[3][0][1]

    height = int(np.sqrt(np.square(x1 - x2) + np.square(y2 - y1)))
    width = int(np.sqrt(np.square(x4 - x1) + np.square(y4 - y1)))
    precision = 10
    # height = doc_copy.shape[0]
    # width = doc_copy.shape[1]

    from_points = np.float32([[x1 + precision, y1 + precision], [x2 + precision, y2 - precision],
                              [x3 - precision, y3 - precision], [x4 - precision, y4 + precision]])
    to_points = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    matrix = cv2.getPerspectiveTransform(from_points, to_points)
    warp_image = cv2.warpPerspective(src=doc_copy, M=matrix, dsize=(width, height))

    return warp_image


def enhance_doc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    enhanced_img = cv2.adaptiveThreshold(src=blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                         thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
    return enhanced_img

def save_file(img):
    height, width = img.shape[0], img.shape[1]
    path, dir, files = next(os.walk('resources/saves'))
    total_files = len(files)
    cv2.imwrite('resources/saves/scanned_doc_{}.jpg'.format(total_files + 1), img)

    cv2.rectangle(img, (0, height // 2 - 80), (width, height // 2 + 40), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, 'Saved...', (width // 2 - 20, height // 2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.2,
                (255, 255, 255), 2)

    images = si.stacked_images(0.5, [img])
    cv2.imshow('doc', images)
    cv2.waitKey(2000)

while True:
    doc = cv2.imread('resources/doc.jpg')
    doc_copy = doc.copy()
    processed_img = process(doc)
    warp_points = find_contour(processed_img)
    if len(warp_points) == 4:
        warp_img = warp(warp_points)
        enhanced_image = enhance_doc(warp_img)

        if cv2.waitKey(0) == ord('s'):
            save_file(enhanced_image.copy())

        images = si.stacked_images(0.5, [enhanced_image])
        cv2.imshow('doc', images)

        if cv2.waitKey(0) == ord('q'):
            break

    else:
        print("Failed... Make sure all corners of the doc is visible in the image")
        break

    cv2.waitKey(0)
