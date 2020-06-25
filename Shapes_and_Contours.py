import cv2
import numpy as np
import stacked_images as si


def get_contour(img, canny):


    contours, hierarchy = cv2.findContours(image=canny_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        area = cv2.contourArea(cnt)
        print('Area: {}'.format(area))

        if area > 500: 

            cv2.drawContours(image=img, contours=cnt, contourIdx=-1, color=(0, 255, 0), thickness=3)

            perimeter = cv2.arcLength(curve=cnt, closed=True)

            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02*perimeter, closed=True) 
            object_corners = len(approx)

            x, y, width, height = cv2.boundingRect(array=approx)
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+width, y+height), color=(0, 0, 255), thickness=3)

            if object_corners == 3: object_shape = 'triangle'
            elif object_corners == 4:
                aspect_ratio = width / float(height)
                if (aspect_ratio > 0.95) and (aspect_ratio < 1.05): object_shape = 'square'
                else: object_shape = 'rectangle'
            elif object_corners > 4: object_shape = 'circle'

            cv2.putText(img=img,
                        text=object_shape,
                        org=(x, y + height + 20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.7,
                        color=(0, 0, 0),
                        thickness=2
                        )

    return img


image = cv2.imread('resources/shape.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (7, 7), 1)
canny_img = cv2.Canny(blur_img, 50, 50)

contour_img = get_contour(image.copy(), canny_img)

stacked_img = si.stacked_images(0.6, [image, gray_img, blur_img], [canny_img, contour_img])
cv2.imshow('Stacked', stacked_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
