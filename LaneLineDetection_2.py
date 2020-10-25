import cv2
import numpy as np
import stacked_images as si
temp_o2 = 0
temp_x1, temp_y1, temp_x2, temp_y2 = 0, 0, 0, 0

def find_coordinate(img, avg_lines):
    # we have a slope and an intercept and we need to find coordinates for single line (either left or right)
    # y1 is obvious, height. we can select y2 as per we want.
    # x coordinates can be obtained by formula:- x = (y - c)/m, ||(i.e. y = mx + c)||
    slope, intercept = avg_lines[0], avg_lines[1]
    h = img.shape[0]
    y1 = h
    y2 = int(h*(3/4))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):
    left_line_parameters = []
    right_line_parameters = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # function returns slope and intercept for given lines
            parameters = np.polyfit(x=(x1, x2), y=(y1, y2), deg=1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:  # when x increases y decreases i.e. -ve slope
                left_line_parameters.append((slope, intercept))
            else:  # when x increases y increases i.e. +ve slope
                right_line_parameters.append((slope, intercept))

    # find average of slopes and intercepts -- axis=0 find average vertically (i.e. avg of a column)
    left_average = np.average(left_line_parameters, axis=0)
    right_average = np.average(right_line_parameters, axis=0)

    if isinstance(left_average, np.float64):
        left_line = find_coordinate(img, [-7.50947801e-01, 1256.14])
    else:
        left_line = find_coordinate(img, left_average)

    if isinstance(right_average, np.float64):
        right_line = find_coordinate(img, [0.88604126, -247.47])
    else:
        right_line = find_coordinate(img, right_average)

    return np.array([left_line, right_line])


def find_roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], np.int32), (255, 255, 255))
    return_roi = cv2.bitwise_and(img, mask)
    return return_roi


def fill_lane(img, lines):
    if lines is not None:
        mask = np.zeros_like(img)
        left = lines[0]
        right = lines[1]
        x1, y1 = left[0], left[1]
        x2, y2 = left[2], left[3]
        x3, y3 = right[2], right[3]
        x4, y4 = right[0], right[1]
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        cv2.fillPoly(mask, np.array([points], np.int32), (255, 0, 0))
        return_img = cv2.addWeighted(img, 0.8, mask, 0.5, 0.0)
    return return_img


def draw_lines(img, lines):
    canvas = np.zeros_like(img)
    global temp_x1, temp_y1, temp_x2, temp_y2
    if lines is not None:
        for line in lines:
            if len(line) > 1:
                x1, y1, x2, y2 = line
            else:
                x1, y1, x2, y2 = line[0]
            try:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 10)
            except OverflowError:
                print('cought')
                x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2
            temp_x1, temp_y1, temp_x2, temp_y2 = x1, y1, x2, y2
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 10)

        img = cv2.addWeighted(img, 0.8, canvas, 1, 0.0)
    return canvas, img


capture = cv2.VideoCapture('resources/driving_3.mp4')
# frame = cv2.imread('resources/road.JPG')

while capture.isOpened():
    ret, frame = capture.read()
# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 1)
# canny_frame = cv2.Canny(blur_frame, 100, 100)
# kernel = np.ones((5, 5))
# canny_frame_dilate = cv2.dilate(canny_frame, kernel, iterations=2)
# canny_frame_erode = cv2.erode(canny_frame_dilate, kernel, iterations=1)

    height = frame.shape[0]
    width = frame.shape[1]

    roi = [(0+200, height),
           (530, height/2 + 100),
           (730, height/2 + 100),
           (width-200, height)]
    roi_result = find_roi(frame.copy(), vertices=roi)
    ret, threshold_roi = cv2.threshold(roi_result, 130, 255, cv2.THRESH_BINARY)
    print(threshold_roi.shape)

    canny_frame = cv2.Canny(threshold_roi, 100, 100)
    kernel = np.ones((5, 5))
    canny_frame_dilate = cv2.dilate(canny_frame, kernel, iterations=2)
    canny_frame_erode = cv2.erode(canny_frame_dilate, kernel, iterations=1)

    lines = cv2.HoughLinesP(canny_frame_erode, 1, np.pi/180, 50, minLineLength=2, maxLineGap=200)
    average_lines = average_slope_intercept(frame.copy(), lines)
    _, lanes = draw_lines(frame.copy(), lines)
    canv, lane = draw_lines(frame.copy(), average_lines)
    fill_area = fill_lane(frame.copy(), average_lines)
    stack = si.stacked_images(0.3, [frame, canny_frame_erode, fill_area], [lane, canv, lanes])
    cv2.imshow('Edge Detected', stack)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()

