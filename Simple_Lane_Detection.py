import cv2
import numpy as np
import stacked_images as si
pre_lines = []
pre_left_lines = []
pre_right_lines = []
capture = cv2.VideoCapture('resources/driving_3.mp4')

# function detects given color range (here detects white and yellow lane lines)
def detect_color(img, lower_range, upper_range):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([lower_range]), np.array([upper_range]))
    return_img = cv2.bitwise_and(img, img, mask=mask)
    return return_img

# function processes given image and returns edge detected image
def process_frame(img):
    yellow_line = detect_color(img.copy(), [12, 46, 87], [35, 255, 255])
    white_line = detect_color(img.copy(), [0, 0, 114], [177, 29, 242])
    comb_img = cv2.bitwise_or(yellow_line, white_line)
    gray = cv2.cvtColor(comb_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 255, 255)
    kernel = np.ones((5, 5))
    canny_dilate = cv2.dilate(canny, kernel, iterations=2)
    canny_erode = cv2.erode(canny_dilate, kernel, iterations=1)
    return canny_erode

# function gives region of interests and masks everything else
def find_roi(img, vertices):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, np.array([vertices], np.int32), (255, 255, 255))
    return_img = cv2.bitwise_and(img, mask, 1)
    return return_img

# function draws line on given image
def draw_lines(img, lines):
    canvas = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 5)
    return_img = cv2.addWeighted(img, 0.8, canvas, 1, 0.0)
    return return_img

# function returns coordinates of line for given slope and intercept
def find_coordinate(img, line):
    if line is not None:
        slope, intercept = line[0], line[1]
        h = img.shape[0]
        # values for Ys' are obvious
        y1 = h
        y2 = h//2 + 150
        # values for Xs' are obtained from eqn of line (y = mx + c)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

# function applied second degree polynomial to obtain slopes and intercept
# taking average values of slopes and intercepts, we can obtain single line for either side
def average_slope_intercept(img, lines):
    global pre_lines, pre_left_lines, pre_right_lines
    if lines is None:
        lines = pre_lines

    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # obtain second degree polynomial parameters from numpy.polyfit
        # function returns slope and intercept for given x-y values
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1)
        slope, intercept = parameters[0], parameters[1]
        # separates left lines and right lines
        if slope < 0:  # when x increases y decreases i.e. -ve slope
            left_lines.append((slope, intercept))
        else:  # when x increases y increases i.e. +ve slope
            right_lines.append((slope, intercept))

    # if lines are less than 12, use previous lines
    if len(left_lines) < 12:
        left_lines = pre_left_lines
    if len(right_lines) < 12:
        right_lines = pre_right_lines

    # find average values for slopes and intercepts
    average_left = np.average(left_lines, axis=0)
    average_right = np.average(right_lines, axis=0)
    left_average_line = find_coordinate(img.copy(), average_left)
    right_average_line = find_coordinate(img.copy(), average_right)

    # store lines in global variables
    pre_lines = lines
    pre_left_lines = left_lines
    pre_right_lines = right_lines
    return np.array([left_average_line, right_average_line])

# function fills the area inside the lines
def fill_area(img, lines):
    mask = np.zeros_like(img)
    left = lines[0]
    right = lines[1]
    x1, y1 = left[0], left[1]
    x2, y2 = left[2], left[3]
    x3, y3 = right[2], right[3]
    x4, y4 = right[0], right[1]
    vertices = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.fillPoly(mask, np.array([vertices], np.int32), (0, 255, 0))
    return_img = cv2.addWeighted(img, 1.2, mask, 0.5, 0.0)
    return return_img


while capture.isOpened():
    ret, frame = capture.read()
    height = frame.shape[0]
    width = frame.shape[1]
    processed_frame = process_frame(frame.copy())

    roi_points = [(0+200, height),
                  (width//2 - 90, 450),
                  (width//2 + 90, 450),
                  (width-200, height)]

    roi_frame = find_roi(processed_frame.copy(), roi_points)

    lines = cv2.HoughLinesP(roi_frame, 1, np.pi/180, 20, minLineLength=2, maxLineGap=1000)

    smooth_lane_lines = average_slope_intercept(frame.copy(), lines)
    smooth_lane = draw_lines(frame.copy(), smooth_lane_lines)

    fill_lane = fill_area(smooth_lane.copy(), smooth_lane_lines)

    warp_roi = bird_view(roi_frame, roi_points)

    stack = si.stacked_images(0.6, [frame], [fill_lane])
    cv2.imshow('road', stack)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
