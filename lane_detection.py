import cv2
import numpy as np

img = cv2.imread("original_lane.jpg")
# img = cv2.imread("original_lane.jpg") # BGR image
cv2.imshow("Original image", img)

# image shape 
imshape = img.shape

# Convert original image to grayscale
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

blur = cv2.GaussianBlur(hsv, (9, 9),0)

lower_bound = np.array([10, 15, 30])
upper_bound = np.array([255,85,255])

mask = cv2.inRange(blur, lower_bound, upper_bound)

transformed = cv2.bitwise_or(img,img, mask= mask)

result = cv2.cvtColor(transformed, cv2.COLOR_HSV2BGR)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# Blurrinf the image
blur = cv2.GaussianBlur(gray, (11, 11), 0)

# Edge detection
edge = cv2.Canny(blur, 90, 120)

# Define the coordinates of the left and right extreme lines
left_extreme_line = [[0, 200, 213, 186]]  # Example left extreme line
right_extreme_line = [[386, 163, 594, 173]]  # Example right extreme line

# Calculate the coordinates of the vertices of the trapezium
vertex1 = (0, imshape[1])
vertex2 = (left_extreme_line[0][0], left_extreme_line[0][1])  # Point where left extreme line touches left edge
vertex3 = (left_extreme_line[0][2], left_extreme_line[0][3])  # Other endpoint of left extreme line
vertex4 = (right_extreme_line[0][0], right_extreme_line[0][1])  # Point where right extreme line touches left edge
vertex5 = (right_extreme_line[0][2], right_extreme_line[0][3])
vertex6 = (imshape[1], imshape[0])  # Bottom-right corner of the image

# Define the vertices of the trapezium
vertices = np.array([[vertex1, vertex2, vertex3, vertex4, vertex5, vertex6]], dtype=np.int32)

# vertices = np.array([[(0,imshape[1]),(150, 400),(150,250),(imshape[0], imshape[1])]], dtype=np.int32)
# trapezium_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
mask = np.zeros_like(edge)   
# fill pixels inside the polygon defined by vertices"with the fill color  
color = 255
poly = cv2.fillPoly(mask, vertices, color)

#----------------------APPLY MASK TO IMAGE-------------------------------
# create image only where mask and edge Detection image are the same
maskedIm = cv2.bitwise_and(edge, mask)

#-----------------------Distance and Slope-------------------------------
def distance(*args):
    if len(args)==1:
        line = args[0]
        x1, y1, x2, y2 = line
    elif len(args) == 4:  
        x1, y1, x2, y2 = args
    else:
        raise ValueError("Invalid arguments. Either pass the line or the coordinates of the line.")
    return np.sqrt((x2-x1)**2+ (y2-y1)**2) 

def Slope(*args):
    if len(args)==1:
        line = args[0]
        x1, y1, x2, y2 = line
    elif len(args) == 4:  
        x1, y1, x2, y2 = args
    else:
        raise ValueError("Invalid arguments. Either pass the line or the coordinates of the line.")
    if x2-x1 == 0:
        return float("inf")
    else:
        return (y2-y1)/(x2-x1)

#--------------------------------Coordinate Extension--------------------------------
def extended_xy(n_x1, n_y1, n_x2, n_y2, y_intercept):
    slope = Slope(n_x1, n_y1, n_x2, n_y2)
    if len(y_intercept)==1:
        x_intercept = int(((y_intercept[0]-n_y1)/slope)+n_x1)
        return x_intercept, y_intercept[0]
    else:
        x_intercept1 = int(((y_intercept[0]-n_y1)/slope)+n_x1)
        x_intercept2 = int(((y_intercept[1]-n_y2)/slope)+n_x2)
        return x_intercept1, y_intercept[0], x_intercept2, y_intercept[1]

#-----------------------Apply Hought transform for line detection--------------------------
rho = 2
theta = np.pi/180
threshold = 45
min_line_len = 40
max_line_gap = 100
lines = cv2.HoughLinesP(maskedIm, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
height, width = img.shape[:2]
center_x, center_y = width//2, height//2
line_img = np.zeros_like(img)

#--------------------------------Sorting the lines based on the lenght of the line------------------------------
if lines is not None:
    sorted_lines = sorted(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2), reverse=True)

#-------------------------------------Storing the slope of every line-------------------------------------------
slope_of_line = [Slope(line[0]) for line in sorted_lines]

#-------------------------------------Filtering our lines with similar slope--------------------------------------
# Define a function to check if two slopes are similar based on a tolerance level
def are_slopes_similar(slope1, slope2, tolerance=0.3):
    return abs(slope1 - slope2) < tolerance

# Define a function to standardize the coordinates of the lines
def standardize_coordinates(lines, center_x, center_y):
    standardized_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1_from_center = distance(center_x, x1, center_y, y1)
        dist2_from_center = distance(center_x, x2, center_y, y2)
        if dist1_from_center < dist2_from_center:
            standardized_lines.append((x1, y1, x2, y2))
        else: 
            standardized_lines.append((x2, y2, x1, y1))
    return standardized_lines

# Standardize the coordinates of the detected lines
standardized_lines = np.array(standardize_coordinates(lines, center_x, center_y))

# Define a list to store unique slopes
unique_slopes = []

# Filter out lines with similar slopes
filtered_lines = []
for line in standardized_lines:
    slope = Slope(line)
    is_unique = True
    for existing_slope in unique_slopes:
        if are_slopes_similar(slope, existing_slope):
            is_unique = False
            break
    if is_unique:
        unique_slopes.append(slope)
        filtered_lines.append(line)

#-----------------------------------------Draw all lines onto the image-----------------------------------------
# Check if we got more than one line
if sorted_lines is not None and len(filtered_lines)>2:
    allLines = np.zeros_like(img)

    for x1, y1, x2, y2 in filtered_lines:
        if distance(x2, y2, x2, height)<50:
            extended_x1, extended_y1, extended_x2, extended_y2 = extended_xy(x1, y1, x2, y2, y_intercept = [160, height])
            line_img = cv2.line(allLines, (extended_x1, extended_y1), (extended_x2, extended_y2), (0,255,0), 2)
        else:
            extended_x1, extended_y1, = extended_xy(x1, y1, x2, y2, y_intercept = [160])
            line_img = cv2.line(allLines, (extended_x1, extended_y1), (x2, y2), (0,0,250), 2)
    
cv2.imshow("Line image",line_img)
cv2.imwrite("Detected_lines.jpg", line_img)
cv2.waitKey(-1)
