import cv2
import numpy as np
import os

#creating a list of all provided images

imgs = os.listdir('./imgs')

n_houses = []
priority = []
rescue_ratio = []

#looping through every image
for __i in imgs:
    # Load the image and convert to HSV colourspace
    image = cv2.imread("./imgs/" + __i)
    img = image
    og = cv2.imread("./imgs" + __i)
    yellow=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo=np.array([4,0,0])
    brown_hi=np.array([37,255,255])



    # Mask image to only select browns
    mask = cv2.inRange(yellow,brown_lo,brown_hi)
    # Change image to yellow where we found brown
    image[mask>0]=(110, 238, 240)

    #mask the image to find green
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    #change green to blue
    img[mask>0] = (240, 230, 127)



    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Function to classify a contour as a triangle
    def is_triangle(cnt):
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        return len(approx) == 3

    # Create a copy of the original image
    output_image = image.copy()
    num_triangle = 0
    triangle_loc = []

    # Draw outlines around the detected triangles
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Filter out small contours
            if is_triangle(contour):
                # Get the bounding rectangle of the triangle
                x, y, w, h = cv2.boundingRect(contour)
                num_triangle += 1
                # Extract the region of interest (ROI) in the grayscale image
                roi = gray[y:y+h, x:x+w]
                # Calculate the mean pixel value of the ROI
                color = (0, 200, 0)
                cv2.drawContours(output_image, [contour], 0, color, 2)  # Draw outline

                
                num_triangle += 1
                triangle_loc.append([x,y,w,h])

                # cv2.circle(output_image, (x+7,y+7), 5, (255,0,0), 5)


    #print(image[triangle_loc[0][1]][triangle_loc[0][0]])
    #looping tghru triangles to find wch one is in burnt and unburnt area
    burnt_h = []
    unburnt_h = []
    blue_h = []
    red_h = []
    _b = [110, 238, 240]
    _ub = [240, 230, 127]
    burnt_priority = 0
    unburnt_priority = 0

    for h in triangle_loc:
        _area = image[h[1]+7][h[0]+7].tolist()
        _l = 0
        _c = 0

        if _area == _b:
            burnt_h.append(h)
            _l = 1
        elif _area == _ub:
            unburnt_h.append(h)
            
        elif _area[2] <= 50 and _area[0] == 0 and _area[1] == 0:  #this command is required because the yellow region or burnt region has some random black dots.
            burnt_h.append(h)
            _l = 1

        _center = (h[0] + h[2]//2, h[1] + h[3]//2)
        _cc = image[_center[1]][_center[0]].tolist()

        if _cc[2] > 200 and _cc[0] < 10 and _cc[1] < 10:
            red_h.append(h)
            _c = 1
        if _cc[0] > 200 and _cc[2] < 10 and _cc[1] < 10:
            blue_h.append(h)

        if _l == 1:
            if _c == 1:
                burnt_priority += 1
            elif _c == 0:
                burnt_priority += 2
        else:
            if _c == 1:
                unburnt_priority += 1
            elif _c == 0:
                unburnt_priority += 2

    # print("Number of houses in burnt region:", len(burnt_h))
    # print("Number of houses in unburnt region:", len(unburnt_h))

    # print("Priority in burnt region:", burnt_priority)
    # print("Priority in unburnt region:", unburnt_priority)

    n_houses.append([len(burnt_h), len(unburnt_h)])
    priority.append([burnt_priority, unburnt_priority])

    rescue_ratio.append(burnt_priority/unburnt_priority)

    cv2.imshow("result"+__i, output_image)


#printing the values found
print("Number of houses:", n_houses)
print("Priority list:", priority)
print("Rescue Ratios:", rescue_ratio)
print("Original images list:", imgs)


#arranging the image names in order of prority ratio
pairs = list(zip(imgs, rescue_ratio))

# Sort the pairs based on the ratios
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])

# Extract the sorted strings from the sorted pairs
sorted_imgs = [pair[0] for pair in sorted_pairs]


print("Imgs sorted in order of rescue ratio:", sorted_imgs)



cv2.waitKey(0)