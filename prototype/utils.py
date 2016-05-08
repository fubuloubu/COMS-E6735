#!/usr/bin/python
import cv2
import numpy as np
import random
import math
import sys
import operator as op
from jenks import jenks
from progress.bar import Bar

# Flip video around
def mirror(frame):
    return cv2.flip(frame,1)

# Crop frame around rectangle
def crop(frame, rect=None):
    if rect is None:
        return frame
    else:
        for x1, y1, x2, y2 in rect:
            return frame[y1:y2, x1:x2]

# Re-reference rectangle out of cropped frame
# by adding the upper left-hand x, y coordinates
# to the cropped object
def reposition(cropped_obj, crop_coords):
    # Sometimes rectangles are encapsulated
    if len(crop_coords) == 1 and len(crop_coords[0]) == 4:
        crop_coords = crop_coords[0]
    # Only apply crop if the cropped object
    # is a point (x,y) or rectangle (x1,y1,x2,y2) object
    # and if the crop is a rectangle as well
    if (len(cropped_obj) == 2 and \
        len(cropped_obj) == 4) or \
        len(crop_coords) == 4:
        return [p + crop_coords[i % 2] for i, p in enumerate(cropped_obj)]
    else:
        sys.stderr.write("Not cropping this:\n{}\nwith this:\n{}\n".format(\
                cropped_obj, crop_coords))
        return cropped_obj

def blankframe(frameToCopy=None, size=[640,480]):
    if frameToCopy is not None:
        size = frameToCopy.shape[:2]
    return np.zeros((size[0], size[1], 3), np.uint8)

# Draw everything as black outside of contour
def cut(frame, contour):
    contourframe = blankframe(frame)
    contourframe = addcontour(contourframe, contour, color=(255,255,255), fill=True)
    contourframe = colorify(blackandwhite(contourframe))
    return cv2.bitwise_and(frame, contourframe)

# Rotation function
def rotate(frame, deg=0):
    rows,cols,k = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    return cv2.warpAffine(frame,M,(cols,rows))

# Return inverted color image
def invert(frame):
    return cv2.bitwise_not(frame)

# Return black and white image thresholded at RGB 0F0F0F
def blackandwhite(frame, lower=(127,127,127),upper=(255,255,255)):
    return inrange(frame, lower, upper)

# Grayscale filter
def grayscale(frame):
    return cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

# Undoes grayscale conversion for display
# NOTE: Still is grayscale, just different type array
def colorify(frame):
    return cv2.cvtColor( frame, cv2.COLOR_GRAY2RGB )

# RGB to HSV color scheme
def rgb2hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# HSV to RGB color scheme
def hsv2rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

# RGB to YCrCb color scheme
def rgb2ycrcb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

# YCrCb to RGB color scheme
def ycrcb2rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)

# Threshold frame to between lower and upper (inside = white, outside = black)
def inrange(frame, lower, upper):
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    return cv2.inRange(frame, lower, upper)

# Gaussian filter
def gaussian(frame, ksize=(1, 1), sigma=(3, 3)):
    return cv2.GaussianBlur(frame, ksize, sigma[0], sigma[1], cv2.BORDER_CONSTANT)

# Bilateral filter
def bilateral(frame, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)

# Erosion filter
def erode(frame, kernalSize=(3, 3), iterations=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernalSize)
    return cv2.erode(frame, kernel, iterations=iterations)

# Dialation filter
def dialate(frame, kernalSize=(3, 3), iterations=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernalSize)
    return cv2.dilate(frame, kernel, iterations=iterations)

# Apply grayscale thresholding to frame using gaussian blur and a MEAN adaptive threshold
def adaptivethreshold(frame, maxValue=250, blockSize=7, C=9):
    frame = gaussian(frame)
    frame = grayscale(frame)
    return cv2.adaptiveThreshold(frame, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)

# Edge Detect Transformation
def edgedetect(frame, threshold1=50, threshold2=300, aperature_size=5):
    frame = gaussian(frame)
    frame = grayscale(frame)
    return cv2.Canny(frame, threshold1, threshold2, aperature_size)

# Detect lines in frame
def linedetect(frame, preprocessor=adaptivethreshold, threshold=50, minLineLength=30, maxLineGap=5):
    frame = preprocessor(frame)
    # rho = 1, theta = Pi/180 or 1 deg
    lines = cv2.HoughLinesP(frame, 1, cv2.cv.CV_PI/180, threshold, 0, minLineLength, maxLineGap)
    if lines is None:
        return []
    return lines.tolist()[0]

# Compute a linemodel 'lm' from a line 'l'
def tolinemodel(l):
    lm = {}
    lm["line"] = l
    lm["length"] = math.sqrt((l[1]-l[3])**2 + (l[0]-l[2])**2)
    if l[0] == l[2]:
        lm["slope"] = float('inf') # vertical line
        lm["intercept"] = float('inf') # y-intercept is undefined
        lm["angle"] = 90 # vertical line (in deg)
        lm["origin"] = l[0] # dist from origin for vertical line is simply x
    else:
        lm["slope"] = (l[3]-l[1])/float(l[2]-l[0]) # m = y2 -y1 / x2 - x1
        lm["intercept"] = l[1] - lm["slope"]*l[0] # b = y - mx
        lm["angle"] = math.atan(lm["slope"]) # theta = atan(m), ignore quadrants
        lm["origin"] = int(lm["intercept"]*math.cos(lm["angle"])) # shortest distance from origin
        lm["angle"] = math.degrees(lm["angle"]) # Save angle in degrees
    return lm

# Compute a linemodel 'lm' from combining two line models 'lm1', 'lm2'
def combine_linemodel(lm1, lm2):
    lm = {}
    
    # Average all this stuff
    lm["origin"] = (lm1["origin"] + lm2["origin"])/2
    lm["angle"] = (lm1["angle"] + lm2["angle"])/2
    
    # Angle of distance measurement perpidicular to the line
    lm["slope"] = math.tan(math.radians(lm["angle"]))
    perp_ang = math.radians(90 - lm["angle"])
    
    # Closest point on line measured to origin
    closest_pt = (lm["origin"]*math.cos(perp_ang), lm["origin"]*math.sin(perp_ang))
    lm["intercept"] = closest_pt[1] - lm["slope"]*closest_pt[0] # b = y - mx
    
    # Merge lines onto ray with same perpindicular origin distance
    def shift_line(l):
        d = lm["origin"] - l["origin"]
        dx = int(d*math.cos(perp_ang))
        dy = int(d*math.sin(perp_ang))
        line = l["line"]
        return [line[0]+dx, line[1]+dy, line[2]+dx, line[3]+dy]
    line1 = shift_line(lm1)
    line2 = shift_line(lm2)

    # Make the longest line possible using both sets of line endpoints
    length = lambda l:  math.sqrt((l[3]-l[1])**2 + (l[2]-l[0])**2)
    def merge_line(l1, l2):
        begin1 = [l1[0], l1[1]]
        begin2 = [l2[0], l2[1]]
        end1   = [l1[2], l1[3]]
        end2   = [l2[2], l2[3]]
        if math.isinf(lm["intercept"]):
            return [min(begin1[0], begin2[0]), begin1[1], max(end1[0], end2[0]), end1[1]]
        intercept_pt = [0, lm["intercept"]]
        if length(intercept_pt + begin1) < length(intercept_pt + begin2):
            # Then line 1 begins left of line 2, so use line 1 begins as an measuring point
            if length(begin1 + end2) > length(begin1 + end1):
                return begin1 + end2
            else:
                return begin1 + end1
        else:
            # Then line 1 begins right of line 2, so use line 2 begins as an measuring point
            if length(begin2 + end2) > length(begin2 + end1):
                return begin2 + end2
            else:
                return begin2 + end1

    lm["line"] = merge_line(line1, line2)
    lm["length"] = length(lm["line"])
    
    return lm

# Binary thresholding processor for skin detection
def skinthresh(frame, minthresh=(0,140,95), maxthresh=(255,170,135)):
    # Preprocessing
    frame = gaussian(frame)
    # Convert to HSV
    frame = rgb2ycrcb(frame)
    # Find region with skin tone in HSV image
    frame = inrange(frame, minthresh, maxthresh)
    #Postprocessing
    frame = erode(frame)
    frame = dialate(frame)
    return frame

# Get contours for the input black/white image
def getcontours(frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect skin using thresholding, attributed to Sam Kahn:
# https://github.com/seereality/opencvDemos/blob/master/skinDetect.py
# Erosion and Dialation tricks attributed to pyimagesearch
# http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
def skindetect(frame, binarypreprocessor=skinthresh):
    # Process frame to extract binary color
    frame = binarypreprocessor(frame)
    # Contourize that region and extract
    contours = getcontours(frame)
    # Filter only contours with a large enough area to be skin
    contours = filter(lambda c: cv2.contourArea(c) > 1000, contours)
    if len(contours) == 0:
        return #nothing
    # Return largest
    largest_area = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area > largest_area:
            largest_contour = c
            largest_area = area

    return largest_contour

# Return an array of convex hull points on the contour
def convexhull(contour):
    if contour is None or len(contour) == 0:
        return []
    point_list = cv2.convexHull(contour)
    return [ list(p[0]) for p in point_list]

# Skeletonize the frame
# http://opencvpython.blogspot.in/2012/05/skeletonization-using-opencv-python.html
def skeletonize(frame, maxIterations=20):
    size = np.size(frame)
    skel = blankframe(frame)
    iterations = 0
    while iterations < maxIterations:# and not cv2.countNonZero(frame):
        eroded = erode(frame)
        temp = dialate(eroded)
        temp = cv2.subtract(frame,temp)
        skel = cv2.bitwise_or(skel,temp)
        frame = eroded
        iterations += 1
    
    return skel

# Helper class to use Cascade Classifier functionality
class Cascade:
    def __init__(self, training_file):
       self.cc = cv2.CascadeClassifier(training_file)
       self.detected_obj = None

    def detect(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(400,200), maxSize=(1600,800)):
        objects = self.cc.detectMultiScale(frame, scaleFactor, 
                minNeighbors, cv2.CASCADE_SCALE_IMAGE, minSize, maxSize)
        sys.stderr.write("Objects Detected: {}\n".format(len(objects)))
        # function returns tuple if nothing was found
        # else returns numpy ndarray list of rectangles
        if len(objects) == 0:
            return []
        # Turn the [x, y, w, h] rectangle into
        # an [x1, y1, x2, y2] rectangle
        objects[:, 2:] += objects[:, :2]
        return np.asmatrix(objects).tolist()
        # If more than one object, try to find
        # closest to prior selection
        if len(objects) != 1:
            # Only find closest if a prior was found
            if self.detected_obj is not None:
                candidate = objects[0]
                # Cost function balances distance translation
                dx = 1
                dy = 1
                # and change in window size
                dwx = 5
                dwy = 5
                costfunc = lambda row: abs(row-self.detected_obj)*\
                        np.matrix([[dx],[dy],[dwx],[dwy]])
                cost = costfunc(candidate)
                for row in objects:
                    rowcost = costfunc(row)
                    if rowcost < cost:
                        candidate = row
                        cost = rowcost
                self.detected_obj = candidate
        else:
            # Only one object found, return that one
            self.detected_obj = objects[0]
        return np.asmatrix(self.detected_obj).tolist()

import operator as op
# Implementation of Lloyd's algorithm
# Adapted from https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# Fast implementation of jenks: https://github.com/perrygeo/jenks
def cluster(items, value=None, K=None):
    if value is None:
        raise ValueError("Distance function not set")
    if K is None:
        raise ValueError("Parameter K not set")

    if len(items) <= K:
        sys.stderr.write("WARNING: NOT ENOUGH ITEMS!\nInput List Size: {}\n".format(len(items)))
        return [[i] for i in items]
    
    distance = lambda p1, p2: value(p1) - value(p2)
    breakpoints = sorted(map(value, items))
    sys.stderr.write("Sorted values list:\n{}\n".format(breakpoints))
    
    # Find natural jenks breakpoints of items
    breakpoints = jenks(breakpoints, K)
    # Remove duplicate
    #breakpoints = list(set(breakpoints))
    sys.stderr.write("Breakpoints:\n{}\n".format(breakpoints))
    
    clustered_items = []
    last_bp = None
    # Lambda to test if item's distance is between breakpoints, using interval: (bp1, bp2]
    between = lambda item: value(item) > last_bp and value(item) <= bp
    # Group items using breakpoints
    for bp in breakpoints:
        # Jenks returns zero distance above
        if last_bp is None:
            last_bp = bp
            continue
        between_items = filter(between, items)
        sys.stderr.write("Values between {} and {}:\n{}\n".format(last_bp, bp, map(value, between_items)))
        if len(between_items) == 0:
            last_bp = bp
            continue
        clustered_items.append(between_items)
        last_bp = bp
    sys.stderr.write("{} clusters found (K={})\nItems:\n{}\n".format(len(clustered_items), K, clustered_items))
    return clustered_items

# Defaults for drawing functions
defaultcolor = (255, 0, 255) # magenta
thickness = 1
lineType = cv2.CV_AA

# Add text to frame at the specified location
def addtext(frame, text="Hello, world!", location="cc", color=defaultcolor):
    # Sizing constants
    (h, w) = frame.shape[:2]
    # Display settings
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = h/480 #scale with height of frame
    linespace = 12*fontScale
    lines = text.split('\n')
    # Function used to create coordinates for each line
    # center level is set by finding center point and biasing each line top down
    numlines = len(lines)
    # hack: if displaying on the lower part of the frame, 
    #       go backwards and fill upwards
    if location[0] == 'l':
        lines.reverse()
    coords = {
        "ul" : (lambda sh, sw, bl, i: (  0     +bl,   0+sw+bl +linespace*i                  )),
        "cl" : (lambda sh, sw, bl, i: (  0     +bl, h/2       +linespace*(i-(numlines-1)/2) )),
        "ll" : (lambda sh, sw, bl, i: (  0     +bl,   h-sw-bl -linespace*i                  )),
        "uc" : (lambda sh, sw, bl, i: (w/2-sh/2   ,   0+sw+bl +linespace*i                  )),
        "cc" : (lambda sh, sw, bl, i: (w/2-sh/2   , h/2       +linespace*(i-(numlines-1)/2) )),
        "lc" : (lambda sh, sw, bl, i: (w/2-sh/2   ,   h-sw-bl -linespace*i                  )),
        "ur" : (lambda sh, sw, bl, i: (  w-sh  -bl,   0+sw+bl +linespace*i                  )),
        "cr" : (lambda sh, sw, bl, i: (  w-sh  -bl, h/2       +linespace*(i-(numlines-1)/2) )),
        "lr" : (lambda sh, sw, bl, i: (  w-sh  -bl,   h-sw-bl -linespace*i                  )),
    }
    # For each line, grab line sizing information and add line to frame
    for i, line in enumerate(lines):
        ((sh,sw),bl) = cv2.getTextSize(line, fontFace, fontScale, thickness)
        cv2.putText(frame, line, coords[location](sh, sw, bl, i), 
                fontFace, fontScale, color, thickness, lineType)
    return frame

# Add a line or list of lines to the frame
def addline(frame, line, color=defaultcolor):
    if len(line) == 4:
        [x1, y1, x2, y2] = line
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, lineType) 
    return frame

# Add a circle centered at specified point with the specified radius
def addcircle(frame, point, radius=10, color=defaultcolor, fill=False):
    if len(point) == 2:
        [x, y] = point
        if fill:
            cv2.circle(frame, (x, y), radius, color, cv2.cv.CV_FILLED, lineType)
        else:
            cv2.circle(frame, (x, y), radius, color, thickness, lineType)
    return frame
    
# Add a rectangle or list of rectangles to the frame
def addrectangle(frame, rect, color=defaultcolor):
    for [x1, y1, x2, y2] in rect:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType)
    return frame
    
# Add a contour to the frame
def addcontour(frame, contour, fill=False, color=defaultcolor):
    # contourIdx = -1, if negative draw all
    #cv2.drawContours(frame, contour, -1, color, thickness, lineType)
    #return frame
    return addshape(frame, contour, fill, color)

# Add shape described by list of points in clockwise direction to frame
# note: last point connects to first point
def addshape(frame, shape_pts, fill=False, color=defaultcolor):
    if shape_pts is None:
        return frame
    pts = np.array(shape_pts, np.int32)
    pts = pts.reshape((-1,1,2))
    if fill:
        cv2.fillPoly(frame, [pts], color, lineType)
    else:
        cv2.polylines(frame, [pts], True, color, thickness, lineType)
    return frame

class VideoHandler:
    def __init__(self, infile=None, outfile=None, open_window=True):
        if infile is None:
            self.infile = 0 # Live Feed Camera 1
        else:
            self.infile = infile
            print "READING FROM FILE: {}".format(self.infile)
        
        if outfile is None:
            self.outfile = None
        else:
            self.outfile = outfile
        
        if open_window:
            self.window_name = "Video Display Window"
        else:
            self.window_name = None

    def __enter__(self):
        if self.window_name is not None:
            # Initialize a full-screen window
            print "INITIALIZING {}".format(self.window_name)
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)          
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        else:
            print "WARNING: NO DISPLAY SET!"
        
        # Initialize capture object
        self.cap = cv2.VideoCapture(self.infile)
        
        # Get/set codec information for video file
        default_codec = 'X264'
        if self.infile == 0:
            self.codec = default_codec
            self.fc = None
            self.bar = None
        else:
            # Set up codec using video file
            fourcc_int = int(self.cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
            fourcc_hex = hex(fourcc_int)[2:]
            if fourcc_hex == '0':
                raise IOError("No Codec Information in file {}".format(self.infile))
            else:
                # Characters are backwards, so decode and reverse
                self.codec = bytearray.fromhex(fourcc_hex).decode()[::-1]
            print 'VIDEO CODEC: {}'.format(self.codec)
            if self.codec == 'avc1':
                self.codec = default_codec
                print 'CHANGED TO: {}'.format(self.codec)
            
            # We're processing a video so setup the progress bar
            self.fc = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            print "FRAME COUNT: {}".format(self.fc)
            self.bar = Bar('Applying Transform', max=self.fc)

        # Get necessary video attributes from file
        self.fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        # HACK: if NaN, set to 30FPS
        if np.isnan(self.fps):
            print "WARNING: NO FPS SET!"
            self.fps = 20
        else:
            self.fps = int(round(self.fps))
        print "FPS: {}".format(self.fps)

        # Height and width
        self.w = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        print "FRAME SIZE: {} x {} px".format(self.w, self.h)
        self.fourcc = cv2.cv.CV_FOURCC(*self.codec)
        
        # Initialize video writer and progress bar
        if self.outfile is None:
            self.writer = None
        else:
            print "WRITING TO FILE: {}".format(self.outfile)
            self.fourcc = -1 #DEBUG
            self.writer = cv2.VideoWriter(self.outfile, self.fourcc, self.fps, (self.w, self.h), True)
        return self

    def get_frame(self):
        frame = self.cap.read()
        return frame

    def write_frame(self, frame):
        if self.writer is None:
            raise IOError("Outfile not writeable")
        if self.bar is not None:
            self.bar.next()
        if frame is not None:
            self.writer.write(frame)

    def display(self, frame):
        if self.window_name is None:
            raise IOError("Window not set up")
        if frame is not None:
            # Display the resulting frame fullscreen
            cv2.imshow(self.window_name, frame)

    def run(self, transform):
        # Get first frame
        ret, frame = self.get_frame()
        while(ret):
            # Our operations on the frame come here
            transform_frame = transform(frame)
            
            if self.window_name is not None:
                # Display frame
                self.display(transform_frame)
                
                # Break if user presses q key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if self.outfile is not None:
                # Write frame to file
                self.write_frame(transform_frame)
            
            # Get next frame so while loop can check status
            ret, frame = self.get_frame()
    
    def __exit__(self, exc_type, exc_value, traceback):
        # When everything done, release the capture
        if self.bar is not None:
            self.bar.finish()
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

# Capture live video and apply transformation function
def capture(transform=lambda x: x):
    with VideoHandler() as vh:
        vh.run(transform)

# Capture live video and silently apply transformation
def debug(transform=lambda x: x):
    with VideoHandler(open_window=False) as vh:
        vh.run(transform)

# Process infile, apply transform frame by frame,
# and show it. Note: removes sound
def show(infile, transform=lambda x: x):
    with VideoHandler(infile=infile) as vh:
        vh.run(transform)

# Process infile, apply transform frame by frame, no display
def test(infile, transform=lambda x: x):
    with VideoHandler(infile=infile, open_window=False) as vh:
        vh.run(transform)

# Save Displayed video while executing H264
def save(outfile, transform=lambda x: x):
    with VideoHandler(outfile=outfile) as vh:
        vh.run(transform)

# Process infile, apply transform frame by frame,
# writing to outfile. Note: removes sound
def execute(infile, outfile, show=False, transform=lambda x: x):
    with VideoHandler(infile=infile, outfile=outfile, open_window=show) as vh:
        vh.run(transform)

# Using errorstring, parse the results in filename into N records of M data
# The errorstring uses {0}, {1}, ... {M} codes to print random data
import re, ast
def getresults(filename, errorstring, resultsfun):
    # Remove the makefile constructs
    print "Video: {0}".format(".".join(filename.split('.')[1:-1]))
    # Get N array of strings matching frame using errorstring
    with open(filename, 'r') as f:
        # Find all matches to our given frame string, grabbing anything for a {} location
        frames = re.findall(errorstring.replace('{}','.*'), f.read())
        # Now get groups of data for each frame using a grouping regexp
        framedata = map(lambda f: re.findall(errorstring.replace('{}','(.*?)'),f), frames)
        # re.findall provides tuples if 2+ matches found, so clean that
        framedata = map(lambda f: [i for i in (f[0] if type(f[0]) is tuple else f)], framedata)
        # Evaluate the matches as literal python code, which should be acceptable 
        # as it should have been printed originally as built-ins
        framedata = map(lambda f: [ast.literal_eval(i) for i in f], framedata)
        # Execute the checking function given to us and print the results
        for data in framedata:
            result = resultsfun(data)
            print "Frame: {:>3}% Correct".format(result)

if __name__ == '__main__':
    # If running this as a script,
    # just run the mirror transformation on live feed
    capture(mirror)
