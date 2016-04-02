#!/usr/bin/python
import cv2
import numpy as np
from progress.bar import Bar

# Flip video around
def mirror(frame):
    return cv2.flip(frame,1)

# Grayscale filter
def greyscale(frame):
    return cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

# Gaussian filter
def gaussian(frame, ksize=(1, 1), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT):
    return cv2.GaussianBlur(frame, ksize, sigmaX, sigmaY, borderType)

# Crop frame around rectangle
def crop(frame, rect=None):
    if rect is None:
        return frame
    else:
        for x1, y1, x2, y2 in rect:
            return frame[y1:y2, x1:x2]

# Re-reference rectangle out of cropped frame
# By moving upper left and lower right vertices
# in the +x and +y direction by bounding box
def uncrop(inner_rect, outer_rect):
    if outer_rect is None or inner_rect is None:
        return inner_rect
    else:
        return inner_rect + outer_rect\
            *np.matrix('1 0 1 0; 0 1 0 1; 0 0 0 0; 0 0 0 0')

# Rotation function
def rotate(frame, deg=0):
    rows,cols,k = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    return cv2.warpAffine(frame,M,(cols,rows))

# Helper class to use Cascade Classifier functionality
class Cascade:
    def __init__(self, training_file):
       self.cc = cv2.CascadeClassifier(training_file)
       self.detected_obj = None

    def detect(self, frame, scaleFactor=1.3, minNeighbors=5):
        objects = self.cc.detectMultiScale(frame, scaleFactor, 
                minNeighbors, flags=cv2.CASCADE_SCALE_IMAGE)
        print 'Number of objects found: {}'.format(len(objects))
        # function returns tuple if nothing was found
        # else returns numpy ndarray list of rectangles
        if len(objects) == 0:
            return []
        # Turn the [x, y, w, h] rectangle into
        # an [x1, y1, x2, y2] rectangle
        objects[:, 2:] += objects[:, :2]
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

# Add text to frame at the specified location
def addtext(frame, text="Hello, world!", location="cc"):
    # Display settings
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    color = (255, 0, 255) # magenta
    thickness = 1
    lineType = cv2.CV_AA
    # Sizing constants
    (h, w) = frame.shape[:2]
    linespace = 12
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

# Convert rectangles into shape point list for below
def addrectangle(frame, rect):
    if len(rect) == 0:
        return frame
    for x1, y1, x2, y2 in rect:
        pts = [(x1, y1),(x1, y2),(x2, y2),(x2, y1)]
    return addshape(frame, pts)

# Add shape described by list of points in clockwise direction to frame
# note: last point connects to first point
def addshape(frame, shape_pts):
    color = (255, 0, 255) # magenta
    thickness = 1
    lineType = cv2.CV_AA
    pts = np.array(shape_pts, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame, [pts], True, color, thickness, lineType)
    return frame

# capture live video and apply transformation function
def capture(transform=lambda x: x):
    cap = cv2.VideoCapture(0)
    
    # Initialize a full-screen window
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        transform_frame = transform(frame)

        # Display the resulting frame fullscreen
        cv2.imshow("test", transform_frame)
        
        # Break if user presses q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Save Displayed video while executing H264
def save(transform=lambda x: x, outfile='test.avi', codec='MJPG', fps=30):
    # initialize the FourCC obj, video writer, 
    # and dimensions of the frame
    fourcc = cv2.cv.CV_FOURCC(*codec)
    global writer
    writer = None
    # Helper function to execute transform, write to file,
    # then return transformed frame for display
    def exec_and_write(frame):
        # On first capture, initialize the video writer
        global writer
        if writer is None:
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter(outfile, fourcc, fps, (w, h), True)
        frame = transform(frame)
        writer.write(frame)
        return frame

    # Call capture with helper function
    capture(exec_and_write) #loop until user finishes

    # Clean up writer
    writer.release()

# Process infile, apply transform frame by frame,
# writing to outfile. Note: removes sound
def execute(infile, outfile, transform=lambda x: x):
    cap = cv2.VideoCapture(infile)

    # Get necessary video attributes from file
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # HACK: if NaN, set to 30FPS
    if np.isnan(fps):
        fps = 30
    else:
        fps = int(round(fps))
    fourcc = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
    w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer and progress bar
    writer = cv2.VideoWriter(outfile, fourcc, fps, (w, h), True)
    bar = Bar('Applying Transform', max=fc)

    # Initialize first frame
    ret, frame = cap.read()
    # While there are frames in video, apply transform and write
    while(ret):
        bar.next()
        frame = transform(frame)
        writer.write(frame)
        ret, frame = cap.read()
    # When everything done, release the capture and writer
    bar.finish()
    cap.release()
    writer.release()

# Process infile, apply transform frame by frame,
# and show it. Note: removes sound
def show(infile, transform=lambda x: x):

    # Initialize a full-screen window
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    
    # Helper function to apply transform to file
    def transform_and_show(frame):
        cv2.imshow("test", transform(frame))

    # Write to a temporary file then execute using helper
    outfile = 'temp_' + infile
    execute(infile, outfile, transform_and_show)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # If running this as a script,
    # just run the mirror transformation on live feed
    capture(mirror)
