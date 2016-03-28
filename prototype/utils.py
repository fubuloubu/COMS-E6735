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

# Rotation function
def rotate(frame, deg=0):
    rows,cols,k = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    return cv2.warpAffine(frame,M,(cols,rows))

# Add text to frame at the specified location
def addtext(frame, text="Hello, world!", location="cc"):
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    color = (255, 0, 255) # magenta
    thickness = 1
    ((sh,sw),bl) = cv2.getTextSize(text, fontFace, fontScale, thickness)
    (h, w) = frame.shape[:2]
    coords = {
        "ul" : (  0     +bl,   0+sw+bl),
        "cl" : (  0     +bl, h/2      ),
        "ll" : (  0     +bl,   h-sw-bl),
        "uc" : (w/2-sh/2   ,   0+sw+bl),
        "cc" : (w/2-sh/2   , h/2      ),
        "lc" : (w/2-sh/2   ,   h-sw-bl),
        "ur" : (  w-sh  -bl,   0+sw+bl),
        "cr" : (  w-sh  -bl, h/2      ),
        "lr" : (  w-sh  -bl,   h-sw-bl),
    }
    lineType = cv2.CV_AA
    cv2.putText(frame, text, coords[location], fontFace, fontScale, color, thickness, lineType)
    return frame

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

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        transform_frame = transform(frame)

        # Display the resulting frame fullscreen
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.imshow("test", transform_frame)
        
        # Break if user presses q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

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

if __name__ == '__main__':
    # If running this as a script,
    # just run the mirror transformation on live feed
    capture(mirror)
