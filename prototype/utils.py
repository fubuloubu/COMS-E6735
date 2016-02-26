#!/usr/bin/python
import cv2

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

def execute(infile, outfile, transform=lambda x: x):
    return

def save(outfile, codec, fps, transform=lambda x: x):
    # initialize the FourCC, video writer, dimensions of the frame, and
    # zeros array
    #fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = None
    (h, w) = (None, None)
    # check if the writer is None
    if writer is None:
        # initialize the video writer
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]
        #writer = cv2.VideoWriter(outfile, -1, fps, (w, h), True)
    
    # Helper function to execute transform
    # Write to file
    # Then return transformed frame for display
    def exe_and_write(x):
        x = transform(x)
        #writer.write(x)
        return x

    # Call capture with helper function
    capture(exe_and_write) #loop until user finishes

    # Clean up writer
    #writer.release()

if __name__ == '__main__':
    # If running this as a script,
    # just run the mirror transformation on live feed
    capture(mirror)
