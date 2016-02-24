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

# Main function captures video,
# and can by called with a transformation function
def main(transform=lambda x: x):
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

if __name__ == '__main__':
    main(mirror)
