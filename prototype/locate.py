#!/usr/bin/env python
import sys
import utils
progname = 'locate.py'
errorstring = '''
locate.py: Stats
       Guitar Coordinates:  {}
locate.py: Stats
'''

# Verify results of each frame
def verify(filename):
    def results_fun(frame_data):
        results  = 100*int(len(frame_data[0]) == 4)
        return results
    utils.getresults(filename, errorstring, results_fun)

# Initialize cascade classifiers
guitar_cascade = utils.Cascade('bass_classifier.xml')

# Main reterival function
def guitar(frame):
    # Find Coordinates of guitar
    guitar_coords = guitar_cascade.detect(frame)
    return guitar_coords

# Apply above locate functions for POI,
# and show them with bounded boxes for user
def main(frame):
    # Get guitar and hand bounding boxes
    guitar_coords = guitar(frame)
    # Print for debugging and testing purposes
    sys.stderr.write( errorstring.format(guitar_coords) ) 
    # Add rectangles for debugging purposes around them
    frame = utils.addrectangle(frame, guitar_coords)
    return frame

if __name__ == '__main__':
    if len(sys.argv) == 2:
        utils.test(sys.argv[1], main)
    else:
        utils.capture(main)
