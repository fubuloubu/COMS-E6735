#!/usr/bin/python
import sys
import utils
progname = 'locate.py'
errorstring = '''
locate.py: Stats
       Guitar Coordinates:  {}
 Picking Hand Coordinates:  {}
Fretting Hand Coordinates:  {}
locate.py: Stats
'''

# Verify results of each frame
def verify(filename):
    def results_fun(frame_data):
        # Half the battle is finding the guitar
        results  = 50*int(len(frame_data[0]) == 4)
        # Then the hands
        results += 25*int(len(frame_data[1]) == 4)
        results += 25*int(len(frame_data[2]) == 4)
        return results
    utils.getresults(filename, errorstring, results_fun)

# Initialize cascade classifiers
guitar_cascade = utils.Cascade('bass_classifier.xml')
pickhand_cascade = utils.Cascade('bass_pickhand_classifier.xml')
frethand_cascade = utils.Cascade('bass_frethand_classifier.xml')

# Main reterival function
def guitar_and_hands(frame):
    # Find Coordinates of guitar
    guitar_coords = guitar_cascade.detect(frame)
    # Crop frame to guitar
    guitar_crop = utils.crop(frame, guitar_coords)
    # Get hands inside cropped guitar frame
    pickhand_coords = pickhand_cascade.detect(guitar_crop)
    frethand_coords = frethand_cascade.detect(guitar_crop)
    # Re-orient hand coordinates back to full frame
    pickhand_coords = utils.reposition(pickhand_coords, guitar_coords)
    frethand_coords = utils.reposition(frethand_coords, guitar_coords)
    return (guitar_coords, pickhand_coords, frethand_coords)

# Apply above locate functions for POI,
# and show them with bounded boxes for user
def main(frame):
    # Get guitar and hand bounding boxes
    (guitar_coords, pickhand_coords, frethand_coords) = \
        guitar_and_hands(frame)
    # Print for debugging and testing purposes
    sys.stderr.write( errorstring.format( \
            guitar_coords, pickhand_coords, frethand_coords) ) 
    # Add rectangles for debugging purposes around them
    frame = utils.addrectangle(frame, guitar_coords)
    frame = utils.addrectangle(frame, pickhand_coords)
    frame = utils.addrectangle(frame, frethand_coords)
    return frame

if __name__ == '__main__':
    if len(sys.argv) == 2:
        utils.test(sys.argv[1], main)
    else:
        utils.capture(main)
