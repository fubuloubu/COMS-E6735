#!/usr/bin/python
import utils

# Find body of guitar for use in hand location algorithms
guitar_cascade = utils.Cascade('guitar_classifier.xml')
def guitar(frame):
    guitar = guitar_cascade.detect(frame)
    return guitar

# Using guitar location in frame, find the picking hand location
pickhand_cascade = utils.Cascade('pickhand_classifier.xml')
def picking_hand(frame):
    pickhand = pickhand_cascade.detect(frame)
    return pickhand

# Using guitar location in frame, find the fretting hand location
frethand_cascade = utils.Cascade('frethand_classifier.xml')
def fretting_hand(frame):
    frethand = frethand_cascade.detect(frame)
    return frethand

# Main reterival function
def guitar_and_hands(frame):
    # Find Coordinates of guitar
    guitar_coords = guitar(frame)
    # Crop frame to guitar
    guitar_crop = utils.crop(frame, guitar_coords)
    # Get hands inside cropped guitar frame
    pickhand_coords = picking_hand(guitar_crop)
    frethand_coords = fretting_hand(guitar_crop)
    # Re-orient hand coordinates back to full frame
    pickhand_coords = utils.uncrop(pickhand_coords, guitar_coords)
    frethand_coords = utils.uncrop(frethand_coords, guitar_coords)
    return (guitar_coords, pickhand_coords, frethand_coords)

# Apply above locate functions for POI,
# and show them with bounded boxes for user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Get guitar and hand bounding boxes
    (guitar_coords, pickhand_coords, frethand_coords) = \
        guitar_and_hands(frame)
    # Add rectangles for debugging purposes around them
    frame = utils.addrectangle(frame, guitar_coords)
    frame = utils.addrectangle(frame, picking_hand_coords)
    frame = utils.addrectangle(frame, fretting_hand_coords)
    return frame

if __name__ == '__main__':
    utils.capture(main)
