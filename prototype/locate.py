#!/usr/bin/python
import utils
import cv2

# Find body of guitar for use in hand location algorithms
def guitar(frame):
    guitar_cascade = cv2.CascadeClassifier('guitar_classifier.xml')
    guitars = guitar_cascade.detectMultiScale(frame, 1.3, 5)
    location_coords = guitars
    return location_coords

# Using guitar location in frame, find the picking hand location
def picking_hand(frame, guitar_coords):
    location_coords = []
    return location_coords

# Using guitar location in frame, find the fretting hand location
def fretting_hand(frame, guitar_coords):
    location_coords = []
    return location_coords

# Apply above locate functions for POI,
# and show them with bounded boxes for user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Find Coordinates of objects
    guitar_coords = guitar(frame)
    picking_hand_coords = picking_hand(frame, guitar_coords)
    fretting_hand_coords = fretting_hand(frame, guitar_coords)
    # Add Shapes around them
    frame = utils.addshape(frame, guitar_coords)
    frame = utils.addshape(frame, picking_hand_coords)
    frame = utils.addshape(frame, fretting_hand_coords)
    return frame

if __name__ == '__main__':
    utils.capture(main)
