#!/usr/bin/python
import utils
import locate

# Intepret hand area of frame into hand model
def to_handmodel(hand_image):
    handmodel = []
    return handmodel

# Transform hand model to a wireframe for display
def to_wireframe(handmodel):
    wireframe = []
    return wireframe

# Get the hand models for both hands
def get_hands(frame, pickhand_coords, frethand_coords):
    # Interpret hand areas to get hand models for both hands
    pickhand = to_handmodel(utils.crop(frame, pickhand_coords))
    frethand = to_handmodel(utils.crop(frame, frethand_coords))
    return (pickhand, frethand)

# Get digit wireframe of hand and present to user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Locate pickhand and frethand
    (guitar_coords, pickhand_coords, frethand_coords) = 
        locate.guitar_and_hands(frame)
    # Get the hands in the frame
    (pickhand, frethand) =
        get_hands(frame, pickhand_coords, frethand_coords)
    # Add wireframes for hands for display
    frame = utils.addshape(frame, to_wireframe(pickhand))
    frame = utils.addshape(frame, to_wireframe(frethand))
    return frame

if __name__ == '__main__':
    utils.capture(main)
