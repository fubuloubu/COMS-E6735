#!/usr/bin/python
import utils
import locate

# Intepret hand area of frame into hand model
def to_handmodel(frame, hand_coords):
    handmodel = []
    return handmodel

# Transform hand model to a wireframe for display
def to_wireframe(handmodel):
    wireframe = []
    return wireframe

# Get the hand models for both hands
def get_hands(frame):
    # Find Coordinates of objects
    guitar_coords = locate.guitar(frame)
    picking_hand_coords = locate.picking_hand(frame, guitar_coords)
    fretting_hand_coords = locate.fretting_hand(frame, guitar_coords)
    # Interpret hand areas to get hand models for both hands
    picking_hand = to_handmodel(frame, picking_hand_coords)
    fretting_hand = to_handmodel(frame, fretting_hand_coords)
    return (picking_hand, fretting_hand)

# Get digit wireframe of hand and present to user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Get the hands in the frame
    (picking_hand, fretting_hand) = get_hands(frame)
    # Add wireframes for hands for display
    frame = utils.addshape(frame, to_wireframe(picking_hand))
    frame = utils.addshape(frame, to_wireframe(fretting_hand))
    return frame

if __name__ == '__main__':
    utils.capture(main)
