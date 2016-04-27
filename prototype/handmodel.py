#!/usr/bin/python
import utils
import locate

# Intepret hand area of frame into hand model
# Handmodel: (x, y) pairs of fingertip points
def to_handmodel(hand_image):
    handcontour = utils.skindetect(hand_image)
    convex_pts = utils.convexhull(handcontour)
    handmodel = [[75, 75]]
    return handmodel

# Get the hand models for both hands
def get_hands(frame, pickhand_coords, frethand_coords):
    # Interpret hand areas to get hand models for both hands
    pickhand = to_handmodel(utils.crop(frame, pickhand_coords))
    frethand = to_handmodel(utils.crop(frame, frethand_coords))
    # Uncrop hands to larger frame
    pickhand = utils.reposition(pickhand, pickhand_coords)
    frethand = utils.reposition(frethand, frethand_coords)
    return (pickhand, frethand)

# Get digit wireframe of hand and present to user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Locate pickhand and frethand
    #(guitar_coords, pickhand_coords, frethand_coords) = \
    #    locate.guitar_and_hands(frame)
    pickhand_coords = [[100, 100, 250, 250]]
    frethand_coords = [[400, 100, 550, 250]]
    
    # Get the hands (list of fingertips) in the frame
    (pickhand, frethand) = \
        get_hands(frame, pickhand_coords, frethand_coords)
    
    # Add overlay circles on fingers for display
    frame = reduce(utils.addcircle, [frame] + pickhand)
    frame = reduce(utils.addcircle, [frame] + frethand)
    
    return frame

if __name__ == '__main__':
    utils.capture(main)
