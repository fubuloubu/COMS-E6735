#!/usr/bin/python
import utils
import locate

# Transform guitar model to a wireframe for display
def to_wireframe(guitarmodel):
    wireframe = []
    return wireframe

# Get the hand models for both hands
def get_guitar(frame):
    guitarmodel = []
    return guitarmodel

# Get wireframe of guitar POI and present to user
def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Get the guitar's POI in the frame
    guitar = get_guitar(locate.guitar(frame))
    # Add wireframes for guitar's POI for display
    frame = utils.addshape(frame, to_wireframe(guitar))
    return frame

if __name__ == '__main__':
    utils.capture(main)
