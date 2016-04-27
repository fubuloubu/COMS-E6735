#!/usr/bin/python
import sys
import utils
import locate
progname = 'handmodel.py'

# Intepret hand area of frame into hand model
# Handmodel: (x, y) pairs of fingertip points
def to_handmodel(hand_image):
    handcontour = utils.skindetect(hand_image)
    handmodel = utils.convexhull(handcontour)
    return handmodel

# Get the hand models for both hands
def get_hands(frame, pickhand_coords, frethand_coords):
    # Interpret hand areas to get hand models for both hands
    pickhand = to_handmodel(utils.crop(frame, pickhand_coords))
    frethand = to_handmodel(utils.crop(frame, frethand_coords))
    # Uncrop hands to larger frame
    pickhand = map(lambda p: utils.reposition(p, pickhand_coords), pickhand)
    frethand = map(lambda p: utils.reposition(p, frethand_coords), frethand)
    return (pickhand, frethand)

# Get digit wireframe of hand and present to user
def main(frame):
    # Locate pickhand and frethand
    #TODO: Get this working
    #(guitar_coords, pickhand_coords, frethand_coords) = \
    #    locate.guitar_and_hands(frame)
    pickhand_coords = [[   0,  520,  800, 1080]] #DEBUG
    frethand_coords = [[ 800,  300, 1800,  900]] #DEBUG

    # Get the hands (list of fingertips) in the frame
    (pickhand, frethand) = \
        get_hands(frame, pickhand_coords, frethand_coords)
    
    # Print for debugging and testing purposes
    sys.stderr.write('''
{0}: Stats
 Picking Hand Fingertip Locations: {1}
Fretting Hand Fingertip Locations: {2}
{0}: Stats
'''.format(progname, pickhand, frethand) ) 
    
    # Add overlay circles on fingers for display
    frame = utils.addrectangle(frame, pickhand_coords)
    frame = utils.addtext(frame, "Pickhand has found {} fingertips".format(len(pickhand)), location="ur")
    frame = reduce(utils.addcircle, [frame] + pickhand)
    frame = utils.addrectangle(frame, frethand_coords)
    frame = utils.addtext(frame, "Frethand has found {} fingertips".format(len(frethand)), location="ul")
    frame = reduce(utils.addcircle, [frame] + frethand)
    return frame

if __name__ == '__main__':
    if len(sys.argv) == 2:
        utils.test(sys.argv[1], main)
    else:
        utils.capture(main)
