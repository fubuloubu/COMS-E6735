#!/usr/bin/python
import sys
import utils
import locate
errorstring = '''
handmodel.py: Stats
 Picking Hand Fingertip Locations: {}
Fretting Hand Fingertip Locations: {}
handmodel.py: Stats
'''

# Verify results of each frame
def verify(filename):
    def results_fun(frame_data):
        results  = 50*min(len(frame_data[0]),5) / 5.0
        results -= 50*(max(len(frame_data[0]),5) - 5)
        results += 50*min(len(frame_data[1]),5) / 5.0
        results -= 50*(max(len(frame_data[1]),5) - 5)
        results = max(results,0)
        return results
    utils.getresults(filename, errorstring, results_fun)

# Intepret hand area of frame into hand model
# Handmodel: (x, y) pairs of fingertip points
def to_handmodel(hand_crop, direction='up'):
    handcontour = utils.skindetect(hand_crop)
    hand = utils.cut(hand_crop, handcontour)
    handskeleton = utils.skeletonize(hand)
    fingerlines = utils.linedetect(handskeleton)
    if direction == 'dn':
        handmodel = map(lambda l: [l[2], l[3]], fingerlines)
    else:
        handmodel = map(lambda l: [l[0], l[1]], fingerlines)
    if len(handmodel) > 4:
        handmodel = utils.cluster(handmodel, \
                value=lambda p: (p[0])**2 + (p[1])**2, \
                K=4)
        combine=lambda p1, p2: [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
        handmodel = map(lambda l: reduce(combine, l), handmodel)
    return handmodel

# Get the hand models for both hands
def get_hands(frame, pickhand_coords, frethand_coords):
    # Interpret hand areas to get hand models for both hands
    pickhand = to_handmodel(utils.crop(frame, pickhand_coords), 'dn')
    frethand = to_handmodel(utils.crop(frame, frethand_coords), 'up')
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
    scale = frame.shape[1]/1920.0
    pickhand_coords = [[int(scale*i) for i in [   0,  520,  800, 1080]]] #DEBUG
    frethand_coords = [[int(scale*i) for i in [ 800,  300, 1800,  900]]] #DEBUG

    # Get the hands (list of fingertips) in the frame
    (pickhand, frethand) = \
        get_hands(frame, pickhand_coords, frethand_coords)
    
    # Print for debugging and testing purposes
    sys.stderr.write( errorstring.format(progname, pickhand, frethand) ) 
    
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
