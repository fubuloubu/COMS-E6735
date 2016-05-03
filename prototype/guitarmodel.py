#!/usr/bin/python
import utils
import sys, math
import locate
errorstring = '''
guitarmodel.py: Stats
Guitar: {0}
guitarmodel.py: Stats
'''

# Initialize guitar data structure for tracking
guitar = {}
guitar["available"] = False
guitar["locations"] = {}
# Each of these keys has a list of line endpoint pairs
# denoting the locations of their respective attributes
guitar["locations"]["strings"] = [] #0-th string is the lowest
guitar["locations"]["frets"] = [] #0-th fret is the nut

# Transform guitar model to a wireframe for display
def display(frame, guitar):
    if guitar["available"]:
        text = ""
        for attr, lines in guitar["locations"].iteritems():
            text += "Number of " + attr + ": {}\n".format(len(lines))
            frame = reduce(utils.addline, [frame] + lines)
        frame = utils.addtext(frame, text, "ul")
    return frame

NUM_STRINGS = 4 # System parameter
NUM_FRETS = 24 # System parameter
# Update the guitar model and attribute locations
def get_guitar(frame):
    guitar["available"] = False
    # Get all lines in current frame that are long enough to be strings
    lines = utils.linedetect(frame, minLineLength=180)
    # Return if no lines
    if len(lines) < 1:
        return guitar
    
    # Convert to line model
    lines = map(lambda l: utils.tolinemodel(l), lines)
    
    # Get average angle of all lines
    string_angle = reduce(lambda avg, lm: avg + lm["angle"], [0] + lines)/len(lines)
    
    # Filter down to all lines within delta angle of the average
    delta_angle = 2 #degs
    lines_near_avg = filter(lambda lm: abs(lm["angle"] - string_angle) < delta_angle, lines)
    # If the lines that fit the average are not the majority, exit
    if len(lines_near_avg) < len(lines)/2:
        return guitar
    
    # Search for strings and append/update string locations
    string_lines = utils.cluster(lines_near_avg, \
            value=lambda lm: lm["origin"], K=NUM_STRINGS)
    string_lines = map(lambda sl: reduce(utils.combine_linemodel, sl), string_lines)
    if string_lines is None:
        return guitar
    
    guitar["available"] = True
    # Update string locations
    if len(guitar["locations"]["strings"]) < 4 or \
            len(string_lines) == NUM_STRINGS:
        # Update all strings
        guitar["locations"]["strings"] = map(lambda lm: lm["line"], string_lines)
    else:
        # TODO: Collectively move strings by average displacement
        guitar["locations"]["strings"] = map(lambda lm: lm["line"], string_lines)
    
    # Bounding box is x, y coordinates of both ends for first and last strings
    bounding_box = []
    bounding_box.append(guitar["locations"]["strings"][0][:2])
    bounding_box.append(guitar["locations"]["strings"][0][2:])
    bounding_box.append(guitar["locations"]["strings"][-1][:2])
    bounding_box.append(guitar["locations"]["strings"][-1][2:])
    print "Bounding box: {}".format(bounding_box)
    
    # Search for frets and append/update string locations
    lines = utils.linedetect(frame, minLineLength=20, maxLineGap=20)
    # Return if no lines
    if len(lines) < 1:
        return guitar
    
    # Convert to line model
    lines = map(lambda l: utils.tolinemodel(l), lines)
    
    # Frets are 90 degrees to the strings
    fret_angle = string_angle + 90
    # Filter down to all lines within twice the delta angle of the fret angle
    lines_near_avg = filter(lambda lm: abs(lm["angle"] - fret_angle) < 2*delta_angle, lines)
    # Filter down to lines that lie partially inside space made by strings
    print "Frets Found:\n{}".format(map(lambda lm: lm["line"], lines_near_avg))
    # TODO: Figure this out
    def box_check(lm): 
       return True

    lines_near_avg = filter(box_check, lines_near_avg)
    
    # If there aren't enough lines to support this operation, return
    if len(lines_near_avg) < 1:#NUM_FRETS/2:
        return guitar
    
    # Search for strings and append/update string locations
    fret_lines = utils.cluster(lines_near_avg, \
            value=lambda lm: lm["origin"], K=NUM_STRINGS)
    fret_lines = map(lambda sl: reduce(utils.combine_linemodel, sl), fret_lines)
    if fret_lines is None:
        return guitar
    guitar["locations"]["frets"] = map(lambda lm: lm["line"], fret_lines)
    guitar["available"] = True
    return guitar

# Get wireframe of guitar POI and present to user
def main(frame):
    # Get the guitar's POI in the frame
    guitar = get_guitar(frame)#locate.guitar(frame))
    # Write status
    sys.stderr.write(errorstring.format(guitar))
    # Return overlaid guitar attributes on frame
    return display(frame, guitar)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        utils.test(sys.argv[1], main)
    else:
        utils.capture(main)
