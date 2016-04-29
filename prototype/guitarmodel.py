#!/usr/bin/python
import utils
import sys, math
import locate
progname = 'guitarmodel.py'

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
    if len(lines) < 8:
        return guitar
    # Convert to line model
    lines = map(lambda l: utils.tolinemodel(l), lines)
    # Get average angle of all lines
    avg_angle = reduce(lambda avg, lm: avg + lm["angle"], [0] + lines)/len(lines)
    origin_line_y = lambda x: math.tan(avg_angle)*x # y = mx+b, m = tan(theta)
    origin = utils.tolinemodel([-10, origin_line_y(-10), 10, origin_line_y(10)])
    delta_angle = 2 #degs
    # Find the strongly connected lines
    # Filter down to all lines within delta angle of the average
    lines_near_avg = filter(lambda lm: abs(lm["angle"] - avg_angle) < delta_angle, lines)
    # If the lines that fit the average are not the majority, exit
    if len(lines_near_avg) < len(lines)/2:
        return
    # If there aren't enough lines to support this operation, return
    if len(lines_near_avg) < NUM_STRINGS:
        return
    # Search for strings and append/update string locations
    string_lines = utils.cluster(lines_near_avg, \
            distance=lambda lm1, lm2: lm1["origin"] - lm2["origin"], \
            combine=utils.combine_linemodel, \
            origin=origin, K=NUM_STRINGS)
    if string_lines is None:
        return guitar
    guitar["locations"]["strings"] = map(lambda lm: lm["line"], string_lines)
    # Search for frets and append/update string locations
    guitar["available"] = True
    return guitar

# Get wireframe of guitar POI and present to user
def main(frame):
    # Get the guitar's POI in the frame
    guitar = get_guitar(frame)#locate.guitar(frame))
    # Return overlaid guitar attributes on frame
    return display(frame, guitar)

if __name__ == '__main__':
    utils.capture(main)
