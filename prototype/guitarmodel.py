#!/usr/bin/python
import utils
import locate
import math

# Transform guitar model to a wireframe for display
def display(frame, guitar):
    if guitar["available"]:
        text = ""
        for attr, lines in guitar["locations"].iteritems():
            text += "Number of " + attr + ": {}\n".format(len(lines))
            frame = reduce(utils.addline, [frame] + lines)
        frame = utils.addtext(frame, text, "ul")
    return frame

# Initialize guitar data structure for tracking
guitar = {}
guitar["available"] = False
guitar["locations"] = {}
# Each of these keys has a list of line endpoint pairs
# denoting the locations of their respective attributes
guitar["locations"]["strings"] = [] #0-th string is the lowest
guitar["locations"]["frets"] = [] #0-th fret is the nut

# Update the guitar model and attribute locations
def get_guitar(frame):
    guitar["available"] = False
    # Get all lines in current frame that could be strings
    lines = utils.linedetect(frame, minLineLength=180)
    if len(lines) < 8:
        return guitar
    # Find the strongly connected lines
    def tolinemodel(l):
        lm = {}
        lm["line"] = l
        lm["length"] = math.sqrt((l[1]-l[3])**2 + (l[0]-l[2])**2)
        if l[0] == l[2]:
            lm["slope"] = float('inf') # vertical line
            lm["intercept"] = float('inf') # y-intercept is undefined
            lm["angle"] = 90 # vertical line (in deg)
            lm["origin"] = l[0] # dist from origin for vertical line is simply x
        else:
            lm["slope"] = (l[3]-l[1])/float(l[2]-l[0]) # m = y2 -y1 / x2 - x1
            lm["intercept"] = l[1] - lm["slope"]*l[0] # b = y - mx
            lm["angle"] = math.atan(lm["slope"]) # theta = atan(m), ignore quadrants
            lm["origin"] = lm["intercept"]*math.cos(lm["angle"]) # shortest distance from origin
        return lm
    lines = map(lambda l: tolinemodel(l), lines)
    avg_angle = reduce(lambda avg, lm: avg + lm["angle"], [0] + lines)/len(lines)
    delta_angle = 2 #degs
    lines_near_avg = filter(lambda lm: abs(lm["angle"] - avg_angle) < delta_angle, lines)
    if len(lines_near_avg) < len(lines)/2:
        # If the lines that fit the average are not the majority, exit
        return guitar
    else:
        # Filter down on lines
        lines = lines_near_avg
    # Search for strings and append/update string locations
    def linemodel_combine(lm1, lm2):
        lm = {}
        # Average all this stuff
        lm["intercept"] = (lm1["intercept"] + lm2["intercept"])/2 # b = y - mx
        lm["origin"] = (lm1["origin"] + lm2["origin"])/2
        lm["angle"] = math.acos(lm["origin"]/lm["intercept"])
        lm["slope"] = math.tan("angle")
        # Make the longest line possible
        perp_ang = math.radians(90) - lm["angle"]
        def squash_line(l):
            d = l["origin"] - lm["origin"] # perpidicular distance to combined line
            # squash begin points of line onto combined ray
            begin = [l["line"][0] - d*math.cos(perp_ang), l["line"][1] - d*math.sin(perp_ang)]
            end = [l["line"][2] - d*math.cos(perp_ang), l["line"][3] - d*math.sin(perp_ang)]
            return [begin[0], begin[1], end[0], end[1]]
        length = lambda l: math.sqrt((l[3]-l[1])**2 + (l[2]-l[0])**2)
        lm["line"] = max(squash_line(lm1), squash_line(lm2), key=length)
        lm["length"] = (lm["line"][3]-lm["line"][1])/float(lm["line"][2]-lm["line"][0])
        return lm
    #strings = utils.1Dcluster(lines, valueFunction=lambda lm: lm["origin"], \
    #        combinationFunction=linemodel_combine,numGroups=4)
    guitar["locations"]["strings"] = map(lambda lm: lm["line"], lines)
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
