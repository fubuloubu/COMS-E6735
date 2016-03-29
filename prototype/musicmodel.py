#!/usr/bin/python
import utils
import handmodel
import guitarmodel

# Using hand and guitar models, determine if a note event occurs
# and return that note, or a non-event ('-')
def get_note(frame):
    # Get the guitar's POI in the frame
    guitar = guitarmodel.get_guitar(frame)
    # Get the hand models for both hands
    (picking_hand, fretting_hand) = handmodel.get_hands(frame)
    
    # Determine if a note was plucked
    validnote = False
    if validnote:
        return note
    else:
        return ['-','-','-','-','-','-']

# Construct tablature from note occurances
#TODO: Involve Beats per minute and Frames Per Second
#      to intelligently reduce number of notes displayed
global tablature 
tablature = []
def add_to_tablature(frame):
    global tablature 
    tablature.append(get_note(frame))
    return tablature

# Get tablature text (last N occurances)
N = 20
def get_tab(frame):
    # Get current tablature
    current_tab = add_to_tablature(frame)
    # Shorten if longer than N
    if len(current_tab) > N:
        current_tab = current_tab[-N:]
    # Convert to text and add to screen
    current_tab = '\n'.join([''.join(i) for i in zip(*current_tab)]) 
    return "Tablature:\n" + current_tab

def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Add tablature to frame in LR corner
    frame = utils.addtext(frame, get_tab(frame), 'lr')
    return frame

if __name__ == '__main__':
    utils.capture(main)
