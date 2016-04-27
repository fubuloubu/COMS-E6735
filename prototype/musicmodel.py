#!/usr/bin/python
import utils
import locate
import handmodel
import guitarmodel

# Using hand and guitar models, determine if a note event occurs
# and return that note, or a non-event ('-')
def get_note(picking_hand, fretting_hand, guitar):
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
    # Get the guitar's POI in the frame
    (guitar_coords, pickhand_coords, frethand_coords) = \
        locate.guitar_and_hands(frame)
    guitar = guitarmodel.get_guitar(guitar_coords)
    if not guitar:
        raise ValueError("No guitar!")
    # Get the hand models for both hands
    (pickhand, frethand) = \
        handmodel.get_hands(frame, pickhand_coords, frethand_coords)
    if not pickhand or not frethand:
        raise ValueError("No hand(s)!")
    
    global tablature 
    tablature.append(get_note(pickhand, frethand, guitar))
    return tablature

# Get tablature text (last N occurances)
N = 20
def get_tab(frame):
    # Get current tablature
    try:
        current_tab = add_to_tablature(frame)
        # Shorten if longer than N
        if len(current_tab) > N:
            current_tab = current_tab[-N:]
        # Convert to text and add to screen
        current_tab = '\n'.join([''.join(i) for i in zip(*current_tab)]) 
        return "Tablature:\n" + current_tab
    except ValueError:
        return "No Guitar and/or Hand(s)!"

def main(frame):
    # Flip the frame for ease of understanding
    frame = utils.mirror(frame)
    # Add tablature to frame in LR corner
    frame = utils.addtext(frame, get_tab(frame), 'lr')
    return frame

if __name__ == '__main__':
    utils.capture(main)
