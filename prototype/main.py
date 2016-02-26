#!/usr/bin/python
import argparse
import utils

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outfile", required=True,
               help="path to output video file")
    ap.add_argument("-f", "--fps", type=int, default=20,
               help="FPS of output video")
    ap.add_argument("-c", "--codec", type=str, default="MJPG",
               help="codec of output video")
    args = vars(ap.parse_args())
    
    # Display and Save video
    utils.save(args["outfile"], args["codec"], args["fps"], utils.mirror)
