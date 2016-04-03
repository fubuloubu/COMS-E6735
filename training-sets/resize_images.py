#!/usr/bin/python
import sys, os, glob, math, argparse
from PIL import Image, ImageOps

def resize_image(image_file, new_size):
    image = Image.open(image_file)
    orig_size = image.size
    # compute largest aspect ratio
    aspect_ratio = max(orig_size[0]/float(new_size[0]), orig_size[1]/float(new_size[1]))
    # Scale size while retaining aspect ratio
    corr_size = (int(round(orig_size[0]/aspect_ratio)), int(round(orig_size[1]/aspect_ratio)))
    image = image.resize( corr_size, Image.ANTIALIAS )
    # Create image with black background
    mode = image.mode
    if len(mode) == 1:  # L, 1
        new_background = (0)  
    if len(mode) == 3:  # RGB
        new_background = (0, 0, 0)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (0, 0, 0, 0) 
    new_image = Image.new(mode, new_size, new_background)
    x1 = (new_size[0] - corr_size[0])/2
    y1 = (new_size[1] - corr_size[1])/2
    x2 = x1 + corr_size[0]
    y2 = y1 + corr_size[1]
    new_image.paste(image, (x1,y1,x2,y2))
    new_image.save(image_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize all jpg files in specified folder to specified size')
    parser.add_argument('directory', metavar='DIRECTORY', nargs=1,
            help='A directory containing 1 or more *.jpg files')
    parser.add_argument('-W', metavar='w', required=True, type=int,
            help='Desired width to set image sizes to')
    parser.add_argument('-H', metavar='h', required=True, type=int,
            help='Desired height to set image sizes to')
    args = parser.parse_args()
    directory = args.directory[0]
    if not os.path.isdir(directory):
        print("ERROR: No directory named " + directory + " exists!")
        sys.exit(1)
    else:
        jpg_files = glob.glob(directory + "/*.jpg")
        if len(jpg_files) == 0:
            print("ERROR: No *.jpg files in directory " + directory)
            sys.exit(1)
        else:
            size = (args.W, args.H)
            map(lambda im: resize_image(im, size), jpg_files)
