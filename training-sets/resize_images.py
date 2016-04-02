#!/usr/bin/python
import sys, os, glob, math, argparse
from PIL import Image, ImageOps

def resize_image(image_file, new_size):
    image = Image.open(image_file)
    orig_size = image.size
    # compute largest aspect ratio
    aspect_ratio = max(orig_size[0]/float(new_size[0]), float(orig_size[1]/new_size[1]))
    # Scale size while retaining aspect ratio
    corr_size = (int(orig_size[0]/aspect_ratio), int(orig_size[1]/aspect_ratio))
    new_image = image.resize( corr_size )
    # Fill in remaining space with black
    color = 0 # Black
    new_image = ImageOps.expand(new_image, new_size, color)
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
        print "ERROR: No directory named " + directory + " exists!"
        sys.exit(1)
    else:
        jpg_files = glob.glob(directory + "/*.jpg")
        if len(jpg_files) == 0:
            print "ERROR: No *.jpg files in directory " + directory
            sys.exit(1)
        else:
            size = (args.W, args.H)
            map(lambda im: resize_image(im, size), jpg_files)
