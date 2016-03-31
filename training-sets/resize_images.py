#!/usr/bin/python
import sys, os
import glob
from PIL import Image, ImageOps

def resize_image(image_file, scale):
    image = Image.open(image_file)
    new_image = image.resize( [int(scale * s) for s in image.size] )
    new_image.save(image_file)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "ERROR: Please pass a folder name to this script!"
        sys.exit(1)
    else:
        directory = sys.argv[1]
        # Check if folder containing JPG files
        if not os.path.isdir(directory):
            print "ERROR: No directory named " + directory + " exists!"
            sys.exit(1)
        else:
            jpg_files = glob.glob(directory + "/*.jpg")
            if len(jpg_files) == 0:
                print "ERROR: No *.jpg files in directory " + directory
                sys.exit(1)
            elif len(sys.argv) < 3:
                print "ERROR: Please pass a scale factor to this script!"
                sys.exit(1)
            else:
                try:
                    scale = float(sys.argv[2])
                except:
                    print "ERROR: Scale Factor doesn't work (scale=" + sys.argv[2] + ")"
                    sys.exit(1)
                map(lambda im: resize_image(im, scale), jpg_files)

