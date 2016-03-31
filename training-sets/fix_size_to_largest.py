#!/usr/bin/python
import sys, os
import glob
from PIL import Image, ImageOps

def increase_image_size(image, filename, border_size):
    color = (245, 241, 222) # Beige
    new_image = ImageOps.expand(image, border_size, color)
    new_image.save(filename)

def main(file_list):
    images = map(Image.open, file_list)
    image_sizes = map(lambda image: image.size, images)
    max_idx, max_sz = max(enumerate(image_sizes), 
            key=lambda(sz): sz[1][0]*sz[1][1])
    border_sizes = map(lambda sz: ((max_sz[0]-sz[0])/2, (max_sz[1]-sz[1])/2), 
            image_sizes)
    for idx, image in enumerate(images):
        if idx == max_idx:
            continue
        increase_image_size(image, file_list[idx], border_sizes[idx])

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
            else:
                main(jpg_files)
