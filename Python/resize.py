from PIL import Image
from sys import argv

if len(argv) != 4:
    exit("usage: python resize.py n infile outfile")

n = int(argv[1])
infile = argv[2]
outfile = argv[3]

inimage = Image.open(infile)
width, heigh = inimage.size
outimage = inimage.resize((width * n, height * n))

outimage.save(outfile)

# To resize small images
# python resize.py n image.extension out.originalImageExtension
# n = *number of resize 
