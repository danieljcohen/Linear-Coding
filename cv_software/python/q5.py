import numpy as np
import matplotlib.patches as mpatches
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage import io, color
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import io, filters
import matplotlib.pyplot as plt
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    #used tutorial for help:
    bboxes = []
    
    bw = color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(bw)
    final_bw = skimage.morphology.binary_opening((bw > thresh), square(1))
    bw = skimage.morphology.binary_closing((bw < thresh), square(9))
    #box = skimage.morphology.binary_opening(final_bw, square(1))
    label_image = label(bw)
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    min_area = (image.shape[0]*image.shape[1]/8000)
    max_area = (image.shape[0]*image.shape[1]/2)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= min_area and region.area < max_area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            bboxes.append((minr, minc, maxr, maxc))
    return bboxes, final_bw