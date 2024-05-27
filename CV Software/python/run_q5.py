import os
import numpy as np
import matplotlib
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q5 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    #plt.imshow(im1)
    #plt.show()
    bboxes, bw = findLetters(im1)

    if False:
        plt.imshow(bw,cmap='gray')
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    
    #sort vertically:
    bboxes_vert = sorted(bboxes, key=lambda x: abs(x[0]+x[2])/2)
    ordered_boxxes = []
    temp = []
    prev_cent = abs(bboxes_vert[0][0]+bboxes_vert[0][2])/2
    for box in bboxes_vert:
        cent = abs(box[0]+box[2])/2
        if abs(prev_cent-cent)<abs(max(box[2]-box[0],box[3]-box[1])):
            temp.append(box)
        else:
            ordered_boxxes.append(temp)
            temp = [box]
            prev_cent = cent
    if not temp == []:
        ordered_boxxes.append(temp)
    #print(ordered_boxxes)
    #sort each ordered_vert horizontally:
    for i in range(len(ordered_boxxes)):
        ordered_boxxes[i] = sorted(ordered_boxxes[i], key=lambda x: x[1])
    rem_list = []
    for i in range(len(ordered_boxxes)):
        prev_col_r = 0
        prev_col_l = 0
        for j in range(len(ordered_boxxes[i])):
            if prev_col_r>ordered_boxxes[i][j][3] and prev_col_l<ordered_boxxes[i][j][1]:
                rem_list.append((i,j))
            elif prev_col_r<ordered_boxxes[i][j][3] and prev_col_l>ordered_boxxes[i][j][1]:
                rem_list.append((i,j-1))
            else:
                prev_col_r = ordered_boxxes[i][j][3]
                prev_col_l = ordered_boxxes[i][j][1]
    for i,j in rem_list:
        ordered_boxxes[i].remove(ordered_boxxes[i][j])


    #adding spaces and returns:
    space_list = []
    newline_list = []
    cur_letter = 0
    for i in range(len(ordered_boxxes)):
        prev_right = ordered_boxxes[i][0][3]
        for j in range(len(ordered_boxxes[i])):
            if ordered_boxxes[i][j][1]-prev_right>(ordered_boxxes[i][j][3]-ordered_boxxes[i][j][1])//1.1:
                space_list.append(cur_letter)
            prev_right = ordered_boxxes[i][j][3]
            cur_letter+=1
        newline_list.append(cur_letter)
    newline_list = newline_list[0:-1]
    #print(newline_list)
    #print(space_list)


    color_names = [
    'red', 'blue', 'green', 'orange', 'purple',
    'cyan', 'magenta', 'yellow', 'black']
    index = 0
    #print("HERE")
    #print(ordered_boxxes)
    if False:
        plt.imshow(bw,cmap='gray')
        for bbox_vert in ordered_boxxes:
            for bbox in bbox_vert:
                minr, minc, maxr, maxc = bbox
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=color_names[index], linewidth=2)
                plt.gca().add_patch(rect)
                index = (index+1)%len(color_names)
        plt.show()
    

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    flattened_ord_boxxes = []
    for item in ordered_boxxes:
        flattened_ord_boxxes.extend(item)
    bw = bw.astype(int)
    #print(bw.shape)
    bw_thicker = skimage.morphology.binary_erosion(bw, skimage.morphology.disk(5))
    bw = skimage.morphology.binary_erosion(bw, skimage.morphology.disk(1))
    #print(bw.shape)
    bw = bw.astype(int)
    bw_thicker = bw_thicker.astype(int)
    subimages = []
    for bbox in flattened_ord_boxxes:
        # Compute center coordinates of the bounding box
        y_change = abs(bbox[0]- bbox[2])
        x_change = abs(bbox[1]- bbox[3])
        max_change = (max(x_change,y_change)//2)+10
        center_row = (bbox[0] + bbox[2]) // 2
        center_col = (bbox[1] + bbox[3]) // 2
        # Extract sub-image centered around the bounding box's center
        subimage = bw[max(center_row - y_change//2, 0):center_row + y_change//2, max(center_col - x_change//2, 0):center_col + x_change//2]
        max_length = int(max(subimage.shape)*1.2)
        pad_width = (((max_length - subimage.shape[0])//2, (max_length - subimage.shape[0])//2), ((max_length - subimage.shape[1])//2, (max_length - subimage.shape[1])//2))
        subimage = np.pad(subimage, pad_width, mode='constant', constant_values=1)
        
        # Resize the sub-image to the desired size
        subimage_resized = skimage.transform.resize(subimage, (32,32), anti_aliasing=True)
        #plt.imshow(subimage_resized)
        #plt.show()
        subimage_resized = np.array(subimage_resized).T
        subimage_resized = subimage_resized.flatten()
        subimage_resized = (subimage_resized/max(subimage_resized))

        if (np.sum(subimage_resized)>900):
            #print("before: ", np.sum(subimage_resized))
            subimage = bw_thicker[max(center_row - y_change//2, 0):center_row + y_change//2, max(center_col - x_change//2, 0):center_col + x_change//2]
            max_length = int(max(subimage.shape)*1.3)
            pad_width = (((max_length - subimage.shape[0])//2, (max_length - subimage.shape[0])//2), ((max_length - subimage.shape[1])//2, (max_length - subimage.shape[1])//2))
            subimage = np.pad(subimage, pad_width, mode='constant', constant_values=1)
            
            # Resize the sub-image to the desired size
            subimage_resized = skimage.transform.resize(subimage, (32,32), anti_aliasing=True)
            #plt.imshow(subimage_resized)
            #plt.show()
            subimage_resized = np.array(subimage_resized).T
            subimage_resized = subimage_resized.flatten()
            subimage_resized = (subimage_resized/max(subimage_resized))
            #print("after: ", np.sum(subimage_resized))


        # Add the resized sub-image to the list


        subimages.append(subimage_resized)
    subimages_array = np.array(subimages)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    if False:
        for im in subimages_array:
        #    print(im)
            plt.imshow(im.reshape(32,32).T)
            plt.show()
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    layer_names = pickle.load(open('layer_names.pickle','rb'))
    result = ""
    #print(params)
    for im in subimages_array:
        probs = None
        #print(im.shape)
        for i in range(len(layer_names)):
            if i ==0:
                probs = forward(im.reshape(1,im.shape[0]),params,layer_names[i])
            elif i == len(layer_names)-1:
                probs = forward(probs,params,name=layer_names[i],activation=softmax)
            else:
                probs = forward(probs,params,name=layer_names[i])
        #print(probs)
        probs_argmax = np.argmax(probs)
        result += letters[probs_argmax]
    

    #adding spaces and newlines in:
    index = 0
    final_result = ""
    while newline_list or space_list or index<len(result):
        if newline_list and newline_list[0]==index:
            final_result += "\n" + result[index]
            newline_list.remove(newline_list[0])
        elif space_list and space_list[0]==index:
            final_result += " " + result[index]
            space_list.remove(space_list[0])
        else:
            final_result+=result[index]
        index +=1

    #final processing (dealing with 0 and O)
    prev_char = final_result[0]
    for i in range(len(final_result)):
        if (final_result[i].isdigit() and prev_char.isalpha()) or (i < len(final_result)-1 and (final_result[i].isdigit() and final_result[i+1].isalpha())):
            if final_result[i] == "0":
                final_result = final_result[:i] + 'O' + final_result[i+1:]
            if final_result[i] == "5":
                final_result = final_result[:i] + "S" + final_result[i+1:]
            if final_result[i] == "8":
                final_result = final_result[:i] + "B" + final_result[i+1:]
        prev_char = final_result[i]
    
    print(final_result)
    print("\n")
    #for im in subimages_array:
    #    plt.imshow(im.reshape(32,32))
    #    plt.show()
    #break


