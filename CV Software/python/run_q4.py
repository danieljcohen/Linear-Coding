import numpy as np
import scipy.io
from nn import *
from tqdm import tqdm
import skimage
# ---------------- settings -------------------

#scheduler:
update_after_num_epochs = 15
learning_rate_factor = .5

# network params:
batch_size = 15
hidden_sizes = [512,256,64]
#layer names:
layer_names = ["layer1","layer2","layer3","output"]


#training params:
max_iters = 250
learning_rate = 5e-3

# ---------------------------------------------


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

train_x_extra = []
train_y_extra = []
#adding some transformations:
#print(train_x.shape)
for crop,y in zip(train_x,train_y):
    import matplotlib.pyplot as plt
    temp = skimage.morphology.erosion(crop.reshape(32,32), skimage.morphology.disk(1))
    temp = skimage.morphology.dilation(temp, skimage.morphology.disk(2))
    train_x_extra.append(temp.flatten())
    train_y_extra.append(y)
train_y_extra = np.array(train_y_extra)
train_x_extra = np.array(train_x_extra)
train_x = np.vstack((train_x, train_x_extra))
train_y = np.vstack((train_y, train_y_extra))
#print(train_x.shape)

batches = get_random_batches(train_x,train_y,batch_size)

batch_num = len(batches)

params = {}

# initialize layers here
for i in range(len(layer_names)):
    if i ==0:
        initialize_weights(train_x.shape[1],hidden_sizes[i],params,layer_names[i])
    elif i == len(layer_names)-1:
        initialize_weights(hidden_sizes[-1],train_y.shape[1],params,layer_names[i])
    else:
        initialize_weights(hidden_sizes[i-1],hidden_sizes[i],params,layer_names[i])
import copy
w_init = copy.deepcopy(params["Wlayer1"])

train_acc_list = []
valid_acc_list = []
train_loss_list = []
learning_rate_list = []
# with default settings, you should get accuracy > 80%
progress_bar = tqdm(range(1,max_iters+1))
best_val_acc = 0
num_epochs_best = 0
for itr in progress_bar:
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        #run forward
        for i in range(len(layer_names)):
            if i ==0:
                probs = forward(xb,params,layer_names[i])
            elif i == len(layer_names)-1:
                probs = forward(probs,params,name=layer_names[i],activation=softmax)
            else:
                probs = forward(probs,params,name=layer_names[i])
        #calculate loss
        loss,acc = compute_loss_and_acc(yb,probs)
        delta1 = probs - yb
        #processing accuracys
        total_acc += acc
        total_loss += loss
        #run backward
        for i in reversed(range(len(layer_names))):
            if i ==len(layer_names)-1:
                delta2 = backwards(delta1,params,layer_names[i],linear_deriv)
            else:
                delta2 = backwards(delta2,params,layer_names[i],sigmoid_deriv)
        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name] -= learning_rate*v
    #training acc final
    total_acc = total_acc/batch_num
    total_loss/=batch_num
    train_acc_list.append(total_acc)
    train_loss_list.append(total_loss)
    learning_rate_list.append(learning_rate)
    # run on validation set
    valid_acc = 0

    for i in range(len(layer_names)):
        if i ==0:
            probs = forward(valid_x,params,layer_names[i])
        elif i == len(layer_names)-1:
            probs = forward(probs,params,name=layer_names[i],activation=softmax)
        else:
            probs = forward(probs,params,name=layer_names[i])
    # loss
    _,valid_acc = compute_loss_and_acc(valid_y,probs)

    #learning rate scheduler:
    if best_val_acc<valid_acc:
        best_val_acc=valid_acc
        num_epochs_best=0
    elif num_epochs_best>update_after_num_epochs:
        learning_rate*=learning_rate_factor
        num_epochs_best=0
    else:
        num_epochs_best+=1

    valid_acc_list.append(valid_acc)
    # be sure to add loss and accuracy to epoch totals
    progress_bar.set_description("itr: {:02d}, loss: {:.2f}, acc : {:.2f}, val acc: {:.4f}, lr: {:.2e}".format(itr,total_loss,total_acc,valid_acc,learning_rate))


#plot data:
import matplotlib
import matplotlib.pyplot as plt
if True:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    
    # Plot accuracy
    axes[0].plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Training Accuracy', marker='o')
    axes[0].plot(range(1, len(valid_acc_list) + 1), valid_acc_list, label='Validation Accuracy', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Training Loss', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].set_title('Cross-Entropy Loss over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    # Plot lr
    axes[2].plot(range(1, len(learning_rate_list) + 1), learning_rate_list, label='learning_rate', marker='o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate over Epochs')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0,.006)
    #axes[2].autoscale_view()

    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        print(crop)
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layer_names.pickle', 'wb') as handle:
    pickle.dump(layer_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q4.3
#matplotlib.use('agg')
from mpl_toolkits.axes_grid1 import ImageGrid

im = []
im.append(params["Wlayer1"])

fig = plt.figure(1, (5, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                 axes_pad=0.2,  # pad between axes in inch.
                 )
grid[1].imshow(im[0])  
grid[0].imshow(w_init)
plt.show()


# Q4.4
for i in range(len(layer_names)):
    if i ==0:
        probs = forward(valid_x,params,layer_names[i])
    elif i == len(layer_names)-1:
        probs = forward(probs,params,name=layer_names[i],activation=softmax)
    else:
        probs = forward(probs,params,name=layer_names[i])
max_ind = np.argmax(probs,axis=1)
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for i in range(len(max_ind)):
    confusion_matrix[max_ind[i],np.argmax(valid_y[i])] +=1


import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
