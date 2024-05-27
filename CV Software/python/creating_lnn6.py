import numpy as np
import scipy.io
from nn import *
from tqdm import tqdm
import skimage

import torch
from torchsummary import summary
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("running on M series chip")
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    print("running on cuda")
else:
    DEVICE = 'cpu'
    print ("MPS or CUDA device not found. Running on CPU")
# ---------------- settings -------------------

config = {
    'batch_size': 64, # Increase this if your GPU can handle it
    'lr': 0.1,
    'epochs': 20, # 20 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
    'batch_size': 64,
    'lr_factor': .5,
    'patience': 2
}

# ---------------------------------------------




train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

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

batches = get_random_batches(train_x,train_y,config["batch_size"])

batch_num = len(batches)

class LinearNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)  # Input layer to hidden layer
        self.fc2 = nn.Linear(2048, 1024)  # Input layer to hidden layer
        self.fc3 = nn.Linear(1024, 512)  # Input layer to hidden layer
        self.fc4 = nn.Linear(512, output_size) # Hidden layer to output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x)) 
        x = torch.softmax(self.fc4(x))
        return x

model = LinearNN(train_x.shape[1],train_y.shape[1]).to(DEVICE)
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['patience'], threshold=1e-2, verbose=True, factor=config['lr_factor'])

def train(model, batches, optimizer, criterion):

    model.train()
    #t_y = torch.argmax(t_y, axis=1)
    # Progress Bar
    batch_bar   = tqdm(total=len(batches), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss  = 0

    for i, (images, labels) in enumerate(batches):

        optimizer.zero_grad() # Zero gradients

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct     += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss      += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            num_correct = num_correct,
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        loss.backward()
        optimizer.step()

        # TODO? Depending on your choice of scheduler,
        # You may want to call some schdulers inside the train function. What are these?

        batch_bar.update() # Update tqdm bar

    batch_bar.close() # You need this to close the tqdm bar

    acc         = 100 * num_correct / (config['batch_size']* len(batches))
    total_loss  = float(total_loss / len(batches))

    return acc, total_loss





def validate(model, images,labels, criterion):

    model.eval()

    num_correct = 0.0
    total_loss = 0.0


    # Move images to device
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Get model outputs
    outputs = model(images)
    loss = criterion(outputs, labels)

    num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
    total_loss += float(loss.item())


    acc = 100 * num_correct / (config['batch_size']* len(batches))
    total_loss = float(total_loss / len(batches))
    return acc, total_loss


gc.collect()

best_valacc = 0
for epoch in range(config['epochs']):

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_acc, train_loss = train(model, batches, optimizer, criterion)
    
    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epochs'],
        train_acc,
        train_loss,
        curr_lr))

    val_acc, val_loss = validate(model, valid_x,valid_y, criterion)
    scheduler.step(val_loss)
    print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

    # If you are using a scheduler in your train function within your iteration loop, you may want to log
    # your learning rate differently

    # #Save model in drive location if val_acc is better than best recorded val_acc
    if val_acc >= best_valacc:
      #path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
      print("Saving model")
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_acc': val_acc,
                  'epoch': epoch}, './checkpoint.pth')
      best_valacc = val_acc


def test(model,dataloader):

  model.eval()
  batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
  test_results = []

  for i, (images) in enumerate(dataloader):
      # TODO: Finish predicting on the test set.
      images = images.to(DEVICE)

      with torch.inference_mode():
        outputs = model(images)

      outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
      test_results.extend(outputs)

      batch_bar.update()

  batch_bar.close()
  return test_results

test_results = test(model, test_loader)







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
