import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

# commented-out lines are config for large-scale experiment

def nearest_sizes(size, sizes):
    nearest = 0
    snearest = 1

    for i, s in enumerate(sizes[:-1]):
        if s < size:
            nearest = s
            snearest = sizes[i+1]
    if nearest == 0:
        return 1, 1, 1
    return nearest, snearest, (snearest-size)/(snearest-nearest)

b=100
#b=1000

sizes = [n for n in range(1,33)]
#sizes = [1,2,4,8,16,24,32,40,48,56,64]

WIDTH=1000
HEIGHT=1000
#MAX_SIZE=64
#MAX_EPOCHS=1500
MAX_SIZE=32
MAX_EPOCHS=500

DIR="deep_double/resnet/"
#DIR="deep_double/resnet_0/"

# vertical scaling function
def get_epoch_coord(p):
    q = (b**p-1)/(b-1)
    return q

def inv_epoch_coord(q):
    p = math.log(q*(b-1)+1, b)
    return p

# plot approximately logarithmically
def make_arr(datas, sizes, max_size=MAX_SIZE, max_epochs=MAX_EPOCHS, width=WIDTH, height=HEIGHT):
    arr = np.zeros((width, height))

    for i in range(width):
        size, ssize, interp = nearest_sizes((i/width)*max_size, sizes)
        for y in range(height):
            coord = math.floor(get_epoch_coord(y/height)*max_epochs)
            arr[i,y] = interp*datas[size][coord] + (1-interp)*datas[ssize][coord]
    
    return arr

raw_datas = {}

for size in sizes:
    with open(DIR+f"resnet18_{size}", "rb") as f:
        raw_datas[size] = pickle.load(f)

test_err = {k:list(map(lambda e: e["test_loss"], v)) for k,v in raw_datas.items()}
train_err = {k:list(map(lambda e: e["train_loss"], v)) for k,v in raw_datas.items()}

test_img = make_arr(test_err, sizes)
train_img = make_arr(train_err, sizes)

fig = plt.figure(figsize=[10, 4])
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

im1 = ax1.imshow(np.rot90(train_img))

#ylabels = [1500,1000,100,10,1]
ylabels=[500,100,10,1]
#xlabels = [1, 16, 32, 48, 64]
xlabels=[1, 8, 16, 24, 32]

ax1.set_title("Train loss")
ax1.set_yticks([(1-inv_epoch_coord(n/MAX_EPOCHS))*HEIGHT for n in ylabels], labels=ylabels)
ax1.set_xticks([n/MAX_SIZE*WIDTH for n in xlabels], labels=xlabels)
ax1.set_ylabel("Epoch")
ax1.set_xlabel("Model size (k)")
fig.colorbar(im1)

im2 = ax2.imshow(np.rot90(test_img))

ax2.set_title("Test loss")
ax2.set_yticks([(1-inv_epoch_coord(n/MAX_EPOCHS))*HEIGHT for n in ylabels], labels=ylabels)
ax2.set_xticks([n/MAX_SIZE*WIDTH for n in xlabels], labels=xlabels)
ax2.set_ylabel("Epoch")
ax2.set_xlabel("Model size (k)")
fig.colorbar(im2)

plt.show()