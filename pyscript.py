import pickle
import numpy as np


with open('Data/MNIST-120k', 'rb') as file:
    mydico = pickle.load(file)
data = mydico['data']
label = mydico['labels']

# Convert labels to a numpy array of uint8
labelint = [int(i) for i in label]
nplabel = np.array(labelint, dtype=np.uint8)
print(nplabel[0:20])

#create a binary file to store the data in a binary format
data.tofile('Data/data.bin')

#creat a binary file to store labels.
nplabel.tofile('Data/label.bin')


# Save the shape of the data for later use in C++
with open('Data/data_shape.txt', 'w') as f:
    f.write(','.join(map(str, data.shape)) + '\n')
    f.write(str(len(labelint)))