import tflearn
import random
import numpy as np

features = [[[1,1],[0,1]],[[2,2],[0,1]],[[6,6],[1,0]],[[7,8],[1,0]]]
'''features.append([[0, 0, 0, 0, 0], [0,1]])
features.append([[0, 0, 0, 0, 1], [0,1]])
features.append([[0, 0, 0, 1, 1], [0,1]])
features.append([[0, 0, 1, 1, 1], [0,1]])
features.append([[0, 1, 1, 1, 1], [0,1]])
features.append([[1, 1, 1, 1, 0], [0,1]])
features.append([[1, 1, 1, 0, 0], [0,1]])
features.append([[1, 1, 0, 0, 0], [0,1]])
features.append([[1, 0, 0, 0, 0], [0,1]])
features.append([[1, 0, 0, 1, 0], [0,1]])
features.append([[1, 0, 1, 1, 0], [0,1]])
features.append([[1, 1, 0, 1, 0], [0,1]])
features.append([[0, 1, 0, 1, 1], [0,1]])
features.append([[0, 0, 1, 0, 1], [0,1]])
features.append([[1, 0, 1, 1, 1], [1,0]])
features.append([[1, 1, 0, 1, 1], [1,0]])
features.append([[1, 0, 1, 0, 1], [1,0]])
features.append([[1, 0, 0, 0, 1], [1,0]])
features.append([[1, 1, 0, 0, 1], [1,0]])
features.append([[1, 1, 1, 0, 1], [1,0]])
features.append([[1, 1, 1, 1, 1], [1,0]])
features.append([[1, 0, 0, 1, 1], [1,0]])'''

# shuffle our features and turn into np.array
random.shuffle(features)
features = np.array(features)

# create train and test lists
train_x = list(features[:,0])
train_y = list(features[:,1])
net = tflearn.input_data(shape=[None, 2])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=500, batch_size=16, show_metric=True)

print(model.predict([[1,2]]))
print(model.predict([[5,6]]))
print(model.predict([[0,0]]))
'''print(model.predict([[1, 0, 1, 0, 1]]))'''
