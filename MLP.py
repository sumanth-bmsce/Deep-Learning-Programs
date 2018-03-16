from __future__ import division, print_function, absolute_import
import tflearn
import tflearn.datasets.mnist as mnist

X,Y,testX,testY = mnist.load_data(one_hot = True)
# Input is X = 55000*784
# testX = 10000*784
# No of classes = 10
input_layer = tflearn.input_data(shape=[None,784])
# 64 is no of hidden neurons in that layer
dense1 = tflearn.fully_connected(input_layer,64,activation = 'tanh', regularizer = 'L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')
# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),show_metric=True, run_id="dense_model")
model.predict(testX)

