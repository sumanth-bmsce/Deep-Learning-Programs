import tflearn

# 2. Logical OR operator / the data
OR = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_truth = [[0.], [1.], [1.], [1.]]

# 3. Building our neural network/layers of functions 
neural_net = tflearn.input_data(shape=[None, 2])
neural_net = tflearn.fully_connected(neural_net, 1, activation='sigmoid')
neural_net = tflearn.regression(neural_net, optimizer='sgd', learning_rate=2, loss='mean_square')

# 4. Train the neural network / Epochs
model = tflearn.DNN(neural_net)
model.fit(OR, Y_truth, n_epoch=2000, snapshot_epoch=False)

# 5. Testing final prediction
print("Testing OR operator")
print("0 or 0:", model.predict([[0., 0.]]))
print("0 or 1:", model.predict([[0., 1.]]))
print("1 or 0:", model.predict([[1., 0.]]))
print("1 or 1:", model.predict([[1., 1.]]))

