# Devsoc_induction
Final assignment of the devsoc AI/ML vertical
Simple neural network that is trained to recognise digits from the mnist dataset.
It 2 hidden layers, the first hidden layer uses ReLU activation and the second layer uses tanh activation.
The output layer is softmax, for which the categorical cross entropy loss is evaluated.
This network will optimise accuracy by minimising the algebraic difference between the predicted matrix and the true matrix, using gradient descent and the upating the weights and biases in accordsance with the gradients and the learning rate.
The learning rate of 1.5 was found to give the best results and an accuracy of roughly 85% was reached in 500 epochs.
The weights and biases were initilaised to random values between -0.5 and 0.5.
