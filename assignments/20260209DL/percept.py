import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_layer(inputs, weights, bias):
    sum_ = np.dot(inputs, weights)
    sum_bias = sum_ + bias
    return sum_bias, sigmoid(sum_bias)

def calculate_loss(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

def calculate_delta(y_true, y_pred):
    return  y_pred - y_true

def train(X, y, layers, epochs=200, lr=1):
    n_layers = len(layers)
    i_bias = 1
    i_weight = 0
    histories = []
    
    for i in range(epochs):
        # Forward
        current_input = X
        layers_output = [X]
        
        for layer in layers:
            _, current_input = calculate_layer(current_input, layer[i_weight], layer[i_bias])
            layers_output.append(current_input)

        # Backpropagation
        d_layers_output = []
        
        # Loss function
        # loss = calculate_loss(y, layers_output[-1])
        delta = calculate_delta(y, layers_output[-1])
        print(f"Epoch {i+1}/{epochs}, Loss: {calculate_loss(y, layers_output[-1]):.4f}")
        d_layers_output.append(delta)
        
        # Hidden layer propagation
        for j in range(n_layers - 1, 0, -1):
            weight = delta.dot(layers[j][i_weight].T)
            delta = weight * sigmoid_derivative(layers_output[j])
            d_layers_output.append(delta)
        d_layers_output.reverse()
        
        # Update weight
        for j in range(n_layers):
            layer_input = layers_output[j]
            delta_j = d_layers_output[j]
            
            layers[j][i_weight] -= layer_input.T.dot(delta_j) * lr
            layers[j][i_bias] -= np.sum(delta_j, axis=0, keepdims=True) * lr
        
        histories.append([ [w.copy(), b.copy()] for w, b in layers ])
        
    return histories, layers

def predict(X, layers):
    i_bias = 1
    i_weight = 0
    input_x = X
    for (j, layer) in enumerate(layers):
        _, input_x = calculate_layer(input_x, layer[i_weight], layer[i_bias])
    return input_x

n_input_neurons = 2
n_hidden_neurons = 2
n_output_neurons = 1

layers = []

w_hidden = np.random.uniform(size=(n_input_neurons, n_hidden_neurons))
b_hidden = np.zeros((1, n_hidden_neurons))
layers.append([w_hidden, b_hidden])

w_out = np.random.uniform(size=(n_hidden_neurons, n_output_neurons))
b_out = np.zeros((1, n_output_neurons))
layers.append([w_out, b_out])

# AND
# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([[0], [0], [0], [1]])

# OR
# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([[0], [1], [1], [1]])

# # XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

histories, model = train(X, y, layers)

print(model)

print("Predictions after training:")
predictions = predict(X, model)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f} to {1 if predictions[i][0] > 0.5 else 0}")