import numpy as np

# 1. Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Sum Function (Forward Pass logic)
def calculate_layer(inputs, weights, bias):
    return sigmoid(np.dot(inputs, weights) + bias)

# 3. Training/Backpropagation Function
def train(X, y, epochs=10000, lr=0.1):
    # Initialize weights and biases
    input_nodes = 2
    hidden_nodes = 2
    output_nodes = 1

    # He initialization-style random weights
    w_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
    b_hidden = np.zeros((1, hidden_nodes))
    w_out = np.random.uniform(size=(hidden_nodes, output_nodes))
    b_out = np.zeros((1, output_nodes))

    for i in range(epochs):
        # --- Forward Prop ---
        hidden_layer_input = np.dot(X, w_hidden) + b_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_layer_input = np.dot(hidden_layer_output, w_out) + b_out
        output = sigmoid(final_layer_input)

        # --- Backpropagation ---
        # Error at output
        error_out = y - output
        d_output = error_out * sigmoid_derivative(output)

        # Error at hidden layer
        error_hidden = d_output.dot(w_out.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        # Updating weights and biases
        w_out += hidden_layer_output.T.dot(d_output) * lr
        b_out += np.sum(d_output, axis=0, keepdims=True) * lr
        w_hidden += X.T.dot(d_hidden) * lr
        b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr
        
    return w_hidden, b_hidden, w_out, b_out

# 4. Prediction Function
def predict(X, w_h, b_h, w_o, b_o):
    hidden = calculate_layer(X, w_h, b_h)
    return calculate_layer(hidden, w_o, b_o)

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Train
w_h, b_h, w_o, b_o = train(X, y)

# Output
print("Predictions after training:")
predictions = predict(X, w_h, b_h, w_o, b_o)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f}")