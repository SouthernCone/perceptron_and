# perceptron_and.py

import numpy as np                                     # 1️⃣ NumPy for vector math and dot products 

# 2️⃣ Dataset: all input pairs for AND and their targets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])           # 2️⃣ 
y = np.array([0, 0, 0, 1])                              # 2️⃣ 

# 3️⃣ Hyperparameters
lr = 0.1                                                # 3️⃣ Learning rate controls step size 
epochs = 50                                             # 3️⃣ Number of full passes through the data 

# 4️⃣ Model parameters (initialized once on import)
weights = np.zeros(2)                                   # 4️⃣ Two weights, one per input feature 
bias = 0.0                                              # 4️⃣ Bias term allows shifting decision boundary 

def predict(x):
    """
    5️⃣ Perceptron prediction: step activation
    Returns 1 if (w·x + b) >= 0, else 0.
    """
    linear = np.dot(x, weights) + bias                 # 5️⃣ Weighted sum plus bias 
    return 1 if linear >= 0 else 0                      # 5️⃣ Heaviside step function 

def train():
    """
    6️⃣ Training function: updates weights & bias for 'epochs' iterations,
    logging every 5 epochs.
    """
    global weights, bias
    # 6.1️⃣ Print header for logs
    print("Epoch | Weights        | Bias  | Total Error")
    print("-------------------------------------------")
    # 6.2️⃣ Epoch loop
    for epoch in range(1, epochs + 1):                  # 6.2.1 Iterates from 1 to epochs 
        total_error = 0
        # 6.2.2 Loop over each training example
        for xi, target in zip(X, y):                   # 6.2.3 zip pairs inputs and targets 
            output = predict(xi)
            error = target - output                     # 6.2.4 Compute error (delta) 
            # 6.2.5 Perceptron weight/bias update rule
            weights += lr * error * xi                  # Δw = η · error · x_i 
            bias += lr * error                          # Δb = η · error   
            total_error += abs(error)
        # 6.3️⃣ Log only every 5 epochs to reduce clutter
        if epoch % 5 == 0:
            # Using f-string for formatted output 
            print(f"{epoch:>5} | {weights.tolist()} | {bias:5.2f} | {total_error}")

def evaluate():
    """7️⃣ After training, display final weights, bias, and predictions."""
    print("\nFinal weights:", weights, "Final bias:", bias)
    print("Predictions on AND inputs:")
    for xi in X:
        print(f"{xi} -> {predict(xi)}")

# 8️⃣ Only run training/evaluation when script is executed directly
if __name__ == "__main__":
    train()
    evaluate()