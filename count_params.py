import torch
import numpy as np
import pickle
from matrix_model import MatrixModel
from cnn_model import CNNModel

def count_parameters(model):
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    elif hasattr(model, 'weights'):
        # Vector model manual count
        count = 0
        for w in model.weights:
            count += w.size
        for b in model.biases:
            count += b.size
        return count
    else:
        return 0

def main():
    try:
        with open("class_names.pkl", "rb") as f:
            class_names = pickle.load(f)
        num_classes = len(class_names)
    except FileNotFoundError:
        print("class_names.pkl not found, assuming 15 classes from observation")
        num_classes = 15

    print(f"Number of classes: {num_classes}")

    # 1. Vector Model (Using MatrixModel logic for calculation as they are identical architectures)
    # Actually, let's calculate manually to be consistent with the user's vector implementation
    input_size = 1600
    hidden_sizes = [2240, 640, 320]
    layer_sizes = [input_size] + hidden_sizes + [num_classes]
    
    vector_params = 0
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        vector_params += (n_in * n_out) + n_out
        
    print(f"Vector Model Parameters: {vector_params:,}")

    # 2. Matrix Model
    matrix_model = MatrixModel(input_size, hidden_sizes, num_classes)
    matrix_params = count_parameters(matrix_model)
    print(f"Matrix Model Parameters: {matrix_params:,}")

    # 3. CNN Model
    cnn_model = CNNModel(1, num_classes)
    cnn_params = count_parameters(cnn_model)
    print(f"CNN Model Parameters: {cnn_params:,}")

if __name__ == "__main__":
    main()
