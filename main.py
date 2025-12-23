import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import numpy as np
import pickle
import time
import argparse
from dataloader import DataLoader
from vector_model import VectorModel
from matrix_model import MatrixModel
from cnn_model import CNNModel
from pathlib import Path

def train_torch_model(model, X_train, y_train, X_test, y_test, model_type, epochs=100, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Reshape if CNN
    if model_type == 'cnn':
        X_train_t = torch.FloatTensor(X_train).view(-1, 1, 40, 40)
        X_test_t = torch.FloatTensor(X_test).view(-1, 1, 40, 40)
    else:
        X_train_t = torch.FloatTensor(X_train)
        X_test_t = torch.FloatTensor(X_test)
        
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'test_loss': [], 'accuracy': [], 'test_accuracy': []}
    
    print(f"Starting training {model_type} for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Test eval
        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            outputs_test = model(X_test_t.to(device))
            loss_test = criterion(outputs_test, y_test_t.to(device))
            test_loss = loss_test.item()
            _, predicted_test = outputs_test.max(1)
            test_correct = predicted_test.eq(y_test_t.to(device)).sum().item()
        test_acc = test_correct / len(y_test)
        
        history['loss'].append(epoch_loss)
        history['test_loss'].append(test_loss)
        history['accuracy'].append(epoch_acc)
        history['test_accuracy'].append(test_acc)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Test Acc: {test_acc:.4f} - Time: {duration:.2f}s")
            
    return history

def train_vector_model(X_train, y_train, X_test, y_test, class_names, epochs=50):
    input_size = 1600
    hidden_sizes = [2240, 640, 320]
    output_size = len(class_names)
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    model = VectorModel(layer_sizes, learning_rate=0.001, weight_decay=0.01, beta1=0.9, beta2=0.99)
    
    history = {'loss': [], 'test_loss': [], 'accuracy': [], 'test_accuracy': []}
    
    print(f"Starting training Vector Model for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss, correct = 0, 0
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for idx in indices:
            loss = model.train_step(X_train[idx], y_train[idx])
            epoch_loss += loss
            activations, _ = model.forward(X_train[idx])
            if np.argmax(activations[-1]) == y_train[idx]:
                correct += 1
                
        avg_loss = epoch_loss / len(X_train)
        acc = correct / len(X_train)
        
        # Test eval
        test_loss = 0
        test_correct = 0
        for i in range(len(X_test)):
            acts, _ = model.forward(X_test[i])
            y_pred = acts[-1]
            test_loss += -np.log(max(y_pred[y_test[i]], 1e-15))
            if np.argmax(y_pred) == y_test[i]:
                test_correct += 1
        avg_test_loss = test_loss / len(X_test)
        test_acc = test_correct / len(X_test)
        
        history['loss'].append(avg_loss)
        history['test_loss'].append(avg_test_loss)
        history['accuracy'].append(acc)
        history['test_accuracy'].append(test_acc)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - Test Acc: {test_acc:.4f} - Time: {time.time()-start_time:.2f}s")
            
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--model", type=str, choices=['vector', 'matrix', 'cnn'], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print("Loading data...")
    loader = DataLoader.from_folders("processed_dataset")
    class_names = loader.class_names
    
    with open("class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)

    if args.model == 'vector':
        model, history = train_vector_model(loader.X_train, loader.y_train, loader.X_test, loader.y_test, class_names, epochs=args.epochs)
        model.save_model("vector_model.pkl")
        with open("training_history.pkl", "wb") as f:
            pickle.dump(history, f)
    elif args.model == 'matrix':
        model = MatrixModel(1600, [2240, 640, 320], len(class_names))
        history = train_torch_model(model, loader.X_train, loader.y_train, loader.X_test, loader.y_test, 'matrix', epochs=args.epochs)
        torch.save(model.state_dict(), "matrix_model.pth")
        with open("matrix_history.pkl", "wb") as f:
            pickle.dump(history, f)
    elif args.model == 'cnn':
        model = CNNModel(1, len(class_names))
        # Batch size 16 for CNN due to size
        history = train_torch_model(model, loader.X_train, loader.y_train, loader.X_test, loader.y_test, 'cnn', epochs=args.epochs, batch_size=16)
        torch.save(model.state_dict(), "cnn_model.pth")
        with open("cnn_history.pkl", "wb") as f:
            pickle.dump(history, f)

    print(f"Training of {args.model} complete.")

if __name__ == "__main__":
    main()
