import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataloader import DataLoader
from vector_model import VectorModel
from matrix_model import MatrixModel
from cnn_model import CNNModel
from pathlib import Path

def get_predictions(model, data, model_type, batch_size=32):
    """
    Unified function to get predictions and probabilities for different model types.
    """
    probs = []
    
    if model_type == 'vector':
        for i in range(len(data)):
            acts, _ = model.forward(data[i])
            probs.append(acts[-1])
        probs = np.array(probs)
        
    elif model_type in ['matrix', 'cnn']:
        model.eval()
        probs_list = []
        device = torch.device('cpu') # Use CPU for visualization to avoid CUDA OOM if that was the issue, or stick to what trained on?
        # Actually simplest to just keep it on CPU for visualization unless huge.
        
        # Convert all data to tensor once if it fits, or better batch conversion too?
        # Batch conversion is safer for memory.
        
        num_samples = len(data)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_data = data[start_idx:end_idx]
                
                if model_type == 'cnn':
                    # Reshape for CNN: (B, 1, 40, 40)
                    batch_tensor = torch.FloatTensor(batch_data).view(-1, 1, 40, 40)
                else:
                    batch_tensor = torch.FloatTensor(batch_data)
                
                outputs = model(batch_tensor)
                batch_probs = torch.softmax(outputs, dim=1).numpy()
                probs_list.append(batch_probs)
                
        if len(probs_list) > 0:
            probs = np.concatenate(probs_list, axis=0)
        else:
            probs = np.array([])

    preds = np.argmax(probs, axis=1)
    return probs, preds

def plot_model_performance(model_type):
    """
    Generates a 3x2 plot for a specific model:
    Row 1: Loss & Accuracy History
    Row 2: Train & Test Confusion Matrix
    Row 3: Train & Test ROC-AUC
    """
    # Configuration based on model type
    config = {
        'vector': {
            'model_path': 'vector_model.pkl',
            'history_path': 'training_history.pkl',
            'title': 'Vector Model'
        },
        'matrix': {
            'model_path': 'matrix_model.pth',
            'history_path': 'matrix_history.pkl',
            'title': 'Matrix Model (PyTorch)'
        },
        'cnn': {
            'model_path': 'cnn_model.pth',
            'history_path': 'cnn_history.pkl',
            'title': 'CNN Model'
        }
    }
    
    c = config.get(model_type)
    if not Path(c['model_path']).exists() or not Path(c['history_path']).exists():
        print(f"Skipping {model_type}: Model or history file not found.")
        return

    print(f"Generating visualizations for {c['title']}...")
    
    # 1. Load Class Names
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    num_classes = len(class_names)
    
    # 2. Load History
    with open(c['history_path'], "rb") as f:
        history = pickle.load(f)
    
    # 3. Load Model
    if model_type == 'vector':
        model = VectorModel.load_model(c['model_path'])
    elif model_type == 'matrix':
        model = MatrixModel(1600, [2240, 640, 320], num_classes)
        model.load_state_dict(torch.load(c['model_path'], map_location='cpu'))
    elif model_type == 'cnn':
        model = CNNModel(1, num_classes)
        model.load_state_dict(torch.load(c['model_path'], map_location='cpu'))
    
    # 4. Load Data
    loader = DataLoader.from_folders("processed_dataset")
    
    print("Computing metrics...")
    probs_train, preds_train = get_predictions(model, loader.X_train, model_type)
    probs_test, preds_test = get_predictions(model, loader.X_test, model_type)
    
    # 5. Plotting
    fig = plt.figure(figsize=(24, 20))
    sns.set_context("paper", font_scale=1.2)
    plt.suptitle(f"Performance Report: {c['title']}", fontsize=24)
    
    # Row 1: History
    ax1 = fig.add_subplot(3, 2, 1)
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'r-o', label='Train Loss')
    if 'test_loss' in history:
        ax1.plot(epochs, history['test_loss'], 'y-s', label='Test Loss')
    ax1.set_title('Training & Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(epochs, history['accuracy'], 'b-o', label='Train Acc')
    ax2.plot(epochs, history['test_accuracy'], 'g-s', label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.grid(True)
    ax2.legend()
    
    # Row 2: Confusion Matrices
    ax3 = fig.add_subplot(3, 2, 3)
    cm_train = confusion_matrix(loader.y_train, preds_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix (TRAIN)')
    
    ax4 = fig.add_subplot(3, 2, 4)
    cm_test = confusion_matrix(loader.y_test, preds_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=ax4)
    ax4.set_title('Confusion Matrix (TEST)')
    
    # Row 3: ROC Curves
    ax5 = fig.add_subplot(3, 2, 5)
    for i in range(len(class_names)):
        if np.sum(loader.y_train == i) > 0:
            fpr, tpr, _ = roc_curve(loader.y_train == i, probs_train[:, i])
            roc_auc = auc(fpr, tpr)
            ax5.plot(fpr, tpr, label=f'{class_names[i]} ({roc_auc:.2f})')
    ax5.plot([0, 1], [0, 1], 'k--')
    ax5.set_title('ROC-AUC (TRAIN)')
    ax5.legend(loc="lower right", fontsize='x-small', ncol=3)
    ax5.grid(True)
    
    ax6 = fig.add_subplot(3, 2, 6)
    for i in range(len(class_names)):
        if np.sum(loader.y_test == i) > 0:
            fpr, tpr, _ = roc_curve(loader.y_test == i, probs_test[:, i])
            roc_auc = auc(fpr, tpr)
            ax6.plot(fpr, tpr, label=f'{class_names[i]} ({roc_auc:.2f})')
    ax6.plot([0, 1], [0, 1], 'k--')
    ax6.set_title('ROC-AUC (TEST)')
    ax6.legend(loc="lower right", fontsize='x-small', ncol=3)
    ax6.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = f"results_{model_type}.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    plt.close()

if __name__ == "__main__":
    import sys
    models_to_plot = ['vector', 'matrix', 'cnn']
    if len(sys.argv) > 1:
        models_to_plot = [sys.argv[1]]
        
    for m in models_to_plot:
        plot_model_performance(m)
