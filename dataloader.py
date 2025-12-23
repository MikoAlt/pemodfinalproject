import numpy as np
from pathlib import Path
from preprocessor import preprocess_image

def load_data_from_folder(folder_path, class_to_idx):
    """
    Load all images from a specific folder (train or test) given class mappings.
    """
    X = []
    y = []
    path = Path(folder_path)
    
    # Iterate through known classes to maintain consistent ordering/indexing
    for class_name, idx in class_to_idx.items():
        class_dir = path / class_name
        if not class_dir.exists():
            continue
            
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                processed_vec, _ = preprocess_image(img_path)
                if processed_vec is not None:
                    X.append(processed_vec)
                    y.append(idx)
                    
    return np.array(X), np.array(y)

class DataLoader:
    def __init__(self, X_train, y_train, X_test, y_test, class_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        
    @classmethod
    def from_folders(cls, dataset_base_path):
        """
        Initialize DataLoader by reading from 'train' and 'test' subdirectories.
        """
        base_path = Path(dataset_base_path)
        train_dir = base_path / "train"
        test_dir = base_path / "test"
        
        # Determine classes from train_dir
        class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        X_train, y_train = load_data_from_folder(train_dir, class_to_idx)
        X_test, y_test = load_data_from_folder(test_dir, class_to_idx)
        
        return cls(X_train, y_train, X_test, y_test, class_names)

    def get_train_batch(self, batch_size=None):
        if batch_size is None:
            return self.X_train, self.y_train
        
        batch_indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        return self.X_train[batch_indices], self.y_train[batch_indices]

    def get_test_data(self):
        return self.X_test, self.y_test
