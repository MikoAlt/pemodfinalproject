import random
from pathlib import Path
from preprocessor import preprocess_image, save_image, rotate_image, flip_image, add_noise

def process_and_save_dataset(input_dir, output_dir, split_ratio=0.8, seed=42):
    """
    Iterate through the input directory, process images, and save them (original + 4 augmented) 
    physically separated into 'train' and 'test' folders to multiply the dataset by 5.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create train and test directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        print(f"Processing class: {class_dir.name}")
        
        images = [f for f in class_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (test_dir / class_dir.name).mkdir(exist_ok=True)
        
        def process_and_augment(img_list, target_base_dir):
            target_class_path = target_base_dir / class_dir.name
            for img_path in img_list:
                _, processed_img = preprocess_image(img_path)
                
                if processed_img is not None:
                    orig_save_path = target_class_path / f"{img_path.stem}_orig.png"
                    save_image(processed_img, orig_save_path)
                    
                    aug1 = flip_image(processed_img, 1)
                    save_image(aug1, target_class_path / f"{img_path.stem}_aug1.png")
                    
                    angle = random.uniform(-20, 20)
                    aug2 = rotate_image(processed_img, angle)
                    save_image(aug2, target_class_path / f"{img_path.stem}_aug2.png")
                    
                    aug3 = add_noise(processed_img, sigma=random.uniform(0.01, 0.05))
                    save_image(aug3, target_class_path / f"{img_path.stem}_aug3.png")
                    
                    aug4 = flip_image(processed_img, 0)
                    aug4 = rotate_image(aug4, random.uniform(-15, 15))
                    save_image(aug4, target_class_path / f"{img_path.stem}_aug4.png")

        process_and_augment(train_images, train_dir)
        process_and_augment(test_images, test_dir)
                    
    print(f"Dataset processed and split into physical train/test folders at: {output_dir}")

if __name__ == "__main__":
    INPUT_DATASET = "dataset"
    OUTPUT_DATASET = "processed_dataset"
    process_and_save_dataset(INPUT_DATASET, OUTPUT_DATASET)
