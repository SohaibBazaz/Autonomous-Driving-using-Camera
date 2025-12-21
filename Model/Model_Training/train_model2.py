import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms 
from model2 import DriverNetDeep

# --- Constants (Must match your saved image dimensions) ---
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

# --- Training Configuration and Paths ---
BASE_DATA_DIR = "D:\\Downloads\\beamng_pics" 
DATA_CSV_FILE_PATH = os.path.join(BASE_DATA_DIR, 'steering_labels.csv')

SAVE_DIR = './output2/'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'ai_driver_cnn_deep.pth') 

MAX_STEERING_ANGLE_DEGREES = 360.0

# Hyperparameters
batch_size = 128
num_epochs = 100
LEARNING_RATE = 0.002
validation_split = 0.25
shuffle_dataset = True
random_seed = 42
num_workers = 0  # Keep at 0 for debugging

# --- Utility Function for Normalization ---
def normalize(image_np):
    """Normalizes image pixel values from [0, 255] to [-1.0, 1.0]."""
    return (image_np / 127.5) - 1.0

# ====================================================================
# --- 1. Data Augmentation and Loading Helpers ---

def load_image(data_dir, image_file):
    """Loads the pre-processed image from file using OpenCV."""
    img_path = os.path.join(data_dir, image_file)
    image = cv2.imread(img_path) 
    if image is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_image(data_dir, center, left, right, steering_angle):
    """Randomly chooses among center, left, or right camera image."""
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """Randomly flips the image horizontally and negates the steering angle."""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """Randomly translates the image horizontally/vertically and adjusts steering."""
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    """Randomly adjusts the brightness (V channel in HSV)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """Applies combined data augmentation steps to an ALREADY PRE-PROCESSED image."""
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle) 
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle

# ====================================================================
# --- 2. Custom Dataset Class ---

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file_path, base_image_dir, transform = None):
        """Initializes the dataset by loading the CSV log."""
        self.csv_file_path = csv_file_path
        self.base_image_dir = base_image_dir
        self.transform = transform
        self.examples = []

        data_df = pd.read_csv(self.csv_file_path)
        
        for index, row in data_df.iterrows():
            center_img_file = os.path.join('front_cam', row['front_cam_frame'])
            left_img_file = os.path.join('left_cam', row['left_cam_frame'])
            right_img_file = os.path.join('right_cam', row['right_cam_frame'])
            steering_angle = float(row['steering'])
            
            self.examples.append((center_img_file, left_img_file, right_img_file, steering_angle))


    def __getitem__(self, index):
        """Retrieves one data point and applies dynamic augmentation ONLY."""
        center_file, left_file, right_file, steering_angle = self.examples[index]

        ##if np.random.rand() < 0.6:
            ##image, steering_angle = augment(self.base_image_dir, center_file, left_file, right_file, steering_angle)
        
        image = load_image(self.base_image_dir, center_file)

        steering_angle = steering_angle / MAX_STEERING_ANGLE_DEGREES

        if self.transform is not None:
            image = self.transform(image) 
            
        steering_angle = torch.tensor(steering_angle, dtype=torch.float32)
            
        return image, steering_angle

    def __len__(self):
        return len(self.examples)

# ====================================================================
# --- 3. Training Utilities ---

def toDevice(data, device):
    """Converts a tensor to float and moves it to the specified device."""
    return data.float().to(device)


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    
    os.makedirs(SAVE_DIR, exist_ok=True) 
    
    epoch_number, train_losses, val_losses = [], [], []
    best_loss = 10000.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Training started on device: {device}")
    print(f"Number of training batches: {len(dataloaders['train'])}")
    print(f"Number of validation batches: {len(dataloaders['val'])}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_number.append(epoch)
        
        # Training Phase
        model.train()
        train_loss = 0.0
        
        batch_count = 0
        print("Starting training phase...")
        for inputs, labels in dataloaders['train']:
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Processing training batch {batch_count}/{len(dataloaders['train'])}")
            
            inputs = toDevice(inputs, device)
            labels = toDevice(labels, device)

            optimizer.zero_grad()
            
            out = model(inputs)
            loss = criterion(out, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training phase complete. Processed {batch_count} batches.")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        batch_count = 0
        print("Starting validation phase...")
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processing validation batch {batch_count}/{len(dataloaders['val'])}")
                
                inputs = toDevice(inputs, device)
                labels = toDevice(labels, device)
                
                out = model(inputs)
                loss = criterion(out, labels.unsqueeze(1))
                val_loss += loss.item()

        print(f"Validation phase complete. Processed {batch_count} batches.")

        # Average and Log Losses
        train_loss = train_loss / len(dataloaders['train'])
        val_loss = val_loss / len(dataloaders['val'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model.")
            torch.save(model, MODEL_SAVE_PATH)
            best_loss = val_loss

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Loss: {:4f}'.format(best_loss))

    # Save final log
    log_frame = pd.DataFrame(columns = ["Epoch", "Train Loss", "Test Loss"])
    log_frame["Epoch"] = epoch_number
    log_frame["Train Loss"] = train_losses
    log_frame["Test Loss"] = val_losses
    log_frame.to_csv(os.path.join(SAVE_DIR, "log.csv"), index = False)

    return model

# ====================================================================
# --- 4. Main Execution ---

if __name__ == "__main__":
    
    # 1. Check Data Availability
    print("=" * 60)
    print("STEP 1: Checking data files...")
    print("=" * 60)
    
    if not os.path.exists(DATA_CSV_FILE_PATH):
        print(f"❌ Error: Data CSV not found at {DATA_CSV_FILE_PATH}.")
        sys.exit(1)
    else:
        print(f"✅ CSV file found: {DATA_CSV_FILE_PATH}")

    if not os.path.exists(BASE_DATA_DIR):
        print(f"❌ Error: Base directory not found at {BASE_DATA_DIR}.")
        sys.exit(1)
    else:
        print(f"✅ Base directory found: {BASE_DATA_DIR}")

    print("\n" + "=" * 60)
    print("STEP 2: Loading dataset...")
    print("=" * 60)
    
    # 2. Data Loading Setup
    transformations = transforms.Compose([transforms.Lambda(normalize)])

    dataset = CustomDataset(DATA_CSV_FILE_PATH, BASE_DATA_DIR, transformations)
    
    dataset_size = len(dataset)
    print(f"Total dataset size: {dataset_size}")
    
    if dataset_size == 0:
        print("❌ Error: Dataset is empty!")
        sys.exit(1)
    
    # Test loading one sample
    print("\nTesting dataset loading...")
    try:
        test_img, test_label = dataset[0]
        print(f"✅ Successfully loaded one sample")
        print(f"   Image shape: {test_img.shape}")
        print(f"   Label: {test_label}")
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("STEP 3: Creating data loaders...")
    print("=" * 60)
    
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers)
    
    data_loader_dict = {
        'train': train_loader,
        'val': validation_loader
    }
    
    print(f"✅ Train loader batches: {len(train_loader)}")
    print(f"✅ Validation loader batches: {len(validation_loader)}")
    
    print("\n" + "=" * 60)
    print("STEP 4: Initializing model...")
    print("=" * 60)
    
    # 3. Model, Loss, and Optimizer Setup
    model_ft = DriverNetDeep()
    
    total_params = sum(p.numel() for p in model_ft.parameters())
    print(f"✅ Model: DriverNetDeep")
    print(f"   Total parameters: {total_params:,}")
    
    criterion = nn.MSELoss() 
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    print(f"✅ Using Adam optimizer with learning rate: {LEARNING_RATE}")

    print("\n" + "=" * 60)
    print("STEP 5: Starting training...")
    print("=" * 60)

    # 4. Start Training
    model_ft = train_model(model_ft, data_loader_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    print("\n" + "=" * 60)
    print("Training complete! Check the 'output' folder for the saved model and log.")
    print("=" * 60)
    
    # 5. Plotting Loss
    try:
        import matplotlib.pyplot as plt
        log_frame = pd.read_csv(os.path.join(SAVE_DIR, "log.csv"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(log_frame["Epoch"], log_frame["Train Loss"], label="Training Loss")
        plt.plot(log_frame["Epoch"], log_frame["Test Loss"], label="Validation Loss")
        
        plt.title('MSE Loss Vs Epoch (DriverNetDeep)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, 'loss_plot.png'))
        print("Loss plot saved to output/loss_plot.png")
    except ImportError:
        print("Matplotlib not found. Skipping loss plotting.")