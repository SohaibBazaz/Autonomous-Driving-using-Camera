#!/usr/bin/env python3
import os
import time
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Electrics
from PIL import Image
import csv
from PIL import Image
import csv
import cv2
import numpy as np
import torch
import pandas as pd

scenario_map = {
    "smallgrid": {
        "mycar": {"position": (0, 0, 0), "rotation": (0, 0, 0, 1)},
        "aicar": {"position": (0, -150, 0), "rotation": (0, 0, 0, 1)},
        "aicar2": {"position": (10, -60, 0), "rotation": (0, 0, 0, 1)},
    },
    "gridmap_v2": {
        "mycar": {"position": (0, 0, 100), "rotation": (0, 0, 0, 1)},
        "aicar": {"position": (0, -10, 100), "rotation": (0, 0, 0, 1)},
    },
    "italy": {
        "mycar": {"position": (-353.763, 1169.096, 168.698), "rotation": (0, 0, 0.7071, 0.7071)},
        "aicar": {"position": (-361.763, 1169.096, 168.698), "rotation": (0, 0, 0.7071, 0.7071)},
    },
    "east_coast_usa": {
        "mycar": {"position": (598.1989, -127.7278, 53.4487), "rotation": (0, 0, 1, 0)},
        "aicar": {"position": (600.1989, -121.7278, 53.4487), "rotation": (0, 0, 1, 0)},
    },
}


def start_scenario(bng, map_name, scenario_name):
    scenario_config = scenario_map.get(map_name)
    if not scenario_config:
        raise ValueError(f"No configuration found for map: {map_name}")

    scenario = Scenario(map_name, scenario_name)

    mycar_pos = scenario_config["mycar"]["position"]
    mycar_rot = scenario_config["mycar"]["rotation"]

    mycar = Vehicle('mycar', model='etk800', licence='ITS', colour='Blue')
    scenario.add_vehicle(mycar, pos=mycar_pos, rot_quat=mycar_rot)
    scenario.make(bng)

    bng.load_scenario(scenario)
    bng.start_scenario()
    bng.resume()

    time.sleep(2)
    print("Vehicle connected!")
    return mycar


def image_and_steering(cameras, vehicle, max_frames, frame_count, base_save_folder, electrics):
    print("🎬 Starting image + steering capture...")

    # Dedicated folder for processed images is no longer needed, 
    # as the images are saved directly into the camera folders.
    
    # CSV file for steering labels
    csv_path = os.path.join(base_save_folder, "steering_labels.csv")
    csv_exists = os.path.isfile(csv_path)

    # Create separate folders for each camera
    camera_folders = {}
    for cam in cameras:
        cam_folder = os.path.join(base_save_folder, cam.name)
        os.makedirs(cam_folder, exist_ok=True)
        camera_folders[cam.name] = cam_folder

    camera_names = [cam.name for cam in cameras]

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write CSV header if it doesn't exist
        if not csv_exists:
            header = [f"{name}_frame" for name in camera_names]
            header.append("steering")
            writer.writerow(header)

        while frame_count < max_frames:
            image_filenames = []
            all_cameras_polled = True
            
            # --- Get Steering from electrics sensor (Polled first to ensure data sync) ---
            
            try:
                # Poll all sensors (cameras and electrics) attached to the vehicle simultaneously.
                vehicle.sensors.poll() 
            except AttributeError:
                try:
                    vehicle.update_state() 
                except:
                    pass
            
            # Retrieve electrics data
            steering = electrics.get("steering", None)


            # --- Camera Poll, Preprocess, and Save ---
            for cam in cameras:
                data = cam.poll()
                if data is None or "colour" not in data:
                    # ... (error handling remains the same) ...
                    continue

                # 1. Get the raw image data. 
                # Ensure it is a NumPy array for preprocessing and slicing.
                raw_img_data = np.array(data["colour"]) # <--- CRITICAL FIX: Explicit conversion to NumPy array
                
                # 2. Apply the full preprocessing pipeline
                # The image is now cropped (66x200), resized, and in YUV format.
                processed_img_np = preprocess(raw_img_data) # <--- THIS NOW RECEIVES A NUMPY ARRAY
                
                # 3. Convert the processed NumPy array to a PIL Image for saving
                processed_img_pil = Image.fromarray(processed_img_np)

                # Save the PROCESSED image in the camera-specific folder
                img_name = f"frame_{frame_count:03d}.png"
                img_path = os.path.join(camera_folders[cam.name], img_name)
                processed_img_pil.save(img_path)
                image_filenames.append(img_name)

            if not all_cameras_polled:
                time.sleep(0.05)
                continue

            # Write CSV row (linking the processed images to the steering angle)
            row_data = image_filenames + [steering]
            writer.writerow(row_data)

            frame_count += 1
            print(f"✅ Saved frame {frame_count}/{max_frames} | steering={steering}")

            time.sleep(0.05)

    print("🎉 Image + steering capture complete!")
    return frame_count

# --- Helper definitions for AI Model Preprocessing ---

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    NVIDIA model crops 60 pixels from top and 25 from bottom.
    """
    # image[y_start:y_end, x_start:x_end, :]
    return image[60:-25, :, :]

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

def resize(image):
    """
    Resize the image to the input shape used by the network model (66x200)
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB (BeamNG default) to YUV (NVIDIA model required input)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    # NOTE: These functions require OpenCV (cv2) which must be imported!
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

# ----------------------------------------------------

# --- Data Reading and Augmentation Helpers ---

def load_image(data_dir, image_file):
    """
    Loads the image from file.
    ASSUMPTION: The image file is ALREADY preprocessed (66x200, YUV format)
    and saved as a standard PNG/image file.
    """
    img_path = os.path.join(data_dir, image_file)
    # We load the image as-is. OpenCV will load the PNG (which contains YUV data)
    # as a BGR image, which we immediately convert to RGB for the augmentation pipeline.
    # NOTE: Since the data stored is YUV, but saved/loaded as standard channels (BGR/RGB), 
    # we proceed without a YUV-to-RGB conversion here, relying on the augmentation 
    # functions (like brightness) to work on the channel data structure.
    image = cv2.imread(img_path) 
    if image is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
        
    # Standard OpenCV loading converts to BGR. We convert to RGB for consistency 
    # with the rest of the augmentation functions.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly chooses among center, left, or right camera image.
    Loads the ALREADY PRE-PROCESSED image.
    """
    choice = np.random.choice(3)
    if choice == 0: # Left camera chosen
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1: # Right camera chosen
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle # Center camera chosen


# --- Augmentation Functions (REMAIN UNCHANGED) ---

def random_flip(image, steering_angle):
    # ... (same as before) ...
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    # ... (same as before) ...
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    """
    Randomly adjusts the brightness (V channel in HSV).
    NOTE: Works by converting to HSV, adjusting V, then converting back to RGB.
    This works on the channel structure of the loaded image, even if the content
    is conceptually YUV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Applies combined data augmentation steps to an ALREADY PRE-PROCESSED image.
    """
    # 1. Choose image (loads the pre-processed image) and adjust steering
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle) 
    
    # 2. Random flip
    image, steering_angle = random_flip(image, steering_angle)
    
    # 3. Random translation
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    
    # 4. Random brightness
    image = random_brightness(image)
    
    return image, steering_angle


class CustomDataset(torch.utils.data.Dataset):

    # ... (__init__ remains the same) ...
    def __init__(self, csv_file_path, base_image_dir, transform = None):
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

        # ------------------------------------------------
        # 1. Dynamic Augmentation (Randomly applied to the pre-processed image)
        if np.random.rand() < 0.6:
            # Augment, which includes choosing an image (center/left/right) and applying flip/translate/brightness
            # The 'augment' function handles loading the pre-processed image.
            image, steering_angle = augment(self.base_image_dir, center_file, left_file, right_file, steering_angle)
        else:
            # Load the pre-processed center image only
            image = load_image(self.base_image_dir, center_file)
            
        # ------------------------------------------------
        # 2. NO PREPROCESSING STEP HERE (It was done offline)
        
        # ------------------------------------------------
        # 3. Final PyTorch Transformations (e.g., Normalization)
        if self.transform is not None:
            # Converts numpy array (H, W, C) to Tensor (C, H, W) and normalizes
            image = self.transform(image) 
            
        steering_angle = torch.tensor(steering_angle, dtype=torch.float32)
            
        return image, steering_angle

    def __len__(self):
        return len(self.examples)

def main():
    set_up_simple_logging()

    # -----------------------------
    # BeamNG Path and Connection
    # -----------------------------
    bng = BeamNGpy('localhost', 25252, home=r'E:\\BeamNG.tech.v0.36.4.0', user=r'E:\\BeamNG.tech.v0.36.4.0')
    
    # Open/launch BeamNG
    bng.open(launch=True)

    map_name = 'east_coast_usa'
    vehicle = start_scenario(bng, map_name, 'gps_test')

    # -----------------------------
    # Setup and Stabilization
    # -----------------------------
    try:
        try:
            bng.settings.set_deterministic(60)
        except Exception:
            pass
        try:
            bng.ui.hide_hud()
        except Exception:
            pass

        print("⏱️  Waiting for physics to stabilize...")
        time.sleep(3)

        # -----------------------------
        # Camera Sensors (Front, Left, Right)
        # -----------------------------
        
        # Base resolution and update settings
        res = (640, 480)
        update_time = 0.05
        fov = 70
        
        # 1. Front Camera (Tuned Dashcam)
        front_cam = Camera(
            name='front_cam', bng=bng, vehicle=vehicle,
            pos=(0.0, -0.5, 1.8),       # Center, Forward (Negative Y), High
            dir=(0, -1, 0),             # Looking Forward (Negative Y-axis)
            field_of_view_y=fov, resolution=res, requested_update_time=update_time,
            is_using_shared_memory=True, is_render_annotations=False, is_render_depth=False, near_far_planes=(0.1, 1000)
        )
        
        # 2. Left Camera (User Specified Position)
        # Assuming: X=Lateral, Y=Forward, Z=Vertical
        # Direction should be along the X-axis (Lateral)
        left_cam = Camera(
            name='left_cam', bng=bng, vehicle=vehicle,
            pos=(1.0, -1.0, 1.0),       # X=1.0, Y=-1.0, Z=1.0 (User specified)
            dir=(0, -1, 0),              # Looking Left (Positive X-axis)
            field_of_view_y=fov, resolution=res, requested_update_time=update_time,
            is_using_shared_memory=True, is_render_annotations=False, is_render_depth=False, near_far_planes=(0.1, 1000)
        )
        
        # 3. Right Camera (User Specified Position)
        # Direction should be opposite the X-axis (Lateral)
        right_cam = Camera(
            name='right_cam', bng=bng, vehicle=vehicle,
            pos=(-1.0, -1.0, 1.0),      # X=-1.0, Y=-1.0, Z=1.0 (User specified)
            dir=(0, -1, 0),             # Looking Right (Negative X-axis)
            field_of_view_y=fov, resolution=res, requested_update_time=update_time,
            is_using_shared_memory=True, is_render_annotations=False, is_render_depth=False, near_far_planes=(0.1, 1000)
        )
        
        # Store all cameras in a list for easier iteration and cleanup
        cameras = [front_cam, left_cam, right_cam]

        # -----------------------------
        # Image Saving Setup
        # -----------------------------
        save_folder = "E:\\Beamng images"
        os.makedirs(save_folder, exist_ok=True)
        max_frames = 100
        frame_count = 0
        electrics = Electrics()
        vehicle.attach_sensor('electrics', electrics)
        # Capture images (pass the list of cameras)
        frame_count = image_and_steering(cameras, vehicle, max_frames, frame_count, save_folder, electrics)
        print(f"Captured {frame_count} frames.")

    finally:
        # Cleanup cameras and close BeamNG even if errors occur
        try:
            # Remove all cameras
            for cam in cameras:
                cam.remove()
        except Exception:
            pass
        try:
            bng.close()
        except Exception:
            pass
        print("🔒 BeamNG closed.")
        


if __name__ == "__main__":
    main()
