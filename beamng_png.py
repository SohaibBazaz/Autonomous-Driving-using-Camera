#!/usr/bin/env python3
import os
import time
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Electrics
from PIL import Image
import csv


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


def image_and_steering(cameras, vehicle, max_frames, frame_count, save_folder, electrics):
    print("🎬 Starting image + steering capture...")

    # CSV file for steering labels
    csv_path = os.path.join(save_folder, "steering_labels.csv")
    csv_exists = os.path.isfile(csv_path)

    camera_names = [cam.name for cam in cameras]
    
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        if not csv_exists:
            header = [f"{name}_frame" for name in camera_names]
            header.append("steering")
            writer.writerow(header)

        while frame_count < max_frames:
            # ---- Camera Poll and Image Save ----
            image_filenames = []
            all_cameras_polled = True
            
            for cam in cameras:
                # Poll data from the specific camera
                data = cam.poll()
                if data is None or "colour" not in data:
                    print(f"⚠️ Warning: No data from {cam.name}. Skipping frame {frame_count} for {cam.name}.")
                    image_filenames.append("") 
                    all_cameras_polled = False
                    continue

                img = data["colour"]
                if not hasattr(img, "save"):
                    img = Image.fromarray(img)

                # Save image
                img_name = f"{cam.name}_frame_{frame_count:03d}.png"
                img_path = os.path.join(save_folder, img_name)
                img.save(img_path)
                image_filenames.append(img_name)
            
            # If any camera failed, skip data capture for this iteration
            if not all_cameras_polled:
                time.sleep(0.05)
                continue
            
            # ---- Vehicle State Update for Steering Data ----
            
            # Use the method known to update the vehicle's internal .state dictionary in older versions
            try:
                vehicle.update_state() 
            except AttributeError:
                # If update_state() also fails, fall back to reading the existing state without refreshing
                print("⚠️ Warning: update_state() failed. Using last known vehicle state.")
                pass
            from beamngpy.sensors import Electrics

# Attach electrics sensor
            vehicle.sensors.poll()
            elec = vehicle.sensors['electrics']
            steering = elec.get("steering", None)

            # Read steering data from the vehicle's internal state
            state = vehicle.sensors.poll()
            elec=vehicle.sensors["electrics"]
            steering=elec.get("steering",None)
            
            ##steering = None
            ##if state and "electrics" in state and "steering" in state["electrics"]:
             ##   steering = state["electrics"]["steering"]
            
            # Save steering to CSV
            row_data = image_filenames + [steering]
            writer.writerow(row_data)

            frame_count += 1 
            print(f"✅ Saved frame {frame_count}/{max_frames} | steering={steering}")
            
            time.sleep(0.05) 

    return frame_count


def main():
    set_up_simple_logging()

    # -----------------------------
    # BeamNG Path and Connection
    # -----------------------------
    bng = BeamNGpy('localhost', 25252, home=r'D:\\Downloads\\BeamNG\\BeamNG', user=r'D:\\Downloads\\BeamNG\\BeamN')
    
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
        save_folder = "C:\\Users\\sohai\\OneDrive\\Desktop\\BEAMNG\\camera_stream_images"
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
