#!/usr/bin/env python3
import os
import time
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera
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
        "mycar": {"position": (598.1989509371833, -127.72784042535113, 53.44879727053376), "rotation": (0, 0, 1, 0)},
        "aicar": {"position": (600.1989509371833, -121.72784042535113, 53.44879727053376), "rotation": (0, 0, 1, 0)},
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


def image_and_steering(cam, vehicle, max_frames, frame_count, save_folder):
    print("🎬 Starting image + steering capture...")

    # CSV file for steering labels
    csv_path = os.path.join(save_folder, "steering_labels.csv")
    csv_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write header if first time
        if not csv_exists:
            writer.writerow(["frame", "steering"])

        while frame_count < max_frames:
            time.sleep(0.05)

            # ---- Camera Poll ----
            data = cam.poll()
            if data is None or "colour" not in data:
                continue

            img = data["colour"]

            if not hasattr(img, "save"):
                img = Image.fromarray(img)

            # Save image
            img_path = os.path.join(save_folder, f"frame_{frame_count:03d}.png")
            img.save(img_path)

            # ---- Vehicle Poll ----
            state = vehicle.poll()  # Full vehicle state
            steering = None
            if state and "electrics" in state and "steering" in state["electrics"]:
                steering = state["electrics"]["steering"]
                print(f"Steering angle: {steering}")
            # Save steering to CSV
            print("Saving steering data...")
            writer.writerow([f"frame_{frame_count:03d}.png", steering])

            frame_count += 1
            print(f"✅ Saved frame {frame_count}/{max_frames} | steering={steering}")

    return frame_count


def main():
    set_up_simple_logging()

    # -----------------------------
    # BeamNG Path and Connection
    # -----------------------------
    bng = BeamNGpy('localhost', 25252, home=r'D:\\Downloads\\BeamNG\\BeamNG', user=r'D:\\Downloads\\BeamNG\\userfolder')
    # Open/launch BeamNG
    bng.open(launch=True)

    map_name = 'east_coast_usa'
    # start_scenario will create scenario, add vehicle, load and start it
    vehicle = start_scenario(bng, map_name, 'gps_test')

    # -----------------------------
    # Keep the original logic but fix incorrect calls / variables
    # -----------------------------
    try:
        # Set deterministic physics (60 Hz) if supported
        try:
            bng.settings.set_deterministic(60)
        except Exception:
            # settings may not exist depending on beamngpy version; ignore if not available
            pass

        # Hide HUD if supported
        try:
            bng.ui.hide_hud()
        except Exception:
            pass

        # Wait for physics to stabilize
        print("⏱️  Waiting for physics to stabilize...")
        time.sleep(3)

        # -----------------------------
        # Camera Sensor
        # -----------------------------
        cam = Camera(
            name='front_cam',
            bng=bng,
            vehicle=vehicle,
            pos=(-5, 0, 1),               # Behind and above vehicle
            dir=(1, 0, 0),                # Forward direction
            field_of_view_y=70,
            is_using_shared_memory=True,
            is_render_annotations=False,
            is_render_depth=False,
            near_far_planes=(0.1, 1000),
            resolution=(640, 480),
            requested_update_time=0.05
        )

        # -----------------------------
        # Image Saving Setup
        # -----------------------------
        save_folder = "C:\\Users\\sohai\\OneDrive\\Desktop\\BEAMNG\\camera_images"
        os.makedirs(save_folder, exist_ok=True)
        max_frames = 100
        frame_count = 0

        # Capture images
        frame_count = image_from_camera(cam, max_frames, frame_count, save_folder)
        print(f"Captured {frame_count} frames.")

    finally:
        # Cleanup camera and close BeamNG even if errors occur
        try:
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
