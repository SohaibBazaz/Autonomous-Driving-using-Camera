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


def image_and_steering(cam, vehicle, electrics, max_frames, frame_count, save_folder):
    print("🎬 Starting image + steering capture...")

    csv_path = os.path.join(save_folder, "steering_labels.csv")
    csv_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if not csv_exists:
            writer.writerow(["frame", "steering"])

        while frame_count < max_frames:
            time.sleep(0.05)

            data = cam.poll()
            if data is None or "colour" not in data:
                continue

            img = data["colour"]
            if not hasattr(img, "save"):
                img = Image.fromarray(img)

            img_path = os.path.join(save_folder, f"frame_{frame_count:03d}.png")
            img.save(img_path)

            # Poll all sensors
            vehicle.sensors.poll()
            elec = vehicle.sensors['electrics']
            steering = elec.get("steering", None)

            writer.writerow([f"frame_{frame_count:03d}.png", steering])
            print(f"Saved frame {frame_count}/{max_frames} | steering={steering}")

            frame_count += 1

    return frame_count


def main():
    set_up_simple_logging()

    bng = BeamNGpy('localhost', 25252,
                   home=r'E:\\BeamNG.tech.v0.36.4.0',
                   user=r'E:\\BeamNG.tech.v0.36.4.0\userfolder')
    bng.open(launch=True)

    map_name = 'east_coast_usa'
    vehicle = start_scenario(bng, map_name, 'gps_test')

    try:
        try:
            bng.settings.set_deterministic(60)
        except:
            pass

        try:
            bng.ui.hide_hud()
        except:
            pass

        print("⏱️  Waiting for physics to stabilize...")
        time.sleep(3)

        cam = Camera(
            name='front_cam',
            bng=bng,
            vehicle=vehicle,
            # (X, Y, Z): X is lateral, Y is longitudinal (forward), Z is vertical
            # Adjusted Y to a negative value to move forward, Z remains 1.8m
            pos=(0.0, -1, 1.8),          # <-- NEW: 0.0m Center, -1m FORWARD, 1.8m UP (unchanged)
            dir=(0, -1, 0),                 # Direction is along the negative Y-axis (Forward)
            field_of_view_y=70,
            is_using_shared_memory=True,
            is_render_annotations=False,
            is_render_depth=False,
            near_far_planes=(0.1, 1000),
            resolution=(640, 480),
            requested_update_time=0.05
        )

        electrics = Electrics()
        vehicle.attach_sensor('electrics', electrics)

        save_folder = "E:\\Beamng images"
        os.makedirs(save_folder, exist_ok=True)

        max_frames = 100
        frame_count = 0

        frame_count = image_and_steering(cam,vehicle, electrics, max_frames, frame_count, save_folder)
        print(f"Captured {frame_count} frames.")

    finally:
        try:
            bng.close()
        except:
            pass
        print("🔒 BeamNG closed.")


if __name__ == "__main__":
    main()
