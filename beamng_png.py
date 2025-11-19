#!/usr/bin/env python3
import os
import time
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera

def main():
    set_up_simple_logging()
    
    # -----------------------------
    # BeamNG Path and Connection
    # -----------------------------
    bng_home = "D:\\Downloads\\BeamNG\\BeamNG"  # <-- your BeamNG path
    beamng = BeamNGpy('localhost', 64256, home=bng_home)
    
    print("🔄 Launching BeamNG.drive...")
    bng = beamng.open(launch=True)
    
    # -----------------------------
    # Scenario and Vehicle
    # -----------------------------
    scenario = Scenario('west_coast_usa', 'Camera_demo', description='Camera capture demo')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='RED', color='Red')
    scenario.add_vehicle(vehicle, pos=(0, 0, 0.5))  # Slightly above ground
    scenario.make(bng)
    
    bng.settings.set_deterministic(60)  # 60 Hz physics
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()
    
    # -----------------------------
    # Wait for physics to stabilize
    # -----------------------------
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
    
    print("🎬 Starting image capture...")
    
    while frame_count < max_frames:
        time.sleep(0.05)
        data = cam.poll()
        if data is None or 'colour' not in data:
            continue
        
        img = data['colour']
        
        # If returned as numpy array, convert to PIL
        if not hasattr(img, 'save'):
            from PIL import Image
            img = Image.fromarray(img)
        
        img.save(os.path.join(save_folder, f"frame_{frame_count:03d}.png"))
        frame_count += 1
        print(f"✅ Saved frame {frame_count}/{max_frames}")
    
    print("🎉 Image capture complete!")
    
    cam.remove()
    bng.close()
    print("🔒 BeamNG closed.")

if __name__ == "__main__":
    main()
