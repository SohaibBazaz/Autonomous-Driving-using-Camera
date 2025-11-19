from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2, os, time
import numpy as np

SAVE_DIR = "camera_images"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_IMAGES = 100

# Launch BeamNG
bng = BeamNGpy('localhost', 64256, home='D:\\Downloads\\BeamNG\\BeamNG')
bng.open(launch=True)

# Scenario + vehicle
scenario = Scenario('west_coast_usa', 'camera_test', description='Camera Test')
vehicle = Vehicle('test_car', model='etk800', licence='CAM123')
scenario.add_vehicle(vehicle, pos=(-717.121, 101, 118.675))
scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()

time.sleep(3)  # wait for physics to stabilize

# Create camera and attach it to vehicle (directly via constructor)
camera = Camera(
    name='front_cam',
    bng=bng,
    vehicle=vehicle,  # <- attach directly here
    pos=(0,0,1.5),
    dir=(0,1,0),
    resolution=(640,480)
)

# Capture 100 images
for i in range(NUM_IMAGES):
    bng.step(1)  # advance physics
    vehicle.sensors.poll()
    img = vehicle.sensors['front_cam']['colour']  # RGBA
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{i:03d}.png"), img_bgr)
    print(f"Saved frame_{i:03d}.png")

print(f"✅ {NUM_IMAGES} images saved in {SAVE_DIR}")
bng.close()
