from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import GPS, Lidar
import time

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
    
    ##mycar.ai_set_mode('manual')
    ##waypoints = [(0, 0, 0), (50, 0, 0), (50, 50, 0), (0, 50, 0)]
   ## mycar.ai_set_line(waypoints)
    ##mycar.ai_set_speed(15, mode='set')
    print("Vehicle connected!")
    return mycar

def print_lidar_data(lidar):
    try:
        data = lidar.poll()
        if not data or 'pointCloud' not in data:
            print("[Lidar] No data received yet.")
            return
        
        point_count = data["pointCloud"].shape[0]
        print(f"[Lidar] Points: {point_count}")

        print("Sample points (x, y, z):")
        for p in data["pointCloud"][:5]:
            print(f"  {p}")

        print("-" * 40)
    except Exception as e:
        print(f"[Lidar Error] {e}")

def aivehicle():
    Vehicle.ai.set_mode("manual")
    

def main():
    bng = BeamNGpy('localhost', 25252, home=r'D:\\Downloads\\BeamNG\\BeamNG', user=r'D:\\Downloads\\BeamNG\\userfolder')
    bng.open(launch=True)

    map_name = 'east_coast_usa'
    mycar = start_scenario(bng, map_name, 'gps_test')
    

    lidar = Lidar(
        name="lidar_test",
        bng=bng,
        vehicle=mycar,
        requested_update_time=0.05,
        vertical_angle=30,
        horizontal_angle=90,
        vertical_resolution=16,
        is_visualised=True,
        is_360_mode=False
    )

    gps = GPS(
        name="gps_test",
        bng=bng,
        vehicle=mycar,
        physics_update_time=1,
        ref_lat=33.6844,
        ref_lon=73.0479,
        is_visualised=True
    )

    print("⏳ Waiting for sensors to initialize...")
    for _ in range(5):
        gps.poll()
        lidar.poll()
        time.sleep(0.5)
    print("✅ Sensors initialized, starting main loop...")    
    
    
    current_brake = 0.0
    while True:
        try:
            # --- GPS ---
            gps_data = gps.poll()
            if gps_data and 'lat' in gps_data and 'lon' in gps_data:
                print(f"[GPS] lat={gps_data['lat']:.6f}, lon={gps_data['lon']:.6f}")
            else:
                print("[GPS] No GPS data yet.")

            # --- Default Controls ---
            throttle_car = 0.5
            target_brake= 0.0
            steering_car = 0.0

            # --- LIDAR ---
            lidar_data = lidar.poll()
            if lidar_data and 'pointCloud' in lidar_data and len(lidar_data['pointCloud']) > 0:
                points = lidar_data['pointCloud']
                print(f"[DEBUG] Total points: {len(points)}")
                # ✅ Filter points roughly in front of car
                forward_points = [p for p in points if p[0] > -1 and abs(p[1]) < 5]
                print(f"[DEBUG] Total points: {len(points)}, forward_points: {len(forward_points)}")

                if forward_points:
                    min_dist = min([p[0] for p in forward_points])

                    if min_dist < 5:
                        throttle_car = 0.0
                        target_brake = 1.0
                        print("🚧 Obstacle detected! Stopping.")
                    else:
                        throttle_car = 0.8
                        target_brake = 0.0
                        print(f"✅ Clear path, nearest obstacle ahead: {min_dist:.2f} m")
                else:
                    print("[LIDAR] No forward points detected.")
            else:
                print("[LIDAR] No data yet.")
            if current_brake < target_brake:
                current_brake = min(current_brake + 0.05, target_brake)
            elif current_brake > target_brake:
                current_brake = max(current_brake - 0.05, target_brake)

            # --- Apply Controls ---
            mycar.control(throttle=throttle_car, brake=current_brake, steering=steering_car)

            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            mycar.control(throttle=0, brake=1.0)  # Safety stop
            time.sleep(1)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    main()