from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import GPS, Lidar
import time
import numpy as np

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

def vehicle_location(vehicle: Vehicle):
    vehicle.sensors.poll()
    if vehicle.state:
        position=vehicle.state['pos']
        return position
    
def ground_removal(points, min_x, max_x, min_y, max_y, min_z, max_z, grid_size, tolerance ):
    in_bounds= (min_x <= points[:,0] & max_x > points[:,0] ) & (min_y <= points[:, 1] & max_y > points[:,1]) & (min_z <= points[:,2] & max_z > points[:,2])
    points_filtered=points[in_bounds]

    grid_width=int(np.ceil(max_x-min_x)/grid_size)
    grid_height=int(np.ceil(max_y-min_y)/grid_size)
    grid = np.full((grid_width, grid_height), np.inf)

    xi = ((points_filtered[:, 0] - min_x) / grid_size).astype(np.int32)
    yi = ((points_filtered[:, 1] - min_y) / grid_size).astype(np.int32)
    zi = points_filtered[:, 2]

    ground_mask = (zi <= (grid[xi, yi] + tolerance))

    non_ground_points = points_filtered[~ground_mask]
    return non_ground_points

def stopping_dist(velocity, reaction_time = 1, friction_coef = 0.8, extra_dist = 1):
    #source: https://korkortonline.se/en/theory/reaction-braking-stopping/
    #Assuming by default we have near-perfect conditions with an extra metre to spare.
    reaction_dist = velocity * reaction_time / 3.6
    braking_dist = velocity**2 / (250 * friction_coef)

    return reaction_dist + braking_dist 

