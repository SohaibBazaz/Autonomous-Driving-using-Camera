import time

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera

def main():
    set_up_simple_logging()

    bng_home = "D:\\Downloads\\BeamNG\\BeamNG"  
    bng = BeamNGpy('localhost', 64256, home=bng_home)

    scenario = Scenario('gridmap', 'Camera demo', description='Camera stream test')

    vehicle = Vehicle('ego_vehicle', model='etk800', license='RED', color='Red')

    scenario.add_vehicle(vehicle)
    scenario.make(bng)

    bng.settings.set_deterministic(60) # Set simulator to 60hz temporal resolution

    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    # NOTE: Create sensor after scenario has started.
    cam1 = Camera(
        'camera1',
        bng,
        vehicle,
        is_using_shared_memory=True,
        pos=(-5, 0, 1),
        dir=(1, 0, 0),
        field_of_view_y=70,
        is_render_annotations=False,
        is_render_depth=False,
        near_far_planes=(0.1, 1000),
        resolution=(3840, 2160),
        requested_update_time=0.05)

    t_last = 0.0
    for _ in range(10000):
        time.sleep(0.015)
        readings_data = cam1.poll()
        t = time.time()
        dt = t - t_last
        t_last = t
        print("time since last poll (s): ", dt)

    cam1.remove()
    bng.close()

if __name__ == '__main__':
    main()
