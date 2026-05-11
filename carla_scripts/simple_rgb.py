import queue
import carla

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.*")[0]
    ego = world.spawn_actor(vehicle_bp, world.get_map().get_spawn_points()[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "360")
    cam_bp.set_attribute("fov", "90")

    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=1.6)),
        attach_to=ego,
    )

    q = queue.Queue()
    cam.listen(q.put)

    try:
        for _ in range(10):
            frame = world.tick()
            img = q.get(timeout=5.0)
            print("tick", frame, "sensor", img.frame)
    finally:
        cam.stop()
        cam.destroy()
        ego.destroy()
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)

if __name__ == "__main__":
    main()
