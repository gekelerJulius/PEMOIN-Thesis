import carla
import sys

town = sys.argv[1] if len(sys.argv) > 1 else "Town01"

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
print("Loading:", town)
world = client.load_world(town)
print("Loaded:", world.get_map().name)
