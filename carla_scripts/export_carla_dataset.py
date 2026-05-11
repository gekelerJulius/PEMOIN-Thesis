from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
from PIL import Image
import carla


CARLA_LABEL_DUMP_PATH = Path(
    "/home/juli/PycharmProjects/PEMOIN/carla_scripts/carla_label_map_dump.json"
)


def _load_carla_label_dump(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data.get("CityObjectLabel_name_to_id")
    if not isinstance(m, dict) or not all(
        isinstance(k, str) and isinstance(v, int) for k, v in m.items()
    ):
        raise ValueError(
            f"Invalid label dump format in {path} (missing CityObjectLabel_name_to_id)."
        )

    # Optional: normalize common singular names to plural keys used by CARLA dump
    # so downstream code can use "Road" / "Sidewalk" etc.
    aliases = {
        "Unlabeled": "NONE",
        "Road": "Roads",
        "Sidewalk": "Sidewalks",
        "Building": "Buildings",
        "Wall": "Walls",
        "Fence": "Fences",
        "Pole": "Poles",
        "TrafficSign": "TrafficSigns",
        "Pedestrian": "Pedestrians",
        "RoadLine": "RoadLines",
    }
    for alias, target in aliases.items():
        if target in m and alias not in m:
            m[alias] = m[target]

    return m


# Load once at import time (fast) so exporter uses exact IDs from your install
CARLA_SEMANTIC_TAGS_NAME_TO_ID: Dict[str, int] = _load_carla_label_dump(
    CARLA_LABEL_DUMP_PATH
)
CARLA_SEMANTIC_TAGS_ID_TO_NAME: Dict[int, str] = {
    v: k for k, v in CARLA_SEMANTIC_TAGS_NAME_TO_ID.items()
}


def write_label_maps(out_root: Path, *, has_sem: bool, has_inst: bool) -> None:
    """
    Writes label maps that explain what IDs mean in the exported *_id PNGs.
    - semseg_id: CARLA semantic tag IDs (as per your CityObjectLabel dump)
    - instseg_id: instance IDs (per-object), NOT a semantic-class label map
    """
    if has_sem:
        payload = {
            "format": "carla_semantic_segmentation_raw",
            "id_channel": "R (red) channel",
            "source": {
                "label_dump_path": str(CARLA_LABEL_DUMP_PATH),
            },
            "name_to_id": CARLA_SEMANTIC_TAGS_NAME_TO_ID,
            "id_to_name": {
                str(k): v for k, v in CARLA_SEMANTIC_TAGS_ID_TO_NAME.items()
            },
            "notes": [
                "semseg_id PNG stores CARLA semantic tag ID per pixel (Raw semantic camera).",
                "Mapping loaded from your local CityObjectLabel dump file.",
            ],
        }
        with open(out_root / "semseg_label_map.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if has_inst:
        payload = {
            "format": "carla_instance_segmentation_raw",
            "notes": [
                "instseg_id PNG stores INSTANCE ID per pixel (changes per run; not a fixed class map).",
                "In CARLA instance segmentation Raw, semantic tag is in R; instance id is encoded in G/B.",
                "Your exporter currently writes only one channel as grayscale; treat it as an ID image, not a class map.",
            ],
        }
        with open(out_root / "instseg_id_description.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# -------------------------
# Logging (unbuffered, timestamped)
# -------------------------
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -------------------------
# Bounded "latest frame" queue
# -------------------------
class LatestQueue:
    def __init__(self, name: str):
        self.name = name
        self._q: "queue.Queue[carla.Image]" = queue.Queue(maxsize=1)
        self.last_put_frame: int = -1

    def put_latest(self, img: carla.Image) -> None:
        self.last_put_frame = int(img.frame)
        try:
            _ = self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(img)
        except queue.Full:
            pass

    def get(self, timeout: float) -> carla.Image:
        return self._q.get(timeout=timeout)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fov_deg: float
    fx: float
    fy: float
    cx: float
    cy: float


def compute_intrinsics(width: int, height: int, fov_deg: float) -> Intrinsics:
    fov = math.radians(fov_deg)
    fx = width / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(width, height, fov_deg, fx, fy, cx, cy)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _enum_name(value: Any) -> str:
    raw = str(value)
    if "." in raw:
        return raw.rsplit(".", 1)[-1]
    return raw


def _color_payload(color: Any) -> Dict[str, int]:
    return {
        "r": int(getattr(color, "r", 0)),
        "g": int(getattr(color, "g", 0)),
        "b": int(getattr(color, "b", 0)),
    }


def _location_payload(location: Any) -> Dict[str, float]:
    return {
        "x": float(getattr(location, "x", 0.0)),
        "y": float(getattr(location, "y", 0.0)),
        "z": float(getattr(location, "z", 0.0)),
    }


def _weather_payload(weather: carla.WeatherParameters) -> Dict[str, float]:
    payload = {
        "cloudiness": float(weather.cloudiness),
        "precipitation": float(weather.precipitation),
        "precipitation_deposits": float(weather.precipitation_deposits),
        "wind_intensity": float(weather.wind_intensity),
        "sun_azimuth_angle": float(weather.sun_azimuth_angle),
        "sun_altitude_angle": float(weather.sun_altitude_angle),
        "fog_density": float(weather.fog_density),
        "fog_distance": float(weather.fog_distance),
        "wetness": float(weather.wetness),
        "fog_falloff": float(weather.fog_falloff),
        "scattering_intensity": float(getattr(weather, "scattering_intensity", 0.0)),
        "mie_scattering_scale": float(getattr(weather, "mie_scattering_scale", 0.0)),
        "rayleigh_scattering_scale": float(
            getattr(weather, "rayleigh_scattering_scale", 0.0)
        ),
    }
    dust_storm = getattr(weather, "dust_storm", None)
    if dust_storm is not None:
        payload["dust_storm"] = float(dust_storm)
    return payload


def _serialize_light_state(light_state: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "repr": str(light_state),
        "enum_name": _enum_name(light_state),
    }
    if hasattr(light_state, "group"):
        payload["group"] = _enum_name(getattr(light_state, "group"))
    if hasattr(light_state, "active"):
        payload["active"] = bool(getattr(light_state, "active"))
    if hasattr(light_state, "intensity"):
        payload["intensity"] = float(getattr(light_state, "intensity"))
    if hasattr(light_state, "color"):
        payload["color"] = _color_payload(getattr(light_state, "color"))
    return payload


def _serialize_scene_light(light: Any) -> Dict[str, Any]:
    return {
        "id": int(getattr(light, "id")),
        "location": _location_payload(getattr(light, "location")),
        "color": _color_payload(getattr(light, "color")),
        "intensity": float(getattr(light, "intensity")),
        "is_on": bool(getattr(light, "is_on")),
        "light_group": _enum_name(getattr(light, "light_group")),
        "light_state": _serialize_light_state(getattr(light, "light_state")),
    }


def _capture_scene_lights(world: carla.World) -> Dict[str, Any]:
    lm = world.get_lightmanager()
    lights = lm.get_all_lights()
    serialized = [_serialize_scene_light(light) for light in lights]
    active_count = sum(1 for item in serialized if item["is_on"])
    group_counts: Dict[str, int] = {}
    for item in serialized:
        group = str(item["light_group"])
        group_counts[group] = group_counts.get(group, 0) + 1
    return {
        "total_light_count": int(len(serialized)),
        "active_light_count": int(active_count),
        "group_counts": group_counts,
        "lights": serialized,
    }


def _carla_version_string() -> str | None:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import carla; print(getattr(carla, '__file__', ''))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return getattr(carla, "__file__", None)
    path = result.stdout.strip()
    return path or getattr(carla, "__file__", None)


def save_rgb(img: carla.Image, path: Path) -> None:
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
        (img.height, img.width, 4)
    )
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    Image.fromarray(rgb).save(path, quality=92, subsampling=0)


def depth_to_meters(depth_img: carla.Image) -> np.ndarray:
    arr = np.frombuffer(depth_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((depth_img.height, depth_img.width, 4))[:, :, :3].astype(
        np.uint32
    )
    b = arr[:, :, 0]
    g = arr[:, :, 1]
    r = arr[:, :, 2]
    normalized = (r + g * 256 + b * 256 * 256) / float(256**3 - 1)
    return (normalized * 1000.0).astype(np.float32)


def save_seg_ids(img: carla.Image, path: Path) -> None:
    img.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
        (img.height, img.width, 4)
    )
    ids = arr[:, :, 2]
    Image.fromarray(ids, mode="L").save(path)


def save_seg_preview(img: carla.Image, path: Path) -> None:
    # Colorized visualization for humans
    img.convert(carla.ColorConverter.CityScapesPalette)
    img.save_to_disk(str(path))


def make_cam_bp(
    bp_lib: carla.BlueprintLibrary, bp_id: str, w: int, h: int, fov: float
) -> carla.ActorBlueprint:
    bp = bp_lib.find(bp_id)
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    if bp.has_attribute("sensor_tick"):
        bp.set_attribute("sensor_tick", "0.0")
    return bp


def make_run_dir(
    parent: Path,
    town: str,
    fps: int,
    width: int,
    height: int,
    fov: float,
    sem: bool,
    inst: bool,
    frames: int,
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_{town}_fps{fps}_{width}x{height}_fov{int(round(fov))}_sem{int(sem)}_inst{int(inst)}_n{frames}"
    run_dir = parent / tag
    ensure_dir(run_dir)
    return run_dir


def _dist(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _pick_spawn_points_near(
    world: carla.World,
    center: carla.Location,
    radius_m: float,
) -> List[carla.Transform]:
    sps = world.get_map().get_spawn_points()
    near = [sp for sp in sps if _dist(sp.location, center) <= radius_m]
    return near if len(near) >= 5 else sps  # fallback if too few nearby


def _sample_nav_locations_near(
    world: carla.World,
    center: carla.Location,
    radius_m: float,
    n: int,
    seed: int,
    max_tries: int = 2000,
) -> List[carla.Location]:
    rng = np.random.RandomState(seed)
    out: List[carla.Location] = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if _dist(loc, center) <= radius_m:
            out.append(loc)
    # If nav sampling near center fails (some towns), fall back to any nav locations
    if len(out) < n:
        while len(out) < n and tries < max_tries * 2:
            tries += 1
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                out.append(loc)
    rng.shuffle(out)
    return out[:n]


def spawn_traffic_near_ego(
    client: carla.Client,
    world: carla.World,
    traffic: carla.TrafficManager,
    ego: carla.Actor,
    *,
    num_vehicles: int,
    num_walkers: int,
    seed: int,
    radius_vehicles_m: float = 120.0,
    radius_walkers_m: float = 80.0,
    tm_speed_diff_pct: float = 0.0,  # +20 slower, -20 faster
    ego_autopilot: bool = True,
    warmup_ticks: int = 30,
) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    """
    Spawns vehicles + pedestrians NEAR the ego so they appear in ego-mounted camera.
    Deterministic-ish under synchronous mode (seeded sampling, but nav sampling depends on map).
    Returns: (vehicles, walkers_and_controllers)
    """
    bp_lib = world.get_blueprint_library()
    rng = np.random.RandomState(seed)

    spawned_vehicles: List[carla.Actor] = []
    spawned_walkers_and_ctrls: List[carla.Actor] = []

    # ---- Traffic manager ----
    traffic.set_random_device_seed(seed)
    traffic.set_synchronous_mode(True)
    traffic.global_percentage_speed_difference(float(tm_speed_diff_pct))

    if ego_autopilot:
        ego.set_autopilot(True, traffic.get_port())

    ego_loc = ego.get_location()

    # ---- Vehicles near ego ----
    spawn_points = _pick_spawn_points_near(world, ego_loc, radius_vehicles_m)
    rng.shuffle(spawn_points)

    vehicle_bps = bp_lib.filter("vehicle.*")
    if not vehicle_bps:
        return spawned_vehicles, spawned_walkers_and_ctrls

    # Use batch spawn for speed + fewer partial failures
    batch = []
    for sp in spawn_points[: min(num_vehicles, len(spawn_points))]:
        bp = vehicle_bps[int(rng.randint(0, len(vehicle_bps)))]
        # Improve variety + avoid some problematic vehicles
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", colors[int(rng.randint(0, len(colors)))])
        batch.append(carla.command.SpawnActor(bp, sp))

    results = client.apply_batch_sync(batch, True)
    vehicle_ids = [r.actor_id for r in results if not r.error]
    vehicles = world.get_actors(vehicle_ids)

    for v in vehicles:
        v.set_autopilot(True, traffic.get_port())
        spawned_vehicles.append(v)

    # ---- Walkers near ego ----
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if walker_bps:
        controller_bp = bp_lib.find("controller.ai.walker")

        nav_locs = _sample_nav_locations_near(
            world, ego_loc, radius_walkers_m, num_walkers, seed=seed + 1
        )
        walker_batch = []
        for loc in nav_locs:
            bp = walker_bps[int(rng.randint(0, len(walker_bps)))]
            walker_batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)))

        walker_results = client.apply_batch_sync(walker_batch, True)
        walker_ids = [r.actor_id for r in walker_results if not r.error]

        ctrl_batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), wid)
            for wid in walker_ids
        ]
        ctrl_results = client.apply_batch_sync(ctrl_batch, True)
        ctrl_ids = [r.actor_id for r in ctrl_results if not r.error]

        walkers = world.get_actors(walker_ids)
        ctrls = world.get_actors(ctrl_ids)

        for ctrl in ctrls:
            ctrl.start()

        # Assign destinations + speeds (near-ish) so they actually move in your view
        for ctrl in ctrls:
            dests = _sample_nav_locations_near(
                world,
                ego_loc,
                radius_walkers_m * 3.0,
                1,
                seed=int(rng.randint(0, 1_000_000)),
            )
            if dests:
                ctrl.go_to_location(dests[0])
            ctrl.set_max_speed(float(rng.uniform(0.8, 1.7)))

        spawned_walkers_and_ctrls.extend(list(walkers))
        spawned_walkers_and_ctrls.extend(list(ctrls))

    # ---- Warm-up ticks (critical so autopilot/controllers begin moving) ----
    for _ in range(int(warmup_ticks)):
        world.tick()

    return spawned_vehicles, spawned_walkers_and_ctrls


def _safe_listen_off(a: Optional[carla.Actor]) -> None:
    if a is None:
        return
    try:
        if getattr(a, "is_alive", False):
            # Sensors support listen(None) to detach callback
            a.listen(None)  # type: ignore[attr-defined]
    except Exception:
        pass


def _batch_destroy(client: carla.Client, actors: List[carla.Actor]) -> None:
    ids = []
    for a in actors:
        try:
            # Prefer IDs; don't call methods if not needed
            if a is not None and getattr(a, "id", None) is not None:
                ids.append(a.id)
        except Exception:
            pass

    if not ids:
        return

    cmds = [carla.command.DestroyActor(x) for x in ids]
    try:
        client.apply_batch_sync(cmds, True)
    except Exception:
        # As a fallback, try per-actor destroy guarded
        for a in actors:
            try:
                if a is not None and getattr(a, "is_alive", False):
                    a.destroy()
            except Exception:
                pass


def main() -> None:
    # -------------------------
    # USER SETTINGS
    # -------------------------
    exports_parent = Path("./../carla_exports")

    fps = 30
    num_frames = fps * 6
    width, height = 1280, 720
    fov = 90.0

    enable_semseg = True
    enable_instseg = True
    enable_lighting_gt = True

    town = "Town01"
    seed = 42
    tm_global_speed_diff_pct = 20  # +20 = slower, -20 = faster
    cam_rel = carla.Transform(
        carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )

    log(
        f"CONFIG: frames={num_frames}, fps={fps}, res={width}x{height}, fov={fov}, "
        f"town={town}, sem={enable_semseg}, inst={enable_instseg}"
    )
    log(f"Python: {sys.version.split()[0]}  PID: {os.getpid()}")

    # Unique export directory per run
    ensure_dir(exports_parent)
    out_root = make_run_dir(
        exports_parent,
        town,
        fps,
        width,
        height,
        fov,
        enable_semseg,
        enable_instseg,
        num_frames,
    )
    log(f"Export dir: {out_root.resolve()}")

    # -------------------------
    # CONNECT
    # -------------------------
    log("Connecting to CARLA...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    log(f"Connected. Current map: {current_map}")
    if current_map != town:
        raise RuntimeError(
            f"CARLA is running '{current_map}' but exporter expects '{town}'. "
            f"Switch map outside exporter (do not load_world here)."
        )

    # Sync settings
    log("Applying synchronous settings...")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(fps)
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    log("Sync settings applied.")

    # Traffic manager (kept but not used by autopilot here)
    log("Configuring TrafficManager...")
    traffic = client.get_trafficmanager(8000)
    traffic.set_synchronous_mode(True)
    traffic.set_random_device_seed(seed)
    traffic.global_percentage_speed_difference(tm_global_speed_diff_pct)
    log("TrafficManager configured.")

    bp_lib = world.get_blueprint_library()

    # -------------------------
    # SPAWN EGO
    # -------------------------
    log("Spawning ego vehicle...")
    vehicle_bp = bp_lib.filter("vehicle.*")[0]
    sp = world.get_map().get_spawn_points()[5]
    ego = world.spawn_actor(vehicle_bp, sp)
    ego.set_autopilot(False)
    log(f"Ego spawned: id={ego.id} type={vehicle_bp.id}")

    ego_z = ego.get_location().z
    wp_z = world.get_map().get_waypoint(ego.get_location()).transform.location.z
    print("ego-road delta:", ego_z - wp_z)

    # -------------------------
    # SPAWN TRAFFIC ACTORS
    # -------------------------

    spawned_vehicles: List[carla.Actor] = []
    spawned_walkers: List[carla.Actor] = []
    enable_traffic = True
    num_vehicles = 1000
    num_walkers = 600

    if enable_traffic:
        log(
            f"Spawning traffic near ego: vehicles={num_vehicles}, walkers={num_walkers}"
        )
        spawned_vehicles, spawned_walkers = spawn_traffic_near_ego(
            client,
            world,
            traffic,
            ego,
            num_vehicles=num_vehicles,
            num_walkers=num_walkers,
            seed=seed,
            radius_vehicles_m=60.0,
            radius_walkers_m=20.0,
            tm_speed_diff_pct=15.0,  # +10 => slightly slower traffic
            ego_autopilot=True,  # IMPORTANT: otherwise ego may never see traffic
            warmup_ticks=30,  # increases chance first recorded frames show motion
        )
        log(
            f"Spawned: vehicles={len(spawned_vehicles)}, walker+ctrl={len(spawned_walkers)}"
        )

    # -------------------------
    # SPAWN SENSORS
    # -------------------------
    log("Creating sensor blueprints...")
    rgb_bp = make_cam_bp(bp_lib, "sensor.camera.rgb", width, height, fov)
    depth_bp = make_cam_bp(bp_lib, "sensor.camera.depth", width, height, fov)
    sem_bp = (
        make_cam_bp(bp_lib, "sensor.camera.semantic_segmentation", width, height, fov)
        if enable_semseg
        else None
    )
    inst_bp = (
        make_cam_bp(bp_lib, "sensor.camera.instance_segmentation", width, height, fov)
        if enable_instseg
        else None
    )
    log("Spawning sensors...")

    rgb = world.spawn_actor(rgb_bp, cam_rel, attach_to=ego)
    log(f"RGB sensor spawned: id={rgb.id}")

    depth = world.spawn_actor(depth_bp, cam_rel, attach_to=ego)
    log(f"Depth sensor spawned: id={depth.id}")

    sem = world.spawn_actor(sem_bp, cam_rel, attach_to=ego) if sem_bp else None
    if sem:
        log(f"SemSeg sensor spawned: id={sem.id}")

    inst = world.spawn_actor(inst_bp, cam_rel, attach_to=ego) if inst_bp else None
    if inst:
        log(f"InstSeg sensor spawned: id={inst.id}")

    q_rgb = LatestQueue("rgb")
    q_depth = LatestQueue("depth")
    q_sem = LatestQueue("semseg") if sem else None
    q_inst = LatestQueue("instseg") if inst else None

    log("Attaching sensor listeners...")
    rgb.listen(q_rgb.put_latest)
    depth.listen(q_depth.put_latest)
    if sem and q_sem:
        sem.listen(q_sem.put_latest)
    if inst and q_inst:
        inst.listen(q_inst.put_latest)
    log("Listeners attached.")

    # -------------------------
    # OUTPUT
    # -------------------------
    log("Preparing output folders...")
    ensure_dir(out_root / "rgb")
    ensure_dir(out_root / "depth_m")
    lighting_gt_dir = out_root / "lighting_gt"
    if enable_lighting_gt:
        ensure_dir(lighting_gt_dir)
    if sem:
        ensure_dir(out_root / "semseg_id")
        ensure_dir(out_root / "semseg_vis")
    if inst:
        ensure_dir(out_root / "instseg_id")
        ensure_dir(out_root / "instseg_vis")

    intr = compute_intrinsics(width, height, fov)
    with open(out_root / "camera_intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intr.__dict__, f, indent=2)
    log("Wrote camera_intrinsics.json")

    # Write run config for reproducibility
    run_cfg = {
        "schema_version": 2,
        "town": town,
        "fps": fps,
        "width": width,
        "height": height,
        "fov": fov,
        "num_frames": num_frames,
        "enable_semseg": enable_semseg,
        "enable_instseg": enable_instseg,
        "enable_lighting_gt": enable_lighting_gt,
        "seed": seed,
        "camera_mount": {
            "location": {
                "x": cam_rel.location.x,
                "y": cam_rel.location.y,
                "z": cam_rel.location.z,
            },
            "rotation": {
                "pitch": cam_rel.rotation.pitch,
                "yaw": cam_rel.rotation.yaw,
                "roll": cam_rel.rotation.roll,
            },
        },
        "carla_map_actual": current_map,
        "carla_module_path": _carla_version_string(),
    }
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)
    log("Wrote run_config.json")

    write_label_maps(out_root, has_sem=bool(sem), has_inst=bool(inst))
    log("Wrote semseg_label_map.json / instseg_id_description.json")

    # Stream metadata to JSONL (more robust than keeping all in RAM)
    meta_path = out_root / "frames.jsonl"
    meta_f = open(meta_path, "w", encoding="utf-8")
    frame_lighting_f = None

    if enable_lighting_gt:
        scene_lights = _capture_scene_lights(world)
        lighting_run = {
            "schema_version": 1,
            "source": "carla_gt",
            "town": town,
            "carla_map_actual": current_map,
            "fps": fps,
            "fixed_delta_seconds": float(settings.fixed_delta_seconds or 0.0),
            "seed": seed,
            "is_night": bool(world.get_weather().sun_altitude_angle < 0.0),
            "weather": _weather_payload(world.get_weather()),
            "scene_lights_summary": {
                "total_light_count": int(scene_lights["total_light_count"]),
                "active_light_count": int(scene_lights["active_light_count"]),
                "group_counts": dict(scene_lights["group_counts"]),
            },
        }
        with open(lighting_gt_dir / "run_lighting.json", "w", encoding="utf-8") as f:
            json.dump(lighting_run, f, indent=2)
        with open(lighting_gt_dir / "scene_lights.json", "w", encoding="utf-8") as f:
            json.dump(scene_lights, f, indent=2)
        frame_lighting_f = open(
            lighting_gt_dir / "frame_lighting.jsonl", "w", encoding="utf-8"
        )

    # -------------------------
    # CAPTURE LOOP
    # -------------------------
    log("Starting capture loop...")
    t0 = time.time()
    captured = 0

    try:
        for i in range(num_frames):
            frame = world.tick()

            if i < 10 or (i + 1) % 50 == 0:
                log(f"Ticked world frame={frame} (i={i+1}/{num_frames})")

            img_rgb = q_rgb.get(timeout=10.0)
            img_depth = q_depth.get(timeout=10.0)

            while img_rgb.frame < frame:
                img_rgb = q_rgb.get(timeout=10.0)
            while img_depth.frame < frame:
                img_depth = q_depth.get(timeout=10.0)

            img_sem = None
            img_inst = None
            if sem and q_sem:
                img_sem = q_sem.get(timeout=10.0)
                while img_sem.frame < frame:
                    img_sem = q_sem.get(timeout=10.0)
            if inst and q_inst:
                img_inst = q_inst.get(timeout=10.0)
                while img_inst.frame < frame:
                    img_inst = q_inst.get(timeout=10.0)

            if (i < 10) or ((i + 1) % 50 == 0):
                log(
                    f"frames: world={frame} rgb={img_rgb.frame} depth={img_depth.frame} "
                    f"sem={(img_sem.frame if img_sem else None)} inst={(img_inst.frame if img_inst else None)}"
                )

            # File paths
            rgb_path = out_root / "rgb" / f"{frame:06d}.jpg"
            depth_path = out_root / "depth_m" / f"{frame:06d}.npy"

            # Save RGB
            save_rgb(img_rgb, rgb_path)

            # Save depth (meters, float32)
            depth_m = depth_to_meters(img_depth)
            np.save(depth_path, depth_m)

            # Save seg IDs + previews
            sem_id_path = sem_vis_path = None
            inst_id_path = inst_vis_path = None

            if img_sem is not None:
                sem_id_path = out_root / "semseg_id" / f"{frame:06d}.png"
                sem_vis_path = out_root / "semseg_vis" / f"{frame:06d}.png"
                save_seg_ids(img_sem, sem_id_path)
                save_seg_preview(img_sem, sem_vis_path)

            if img_inst is not None:
                inst_id_path = out_root / "instseg_id" / f"{frame:06d}.png"
                inst_vis_path = out_root / "instseg_vis" / f"{frame:06d}.png"
                save_seg_ids(img_inst, inst_id_path)
                save_seg_preview(img_inst, inst_vis_path)

            # Pose + metadata (streaming)
            snap = world.get_snapshot()
            T_world_from_cam = rgb.get_transform().get_matrix()

            rec = {
                "frame": int(frame),
                "timestamp": float(snap.timestamp.elapsed_seconds),
                "T_world_from_camera": T_world_from_cam,
                "rgb": f"rgb/{frame:06d}.jpg",
                "depth_m": f"depth_m/{frame:06d}.npy",
                "semseg_id": f"semseg_id/{frame:06d}.png" if sem_id_path else None,
                "semseg_vis": f"semseg_vis/{frame:06d}.png" if sem_vis_path else None,
                "instseg_id": f"instseg_id/{frame:06d}.png" if inst_id_path else None,
                "instseg_vis": (
                    f"instseg_vis/{frame:06d}.png" if inst_vis_path else None
                ),
            }
            meta_f.write(json.dumps(rec) + "\n")

            if enable_lighting_gt and frame_lighting_f is not None:
                weather = world.get_weather()
                active_scene_light_count = _capture_scene_lights(world)[
                    "active_light_count"
                ]
                frame_lighting_f.write(
                    json.dumps(
                        {
                            "frame": int(frame),
                            "timestamp": float(snap.timestamp.elapsed_seconds),
                            "weather": _weather_payload(weather),
                            "is_night": bool(weather.sun_altitude_angle < 0.0),
                            "active_scene_light_count": int(active_scene_light_count),
                        }
                    )
                    + "\n"
                )
            captured += 1

            # Light flush to reduce data loss risk without killing performance
            if (i + 1) % 200 == 0:
                meta_f.flush()
                os.fsync(meta_f.fileno())
                if frame_lighting_f is not None:
                    frame_lighting_f.flush()
                    os.fsync(frame_lighting_f.fileno())

    except Exception as e:
        log("EXCEPTION in capture loop:")
        log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise

    finally:
        if frame_lighting_f is not None:
            frame_lighting_f.close()
        meta_f.close()
        # Stop sensor callbacks first (prevents shutdown races)
        for s in [rgb, depth, sem, inst]:
            _safe_listen_off(s)

        # Collect everything you spawned
        to_destroy: List[carla.Actor] = []
        to_destroy.extend([a for a in [rgb, depth, sem, inst] if a is not None])
        to_destroy.extend(spawned_vehicles)
        to_destroy.extend(spawned_walkers)  # includes controllers in your function
        to_destroy.append(ego)

        # Destroy via batch by id (robust even if some handles are already dead)
        _batch_destroy(client, to_destroy)

        # Restore settings last
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)

        except Exception:
            pass

        try:
            traffic.set_synchronous_mode(False)

        except Exception:
            pass


if __name__ == "__main__":
    main()
