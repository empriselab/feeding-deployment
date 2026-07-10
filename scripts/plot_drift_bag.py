#!/usr/bin/env python3
"""
plot_drift_bag.py -- offline PNG of the fused_odom_observer drift traces.

Reads a fused_drift_*.bag (recorded by fused_odom_observer.launch record:=true)
and overlays, on the localization map, the five map-anchored base trajectories
the run captured:

  carto          green    map->base (lidar-corrected reference, closed loop)
  zed            red      raw ZED odom            (open loop)
  zed_sanitized  orange   sanitized ZED odom      (open loop)
  wheel          blue     /wheel_odom             (open loop)
  fused          purple   /odometry/fused (EKF: sanitized-ZED + wheel), open loop

The baking into `map` was already done live by drift_trace_compare.py and
recorded as /drift_test/<name>_path, so this script just reads the LAST (most
complete) Path per topic and plots it -- no offline TF math. Read-only; writes
a PNG and touches no ROS master.

Needs a python with rosbag + matplotlib + numpy (the ROS/conda env that runs
the nodes, e.g. ~/miniconda3/envs/feeding_deployment/bin/python -- NOT the
bare /usr/bin/python3, which lacks rosbag). --map is optional (needs a P5 .pgm
beside the yaml).

Examples:
  python scripts/plot_drift_bag.py                         # newest fused_drift bag
  python scripts/plot_drift_bag.py <bag> --map maps/aimee-7-1.yaml
  python scripts/plot_drift_bag.py <bag> --out /tmp/drift.png
"""

import argparse
import glob
import json
import os
import sys

# (topic, legend name, color) -- order = z-order (carto reference on top).
TRACES = [
    ("/drift_test/zed_path", "zed (raw ZED)", "red"),
    ("/drift_test/zed_sanitized_path", "zed_sanitized", "orange"),
    ("/drift_test/wheel_path", "wheel", "blue"),
    ("/drift_test/fused_path", "fused (stock EKF)", "purple"),
    # Stage-1 ablation variants -- present only on ablate:=true runs; a bag
    # without one of these topics simply omits that line.
    ("/drift_test/fused_gyro_path", "fused +gyro-yaw", "cyan"),
    ("/drift_test/fused_gate_path", "fused +reject-gate", "magenta"),
    ("/drift_test/fused_zupt_path", "fused +ZUPT", "brown"),
    ("/drift_test/fused_improved_path", "fused improved (all)", "black"),
    ("/drift_test/fused_novio_path", "fused no-VIO (wheel+IMU)", "teal"),
    ("/drift_test/carto_path", "carto (reference)", "green"),
]
LOG_BASE = os.path.expanduser(
    "~/deployment_ws/src/feeding-deployment/src/feeding_deployment/integration/"
    "log/system_logs")


def newest_bag():
    bags = glob.glob(os.path.join(LOG_BASE, "fused_drift*.bag"))
    return max(bags, key=os.path.getmtime) if bags else None


def read_pgm(path):
    """Read a binary (P5) PGM -> (bytes, W, H) or None. (Same parser as
    plot_nav_traces.py, kept local so this script has no cross-imports.)"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    idx, tokens = 0, []
    while raw[idx:idx + 1].isspace():
        idx += 1
    if raw[idx:idx + 2] != b"P5":
        return None
    idx += 2
    while len(tokens) < 3:
        while raw[idx:idx + 1].isspace():
            idx += 1
        if raw[idx:idx + 1] == b"#":
            while raw[idx:idx + 1] not in (b"\n", b""):
                idx += 1
            continue
        start = idx
        while not raw[idx:idx + 1].isspace():
            idx += 1
        tokens.append(int(raw[start:idx]))
    W, H, _maxv = tokens
    idx += 1
    return raw[idx:idx + W * H], W, H


def load_map(yaml_path, np):
    """map_server yaml -> (img2d, extent) or None."""
    if not yaml_path or not os.path.isfile(yaml_path):
        return None
    meta = {}
    with open(yaml_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    res = float(meta.get("resolution", "0.05"))
    origin = json.loads(meta.get("origin", "[0,0,0]").replace("(", "[").replace(")", "]"))
    img = meta.get("image", "map.pgm")
    if not os.path.isabs(img):
        img = os.path.join(os.path.dirname(yaml_path), img)
    pgm = read_pgm(img)
    if pgm is None:
        return None
    data, W, H = pgm
    arr = np.frombuffer(data, dtype=np.uint8)[:W * H].reshape(H, W)
    extent = [origin[0], origin[0] + W * res, origin[1], origin[1] + H * res]
    return arr, extent


def load_paths(bag_path, np):
    """Last (fullest) Path per drift topic -> {name: (xy Nx2 or None, color)}."""
    import rosbag
    topics = [t for t, _, _ in TRACES]
    last = {}
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=topics):
            last[topic] = msg
    out = []
    for topic, name, color in TRACES:
        m = last.get(topic)
        if m is None or not m.poses:
            out.append((name, None, color))
            continue
        xy = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])
        out.append((name, xy, color))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", default=None,
                    help="fused_drift bag (default: newest under system_logs/)")
    ap.add_argument("--map", default=None, help="map_server yaml for the background")
    ap.add_argument("--out", default=None, help="output PNG (default: <bag>_traces.png)")
    args = ap.parse_args()

    bag = args.bag or newest_bag()
    if not bag or not os.path.isfile(bag):
        sys.exit("no fused_drift bag found -- pass one explicitly "
                 "(did you launch with record:=true?)")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = load_paths(bag, np)
    if all(xy is None for _, xy, _ in traces):
        sys.exit("bag has no /drift_test/*_path messages -- did you lock the "
                 "anchor (drift_lock.py) and drive before stopping the bag?")

    fig, ax = plt.subplots(figsize=(9, 9))
    bg = load_map(args.map, np) if args.map else None
    if bg is not None:
        img, extent = bg
        ax.imshow(img, cmap="gray", extent=extent, origin="upper", alpha=0.6, zorder=0)
    elif args.map:
        print(f"[warn] could not load map {args.map}; plotting traces only")

    for name, xy, color in traces:
        if xy is None:
            print(f"[warn] no data for {name}")
            continue
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8, label=name, zorder=3)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, ms=6, mec="k", zorder=4)  # start

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(os.path.basename(bag) + "\n(circles = anchor/start)")

    out = args.out or (os.path.splitext(bag)[0] + "_traces.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
