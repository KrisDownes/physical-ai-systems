"""LILocBench ROS1 -> explicit, independently auditable offline AMCL inputs.

No evaluation truth is included in frames.txt. Only the initial tracking prior
is exported separately. Dataset: Trekel et al., IROS 2025, CC BY-NC-SA.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import yaml


def yaw(q):
    return np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def interpolate(times, poses, query, max_gap=.15):
    """Interpolate only bracketed data, returning support mask; yaw is circular."""
    times, poses, query = map(np.asarray, (times, poses, query))
    if np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be strictly increasing')
    hi = np.searchsorted(times, query, side='right')
    lo = np.clip(hi-1, 0, len(times)-2)
    hi = lo+1
    valid = (query >= times[0]) & (query <= times[-1]) & ((times[hi]-times[lo]) <= max_gap)
    p = poses.copy()
    p[:, 2] = np.unwrap(p[:, 2])
    value = np.column_stack([np.interp(query, times, p[:, j]) for j in range(3)])
    value[:, 2] = np.arctan2(np.sin(value[:, 2]), np.cos(value[:, 2]))
    return value, valid


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sequence', choices=['dynamics_0', 'lt_changes_0'])
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    data = root/'data'
    out = data/args.sequence
    out.mkdir(exist_ok=True)
    spec = yaml.safe_load((data/'map/map_office.yaml').read_text())
    assert spec['origin'][2] == 0 and spec['negate'] == 0
    image = np.asarray(Image.open(data/'map'/spec['image']).convert('L'))
    occupied_prob = (255-image.astype(float))/255
    grid = np.where(occupied_prob > spec['occupied_thresh'], 1,
                    np.where(occupied_prob < spec['free_thresh'], -1, 0))
    grid = np.flipud(grid)  # image row0 is top, map row0 is bottom
    with (out/'map.txt').open('w') as f:
        f.write(f'{grid.shape[1]} {grid.shape[0]} {spec["resolution"]} {spec["origin"][0]} {spec["origin"][1]}\n')
        np.savetxt(f, grid, fmt='%d')
    store = get_typestore(Stores.ROS1_NOETIC)
    scans, odom, extrinsics = [], [], []
    with Reader(data/f'{args.sequence}.bag') as reader:
        for c in reader.connections:
            store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
        conns = [c for c in reader.connections if c.topic in
                 ['/laser_scan_front/scan', '/dingo_velocity_controller/odom', '/tf', '/tf_static']]
        for c, _, raw in reader.messages(connections=conns):
            msg = store.deserialize_ros1(raw, c.msgtype)
            if c.topic == '/laser_scan_front/scan':
                scans.append(msg)
            elif c.topic == '/dingo_velocity_controller/odom':
                pose = msg.pose.pose
                odom.append([stamp(msg.header), pose.position.x, pose.position.y, yaw(pose.orientation)])
            else:
                for tf in msg.transforms:
                    if tf.header.frame_id == 'base_link' and tf.child_frame_id == 'scan_front_link':
                        tr = tf.transform
                        extrinsics.append([tr.translation.x, tr.translation.y, yaw(tr.rotation)])
    if not extrinsics or not np.allclose(extrinsics, extrinsics[0], atol=1e-9):
        raise ValueError('Missing or changing base-to-laser transform')
    odom = np.array(odom)
    times = np.array([stamp(s.header) for s in scans])
    if np.any(np.diff(times) <= 0): raise ValueError('unordered scans')
    odom_at, supported = interpolate(odom[:, 0], odom[:, 1:], times)
    gt = np.loadtxt(data/f'{args.sequence}_gt.txt')
    # TUM quaternion x y z w. Full yaw projection, not 2*atan(z/w).
    gt_yaw = np.arctan2(2*(gt[:, 7]*gt[:, 6]+gt[:, 4]*gt[:, 5]),
                       1-2*(gt[:, 5]**2+gt[:, 6]**2))
    truth, gt_supported = interpolate(gt[:, 0], np.c_[gt[:, 1:3], gt_yaw], times)
    # Do not filter estimator input on evaluation truth availability.
    keep = np.flatnonzero(supported)
    initial = keep[0]
    if not gt_supported[initial]: raise ValueError('No supported initial prior')
    with (out/'frames.txt').open('w') as f:
        f.write(' '.join(map(str, [*extrinsics[0], 30.0]))+'\n')
        for i in keep:
            s = scans[i]
            ranges = np.array(s.ranges, dtype=float)
            ranges[(~np.isfinite(ranges)) | (ranges < s.range_min) | (ranges >= 30)] = 30
            angles = s.angle_min + np.arange(len(ranges))*s.angle_increment
            values = np.ravel(np.c_[ranges, angles])
            f.write(' '.join(map(str, [times[i], *odom_at[i], len(ranges), *values]))+'\n')
    np.savez_compressed(out/'judge.npz', times=times[keep], truth=truth[keep],
                        supported=gt_supported[keep], odom=odom_at[keep])
    manifest = dict(sequence=args.sequence, split='development',
                    initial_pose=truth[initial].tolist(), total_scans=len(scans),
                    retained_scans=len(keep), odom_unsupported=int((~supported).sum()),
                    truth_unsupported_retained=int((~gt_supported[keep]).sum()),
                    laser_extrinsic=extrinsics[0], max_interpolation_gap_s=.15,
                    range_cap_m=30, deskew=False,
                    input_hashes={str(x.relative_to(root)):sha(x) for x in
                                  [data/f'{args.sequence}.bag', data/f'{args.sequence}_gt.txt',
                                   data/'map_office.zip', out/'frames.txt', out/'map.txt']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
