"""Independent ceiling-fiducial judge; never align fitted trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0]+c*b[..., 0]-s*b[..., 1],
                     a[..., 1]+s*b[..., 0]+c*b[..., 1],
                     a[..., 2]+b[..., 2]], axis=-1)


def inverse(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-c*a[..., 0]-s*a[..., 1], s*a[..., 0]-c*a[..., 1], -a[..., 2]], axis=-1)


def errors(est, truth):
    position = np.linalg.norm(est[:, :2]-truth[:, :2], axis=1)
    d = est[:, 2]-truth[:, 2]
    return position, np.abs(np.arctan2(np.sin(d), np.cos(d)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sequence'); p.add_argument('estimates', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    judge = np.load(root/'data'/args.sequence/'judge.npz')
    est = np.genfromtxt(args.estimates, delimiter=',', names=True)
    if len(est) != len(judge['times']) or not np.allclose(est['timestamp'], judge['times'], rtol=0, atol=1e-5):
        raise ValueError('Estimator output must account for every retained input scan in order')
    if not np.isfinite(np.c_[est['x'],est['y'],est['yaw']]).all(): raise ValueError('nonfinite poses')
    mask = judge['supported']
    truth = judge['truth']
    odom = compose(compose(truth[0], inverse(judge['odom'][0])), judge['odom'])
    poses = np.c_[est['x'],est['y'],est['yaw']]
    summary = dict(sequence=args.sequence, split='development', scans=len(est),
                   judged_scans=int(mask.sum()), unjudged_scans=int((~mask).sum()),
                   thresholds='engineering diagnostic, not customer acceptance', methods={})
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(truth[mask,0],truth[mask,1],label='ceiling-fiducial reference',color='black')
    for name, xy in [('initially_aligned_odometry',odom),('stock_amcl',poses)]:
        pos, angle = errors(xy[mask],truth[mask])
        summary['methods'][name] = dict(position_rmse_m=float(np.sqrt(np.mean(pos**2))),
            position_p95_m=float(np.quantile(pos,.95)),position_max_m=float(pos.max()),
            yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angle**2)))),
            fraction_over_025m=float(np.mean(pos>.25)),
            fraction_over_05m=float(np.mean(pos>.5)))
        ax[0].plot(xy[:,0],xy[:,1],label=name,alpha=.75)
        ax[1].plot(judge['times'][mask]-judge['times'][0],pos,label=name)
    ax[0].axis('equal'); ax[0].set(xlabel='map x [m]',ylabel='map y [m]')
    ax[1].set(xlabel='elapsed sensor time [s]',ylabel='position error [m]')
    for a in ax:a.legend();a.grid(alpha=.3)
    fig.suptitle(args.sequence+' — development replay; no closed-loop claim')
    fig.tight_layout();fig.savefig(args.estimates.with_suffix('.png'),dpi=150)
    args.estimates.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
