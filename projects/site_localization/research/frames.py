"""Conventional SE(2) frame composition; no estimation or truth dependence."""
import numpy as np


def compose(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c*b[..., 0] - s*b[..., 1],
                     a[..., 1] + s*b[..., 0] + c*b[..., 1],
                     np.arctan2(np.sin(a[..., 2]+b[..., 2]), np.cos(a[..., 2]+b[..., 2]))],axis=-1)


def inverse(a):
    x,y,t=np.asarray(a,dtype=float);c,s=np.cos(t),np.sin(t)
    return np.array([-c*x-s*y,s*x-c*y,-t])
