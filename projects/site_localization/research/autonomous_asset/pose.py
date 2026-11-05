"""Conventional planar PnP with chart fallback and explicit ambiguity rejection."""
import cv2
import numpy as np


def solve_pose(square, uv, k, config):
    square=np.asarray(square,dtype=np.float64);uv=np.asarray(uv,dtype=np.float64)
    audit=dict(raw_solutions=[],usable_solutions=[],fallback_used=False,
               ambiguity_error_gap_px=.25,ambiguity_normal_rad=float(np.deg2rad(15)),ambiguity_position_m=.15)
    if uv.shape!=(4,2) or not np.isfinite(uv).all() or not cv2.isContourConvex(uv.astype(np.float32)):
        audit['decision']='invalid_corners';return None,audit
    def solutions(chart=False):
        q=np.diag([1.,-1.,-1.]) if chart else np.eye(3)
        obj=square@q
        try:
            _,rvs,tvs,_=cv2.solvePnPGeneric(obj,uv,k,np.zeros(5),flags=cv2.SOLVEPNP_IPPE if chart else cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return []
        usable=[]
        for rv,tv in zip(rvs,tvs):
            if not(np.isfinite(rv).all() and np.isfinite(tv).all()):continue
            rot=cv2.Rodrigues(rv)[0]@q;translation=tv.ravel()
            # Compose matrices before converting the result back to rotation vector.
            rv=cv2.Rodrigues(rot)[0]
            projected=cv2.projectPoints(square,rv,tv,k,np.zeros(5))[0].reshape(4,2)
            rms=float(np.sqrt(np.mean(np.sum((projected-uv)**2,axis=1))))
            upright=float(np.linalg.norm(rot[:,1]-[0,-1,0]))
            front=bool(np.dot(rot[:,2],translation)<0)
            gauge=translation+rot@np.array([config['marker_to_gauge_right_m'],0,0])
            valid=bool(np.all((square@rot.T+translation)[:,2]>0) and front and upright<=.35
                       and np.isfinite(rms) and rms<=2 and .25<np.linalg.norm(gauge)<5)
            row=dict(rms_px=rms,rvec=rv.ravel().tolist(),tvec=translation.tolist(),
                gauge_camera=gauge.tolist(),normal_camera=rot[:,2].tolist(),upright_error=upright,
                solver='IPPE_chart' if chart else 'IPPE_SQUARE',accepted_geometry=valid)
            audit['raw_solutions'].append(row)
            if valid:usable.append(row)
        return usable
    # Keep square results for diagnosis; chart IPPE defines the usable pair.
    solutions()
    audit['fallback_used']=True;poses=solutions(chart=True)
    poses=sorted(poses,key=lambda p:p['rms_px']);audit['usable_solutions']=poses
    if not poses:audit['decision']='no_valid_pose';return None,audit
    best=poses[0]
    for other in poses[1:]:
        delta=float(np.linalg.norm(np.array(best['gauge_camera'])-other['gauge_camera']))
        normal=float(np.arccos(np.clip(np.dot(best['normal_camera'],other['normal_camera']),-1,1)))
        if other['rms_px']-best['rms_px']<=.25 and (delta>.15 or normal>np.deg2rad(15)):
            audit.update(decision='ambiguous_pose',normal_spread_rad=normal,position_spread_m=delta)
            return None,audit
    audit['decision']='usable_pose';return dict(best),audit
