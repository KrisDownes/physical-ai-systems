# SPDX-License-Identifier: MIT
"""Render 3 deterministic side-by-side EGL demo MP4 videos (Stiff left vs. GravComp-TL right).

Outputs in `projects/gravcomp_tl/media/`:
  1. `demo_1_ultratight_030mm.mp4` — Blind Insertion `OOD_UltraTight_0.30mm` (`p6-ut-01`, square 0.30mm)
  2. `demo_2_nonconvex_asym.mp4`   — Blind Insertion `OOD_NonConvex_Asym` (`p6-nc-02`, L-shape)
  3. `demo_3_ood_drawer.mp4`       — Blind Drawer `Split OOD` (`drw-ood-06`, -14 deg yaw, +15mm, light)
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault('MUJOCO_GL', 'egl')

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.phase6_ood_eval import (
    BASE_SPIRAL_ALPHA,
    PausableBaseSpiralController,
    Phase4ResidualBuilder,
    Phase5ResidualPPO,
    Phase6OODEnv,
    build_phase6_ood_suites,
)
from projects.blind_drawer.benchmarks import build_split_ood_light_height_angle
from projects.blind_drawer.controllers import NominalStraightBackController
from projects.blind_drawer.env import BlindDrawerEnv
from projects.blind_drawer.phase1_policy import CRITIC_PRIV_DIM, GRU_HIDDEN_DIM, DrawerObsBuilder, DrawerResidualGRUPPO

MEDIA_DIR = Path(__file__).resolve().parents[1] / 'media'
INS_CKPT_DIR = REPO_ROOT / 'projects' / 'blind_insertion' / 'phase5_checkpoints'
DRW_P1_CKPT_DIR = REPO_ROOT / 'projects' / 'blind_drawer' / 'phase1_checkpoints'
DRW_P2_CKPT_DIR = REPO_ROOT / 'projects' / 'blind_drawer' / 'phase2_checkpoints'

PANEL_W, PANEL_H = 480, 360
HEADER_H, FOOTER_H = 44, 68
FULL_W = PANEL_W * 2
FULL_H = HEADER_H + PANEL_H + FOOTER_H


def _load_font(size: int):
    for p in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(15)
FONT_SUB = _load_font(12)
FONT_SMALL = _load_font(11)


def _make_camera(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _record_insertion_rollout(ood_task, ckpt_path: Path, mode: str, cam: mujoco.MjvCamera) -> list[dict]:
    env = Phase6OODEnv(ood_task=ood_task, render=False)
    renderer = mujoco.Renderer(env.model, height=PANEL_H, width=PANEL_W)
    frames: list[dict] = []
    try:
        obs, info = env.reset()
        base_ctrl = PausableBaseSpiralController(torque_scale=1.0)
        base_ctrl.reset(obs)
        builder = Phase4ResidualBuilder(mode=mode)
        pol = Phase5ResidualPPO.load(ckpt_path)
        h = pol.init_hidden()

        terminated = truncated = False
        cur_alpha = (1.0, 1.0) if mode == 'res_pos_only' else (BASE_SPIRAL_ALPHA, BASE_SPIRAL_ALPHA)

        while not (terminated or truncated):
            renderer.update_scene(env.data, camera=cam)
            rgb = renderer.render().copy()
            priv = info['privileged']
            frames.append({
                'rgb': rgb,
                'force_N': float(priv.get('peak_contact_force_N', info.get('episode_peak_contact_force_N', 0.0))),
                'peak_force_N': float(info.get('episode_peak_contact_force_N', 0.0)),
                'disp_mm': float(priv.get('socket_disp_mm', 0.0)),
                'progress_str': f"depth {float(priv.get('peg_depth', 0.0))*1000.0:4.1f} mm",
                'alpha': cur_alpha,
                'alpha_labels': ('lat', 'arm'),
                'status': 'SEATED' if info['success'] else str(info.get('failure_mode') or 'in_progress'),
                't_s': float(info['time_s']),
            })

            base_dxyz = base_ctrl.act_base(obs)
            base_r_m = min((base_ctrl.pitch_m / (2.0 * math.pi)) * base_ctrl.theta, base_ctrl.max_radius_m)
            x_vec = builder.step(obs, base_dxyz, base_ctrl.phase, base_r_m, base_ctrl.insert_stall_steps)
            env_act, _, g_spiral, alpha_pair, _, h, _, _ = pol.act_step_p4(x_vec, h, base_dxyz, deterministic=True)
            base_ctrl.set_spiral_gate(g_spiral)
            builder.set_prev_controls(g_spiral, alpha_pair)
            cur_alpha = (1.0, 1.0) if alpha_pair is None else (float(alpha_pair[0]), float(alpha_pair[1]))
            obs, _, terminated, truncated, info = env.step(env_act)

        # Hold final frame briefly
        renderer.update_scene(env.data, camera=cam)
        rgb = renderer.render().copy()
        priv = info['privileged']
        final_rec = {
            'rgb': rgb,
            'force_N': float(priv.get('peak_contact_force_N', info.get('episode_peak_contact_force_N', 0.0))),
            'peak_force_N': float(info.get('episode_peak_contact_force_N', 0.0)),
            'disp_mm': float(priv.get('socket_disp_mm', 0.0)),
            'progress_str': f"depth {float(priv.get('peg_depth', 0.0))*1000.0:4.1f} mm",
            'alpha': cur_alpha,
            'alpha_labels': ('lat', 'arm'),
            'status': 'SEATED' if info['success'] else str(info.get('failure_mode') or 'failed').upper(),
            't_s': float(info['time_s']),
            'success': bool(info['success']),
        }
        frames.append(final_rec)
    finally:
        renderer.close()
    return frames


def _record_drawer_rollout(task, ckpt_path: Path, mode: str, cam: mujoco.MjvCamera) -> list[dict]:
    env = BlindDrawerEnv(task=task, render=False)
    renderer = mujoco.Renderer(env.model, height=PANEL_H, width=PANEL_W)
    frames: list[dict] = []
    try:
        obs, info = env.reset()
        base_ctrl = NominalStraightBackController(mode='stiff', pull_speed_m_s=0.028, down_preload_m=0.0012)
        base_ctrl.reset(obs)
        builder = DrawerObsBuilder(mode=mode)
        builder.reset()
        pol = DrawerResidualGRUPPO.load(ckpt_path)
        h = pol.init_hidden()

        terminated = truncated = False
        is_5dof = (mode == 'res_gravcomp_tl_5dof')
        cur_alpha = (1.0, 1.0, 1.0, 1.0, 1.0) if not is_5dof else (0.30, 0.45, 0.45, 0.35, 0.35)
        alabels = ('pan', 'lift', 'elb', 'wflx', 'wrol')

        while not (terminated or truncated):
            renderer.update_scene(env.data, camera=cam)
            rgb = renderer.render().copy()
            priv = info['privileged']
            frames.append({
                'rgb': rgb,
                'force_N': float(priv['peak_contact_force_N']),
                'peak_force_N': float(info['episode_peak_contact_force_N']),
                'disp_mm': float(priv['cabinet_disp_mm']),
                'progress_str': f"open {float(priv['drawer_travel_mm']):4.1f}/70 mm",
                'alpha': cur_alpha,
                'alpha_labels': alabels,
                'status': 'OPENED' if info['success'] else str(info.get('failure_mode') or 'in_progress'),
                't_s': float(info['time_s']),
            })

            base_a = base_ctrl.act(obs)[:3]
            x_vec = builder.build(obs, base_a)
            env_act, _, a_out, _, h, _, _ = pol.act_step(x_vec, h, base_a, deterministic=True)
            builder.set_prev_alpha(a_out)
            if a_out is not None:
                cur_alpha = tuple(float(v) for v in a_out)
            obs, _, terminated, truncated, info = env.step(env_act)

        renderer.update_scene(env.data, camera=cam)
        rgb = renderer.render().copy()
        priv = info['privileged']
        frames.append({
            'rgb': rgb,
            'force_N': float(priv['peak_contact_force_N']),
            'peak_force_N': float(info['episode_peak_contact_force_N']),
            'disp_mm': float(priv['cabinet_disp_mm']),
            'progress_str': f"open {float(priv['drawer_travel_mm']):4.1f}/70 mm",
            'alpha': cur_alpha,
            'alpha_labels': alabels,
            'status': 'OPENED' if info['success'] else str(info['failure_mode']).upper(),
            't_s': float(info['time_s']),
            'success': bool(info['success']),
        })
    finally:
        renderer.close()
    return frames


def _compose_frame(
    title: str,
    left_label: str,
    right_label: str,
    left_rec: dict,
    right_rec: dict,
    disp_label: str,
) -> np.ndarray:
    canvas = Image.new('RGB', (FULL_W, FULL_H), (18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Header bar
    draw.rectangle([0, 0, FULL_W, HEADER_H], fill=(24, 30, 38))
    draw.text((14, 12), title, fill=(238, 242, 246), font=FONT_TITLE)

    # Paste left and right RGB panels
    canvas.paste(Image.fromarray(left_rec['rgb']), (0, HEADER_H))
    canvas.paste(Image.fromarray(right_rec['rgb']), (PANEL_W, HEADER_H))
    draw.line([(PANEL_W, HEADER_H), (PANEL_W, FULL_H)], fill=(55, 65, 80), width=2)

    # Side badges + footer HUD
    for idx, (lbl, rec, accent) in enumerate([
        (left_label, left_rec, (148, 163, 184)),
        (right_label, right_rec, (42, 185, 165)),
    ]):
        x0 = idx * PANEL_W
        # Top overlay label on video
        draw.rectangle([x0 + 8, HEADER_H + 8, x0 + 235, HEADER_H + 30], fill=(15, 20, 26))
        draw.text((x0 + 14, HEADER_H + 12), lbl, fill=accent, font=FONT_SUB)

        # Status pill (top-right of panel)
        st = rec['status']
        if st in ('SEATED', 'OPENED', 'SUCCESS'):
            st_bg, st_fg = (22, 128, 72), (240, 255, 245)
        elif st != 'in_progress':
            st_bg, st_fg = (168, 42, 42), (255, 240, 240)
        else:
            st_bg, st_fg = (36, 44, 56), (200, 210, 220)
        draw.rectangle([x0 + PANEL_W - 155, HEADER_H + 8, x0 + PANEL_W - 8, HEADER_H + 30], fill=st_bg)
        draw.text((x0 + PANEL_W - 148, HEADER_H + 12), f"{st[:18]}", fill=st_fg, font=FONT_SMALL)

        # Footer metrics
        fy = HEADER_H + PANEL_H + 8
        hud_line = (
            f"t={rec['t_s']:4.2f}s | Peak F={rec['peak_force_N']:4.1f} N | "
            f"{disp_label}={rec['disp_mm']:4.1f} mm | {rec['progress_str']}"
        )
        draw.text((x0 + 12, fy), hud_line, fill=(225, 232, 240), font=FONT_SMALL)

        # Live alpha_i bar strip
        by = fy + 24
        alphas = rec['alpha']
        alabels = rec['alpha_labels']
        n_a = len(alphas)
        bar_area_w = PANEL_W - 24
        slot_w = bar_area_w // n_a
        for j, (albl, aval) in enumerate(zip(alabels, alphas)):
            bx = x0 + 12 + j * slot_w
            bw = slot_w - 10
            draw.text((bx, by), f"{albl}:{aval:.2f}", fill=(185, 196, 210), font=FONT_SMALL)
            draw.rectangle([bx, by + 15, bx + bw, by + 24], fill=(42, 50, 62))
            fill_w = int(np.clip(aval, 0.0, 1.0) * bw)
            bar_col = (120, 138, 156) if idx == 0 else (42, 185, 165) if aval < 0.40 else (231, 111, 81)
            draw.rectangle([bx, by + 15, bx + fill_w, by + 24], fill=bar_col)

    return np.asarray(canvas, dtype=np.uint8)


def _write_mp4_ffmpeg(frames_rgb: list[np.ndarray], out_path: Path, fps: int = 30) -> None:
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{FULL_W}x{FULL_H}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-',
        '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'fast',
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames_rgb:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    assert proc.stderr is not None
    err = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg failed for {out_path}: {err.decode(errors="ignore")}')


def _stitch_and_save(
    out_path: Path,
    title: str,
    left_label: str,
    right_label: str,
    left_frames: list[dict],
    right_frames: list[dict],
    disp_label: str,
    fps: int = 30,
    stride: int = 1,
) -> dict:
    # Align lengths (hold final frame of shorter rollout + 1.0 s end pause)
    n_max = max(len(left_frames), len(right_frames))
    idxs = list(range(0, n_max, stride))
    if idxs[-1] != n_max - 1:
        idxs.append(n_max - 1)
    composed = []
    for k in idxs:
        lr = left_frames[min(k, len(left_frames) - 1)]
        rr = right_frames[min(k, len(right_frames) - 1)]
        composed.append(_compose_frame(title, left_label, right_label, lr, rr, disp_label))
    # Hold final comparison frame for 1.0 s
    for _ in range(fps):
        composed.append(composed[-1])
    _write_mp4_ffmpeg(composed, out_path, fps=fps)
    return {
        'path': str(out_path),
        'n_frames': len(composed),
        'duration_s': round(len(composed) / float(fps), 2),
        'left_final': left_frames[-1]['status'],
        'right_final': right_frames[-1]['status'],
        'left_peak_force_N': round(left_frames[-1]['peak_force_N'], 1),
        'right_peak_force_N': round(right_frames[-1]['peak_force_N'], 1),
    }


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    r1_ckpt = INS_CKPT_DIR / 'R1_v2_Res_PosOnly_seed101.npz'
    r2_ckpt = INS_CKPT_DIR / 'R2_v2_Res_GravComp_TL_seed101.npz'
    d1_ckpt = DRW_P1_CKPT_DIR / 'D1_Res_Stiff_seed101.npz'
    d2_ckpt = DRW_P2_CKPT_DIR / 'D2_PerJoint_seed101.npz'

    cam_ins = _make_camera(lookat=(0.25, 0.0, 0.035), distance=0.28, azimuth=132.0, elevation=-24.0)
    cam_drw = _make_camera(lookat=(0.26, 0.0, 0.055), distance=0.42, azimuth=125.0, elevation=-25.0)

    p6_suites = build_phase6_ood_suites()

    # Demo 1: OOD_UltraTight_0.30mm (p6-ut-01, square 0.30mm)
    ut_task = p6_suites['OOD_UltraTight_0.30mm'][1]
    l1 = _record_insertion_rollout(ut_task, r1_ckpt, 'res_pos_only', cam_ins)
    r1 = _record_insertion_rollout(ut_task, r2_ckpt, 'res_gravcomp_tl', cam_ins)
    s1 = _stitch_and_save(
        MEDIA_DIR / 'demo_1_ultratight_030mm.mp4',
        'Demo 1: Blind Insertion — OOD Ultra-Tight 0.30 mm (< 1 STS3215 Encoder Count)',
        'LEFT: R1_v2 Stiff (ts = 1.0)',
        'RIGHT: R2_v2 GravComp-TL',
        l1, r1, disp_label='Sock disp', fps=30, stride=2,
    )
    print('Rendered Demo 1:', s1, flush=True)

    # Demo 2: OOD_NonConvex_Asym (p6-nc-02, L-shape)
    nc_task = p6_suites['OOD_NonConvex_Asym'][2]
    l2 = _record_insertion_rollout(nc_task, r1_ckpt, 'res_pos_only', cam_ins)
    r2 = _record_insertion_rollout(nc_task, r2_ckpt, 'res_gravcomp_tl', cam_ins)
    s2 = _stitch_and_save(
        MEDIA_DIR / 'demo_2_nonconvex_asym.mp4',
        'Demo 2: Blind Insertion — Zero-Shot Non-Convex L-Shape Socket',
        'LEFT: R1_v2 Stiff (ts = 1.0)',
        'RIGHT: R2_v2 GravComp-TL',
        l2, r2, disp_label='Sock disp', fps=30, stride=2,
    )
    print('Rendered Demo 2:', s2, flush=True)

    # Demo 3: Blind Drawer Split OOD (drw-ood-06, -14 deg yaw, +15mm height, light cabinet)
    drw_task = build_split_ood_light_height_angle()[6]
    l3 = _record_drawer_rollout(drw_task, d1_ckpt, 'res_pos_only', cam_drw)
    r3 = _record_drawer_rollout(drw_task, d2_ckpt, 'res_gravcomp_tl_5dof', cam_drw)
    s3 = _stitch_and_save(
        MEDIA_DIR / 'demo_3_ood_drawer.mp4',
        'Demo 3: Blind Drawer — Zero-Shot OOD (Light 300g, μ=0.35, +15mm, -14° Yaw)',
        'LEFT: D1_Res_Stiff (ts = 1.0)',
        'RIGHT: D2_PerJoint (5-DoF GravComp-TL)',
        l3, r3, disp_label='Cab disp', fps=30, stride=1,
    )
    print('Rendered Demo 3:', s3, flush=True)


if __name__ == '__main__':
    main()
