# SPDX-License-Identifier: MIT
"""50 Hz Feetech STS3215 hardware bridge with dry-run backend and safety watchdog.

Safety watchdog enforces on every 50 Hz step:
  1. Hard joint-limit clamping (`joint_limits_rad +- margin`).
  2. Max torque-slew rate limiting (`max_torque_slew_nm_s * dt`) to prevent sudden jerks.
  3. Overload / excessive tracking-error auto-soften (`alpha -> soft_alpha`).
  4. NaN/Inf fail-safe & clean zero-torque release on `shutdown()` / context-manager exit.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol
import numpy as np

from .core import (
    DEFAULT_SPEC,
    GravCompTL,
    STS3215Spec,
    nm_to_sts3215_ticks,
    rad_to_sts3215_ticks,
    sts3215_ticks_to_rad,
)


class STS3215Bus(Protocol):
    def read_present_positions_ticks(self) -> np.ndarray: ...
    def read_present_loads_nm(self) -> np.ndarray: ...
    def write_torque_limits_ticks(self, reg_0x30: np.ndarray) -> None: ...
    def write_goal_positions_ticks(self, goal_ticks: np.ndarray) -> None: ...


class DryRunSTS3215Bus:
    """Deterministic in-memory 5-servo STS3215 bus for CI/offline verification (no hardware)."""

    def __init__(
        self,
        init_ticks: np.ndarray | None = None,
        external_load_ticks: np.ndarray | None = None,
        gravity_model: GravCompTL | None = None,
        spec: STS3215Spec = DEFAULT_SPEC,
    ):
        self.spec = spec
        self.gm = gravity_model
        self.present_ticks = np.full(5, 2048, dtype=np.int32) if init_ticks is None else np.asarray(init_ticks, dtype=np.int32).copy()
        self.goal_ticks = self.present_ticks.copy()
        self.torque_limit_reg = np.full(5, 1000, dtype=np.int32)
        self.external_load_ticks = np.zeros(5, dtype=float) if external_load_ticks is None else np.asarray(external_load_ticks, dtype=float).copy()
        self.write_count = 0

    def read_present_positions_ticks(self) -> np.ndarray:
        return self.present_ticks.copy()

    def read_present_loads_nm(self) -> np.ndarray:
        if self.gm is not None:
            q = sts3215_ticks_to_rad(self.present_ticks, spec=self.spec)
            return self.gm.compute_gravity_torque(q)
        return np.zeros(5, dtype=float)

    def write_torque_limits_ticks(self, reg_0x30: np.ndarray) -> None:
        self.torque_limit_reg = np.clip(np.asarray(reg_0x30, dtype=np.int32).reshape(5), 0, 1000)

    def write_goal_positions_ticks(self, goal_ticks: np.ndarray) -> None:
        self.goal_ticks = np.clip(np.asarray(goal_ticks, dtype=np.int32).reshape(5), 0, 4095)
        stiff = np.maximum(self.torque_limit_reg / 1000.0, 0.05)
        self.present_ticks = np.clip(
            np.rint(self.goal_ticks - self.external_load_ticks / stiff), 0, 4095
        ).astype(np.int32)
        self.write_count += 1


class FeetechSCServoBus:
    """Real Feetech STS3215 1 Mbps TTL serial bus adapter (requires `scservo_sdk`)."""

    def __init__(
        self,
        port: str = '/dev/ttyACM0',
        baudrate: int = 1_000_000,
        servo_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
        spec: STS3215Spec = DEFAULT_SPEC,
    ):
        try:
            import scservo_sdk as scs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                'FeetechSCServoBus requires `scservo_sdk` (`pip install feetech-servo-sdk`). '
                'Use `DryRunSTS3215Bus` for offline testing.'
            ) from exc
        self._scs = scs
        self.servo_ids = tuple(int(i) for i in servo_ids)
        self.spec = spec
        self.port_handler = scs.PortHandler(port)
        self.packet_handler = scs.PacketHandler(0)
        if not self.port_handler.openPort():
            raise RuntimeError(f'Failed to open STS3215 port {port}')
        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(f'Failed to set baudrate {baudrate} on {port}')

    def read_present_positions_ticks(self) -> np.ndarray:
        out = np.zeros(len(self.servo_ids), dtype=np.int32)
        for i, sid in enumerate(self.servo_ids):
            val, comm, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, sid, self.spec.present_position_addr
            )
            if comm != self._scs.COMM_SUCCESS:
                raise RuntimeError(f'STS3215 position read failed on servo {sid}: comm={comm}')
            out[i] = int(val) & 0x0FFF
        return out

    def read_present_loads_nm(self) -> np.ndarray:
        out = np.zeros(len(self.servo_ids), dtype=float)
        for i, sid in enumerate(self.servo_ids):
            raw, comm, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, sid, self.spec.present_load_addr
            )
            if comm != self._scs.COMM_SUCCESS:
                raise RuntimeError(f'STS3215 load read failed on servo {sid}: comm={comm}')
            mag = float(int(raw) & 0x03FF) / 1000.0
            sign = -1.0 if (int(raw) & 0x0400) else 1.0
            out[i] = sign * mag * self.spec.stall_torque_nm
        return out

    def write_torque_limits_ticks(self, reg_0x30: np.ndarray) -> None:
        vals = np.clip(np.asarray(reg_0x30, dtype=np.int32).reshape(len(self.servo_ids)), 0, self.spec.torque_limit_reg_max)
        for sid, v in zip(self.servo_ids, vals):
            self.packet_handler.write2ByteTxRx(self.port_handler, sid, self.spec.torque_limit_addr, int(v))

    def write_goal_positions_ticks(self, goal_ticks: np.ndarray) -> None:
        vals = np.clip(np.asarray(goal_ticks, dtype=np.int32).reshape(len(self.servo_ids)), 0, self.spec.encoder_ticks - 1)
        for sid, v in zip(self.servo_ids, vals):
            self.packet_handler.write2ByteTxRx(self.port_handler, sid, self.spec.goal_position_addr, int(v))

    def close(self) -> None:
        self.port_handler.closePort()


@dataclass(frozen=True)
class WatchdogConfig:
    joint_limit_margin_rad: float = 0.02
    max_torque_slew_nm_s: float = 20.0      # max torque-limit ramp rate (0.40 N·m per 20 ms step)
    overload_err_rad: float = 0.28          # auto-soften if |tracking_error| exceeds ~16 deg
    overload_soft_alpha: float = 0.12       # fallback alpha when overload triggers


@dataclass
class ServoState:
    q_quantized: np.ndarray
    q_dot_quantized: np.ndarray
    tracking_error: np.ndarray
    ee_rel_start: np.ndarray
    ee_twist_fk: np.ndarray
    overload_triggered: bool
    timestamp_s: float


class STS3215GravCompController:
    """50 Hz GravComp-TL servo controller with joint clamping, torque-slew limiter, and failsafe."""

    def __init__(
        self,
        bus: STS3215Bus,
        gravcomp: GravCompTL | None = None,
        zero_ticks: np.ndarray | int = 2048,
        dt: float = 0.02,
        watchdog: WatchdogConfig | None = None,
        spec: STS3215Spec = DEFAULT_SPEC,
    ):
        self.bus = bus
        self.gc = gravcomp if gravcomp is not None else GravCompTL(spec=spec)
        self.zero_ticks = zero_ticks
        self.dt = float(dt)
        self.wd = watchdog if watchdog is not None else WatchdogConfig()
        self.spec = spec
        jlim = self.gc.joint_limits_rad
        self.j_lo = jlim[:, 0] + self.wd.joint_limit_margin_rad
        self.j_hi = jlim[:, 1] - self.wd.joint_limit_margin_rad

        self._q_prev: np.ndarray | None = None
        self._q_cmd: np.ndarray | None = None
        self._lim_prev_nm: np.ndarray | None = None
        self._start_ee_fk: np.ndarray | None = None
        self.overload_count = 0

    def __enter__(self) -> 'STS3215GravCompController':
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def reset(self) -> ServoState:
        ticks = self.bus.read_present_positions_ticks()
        q = sts3215_ticks_to_rad(ticks, zero_ticks=self.zero_ticks, spec=self.spec)
        self._q_prev = q.copy()
        self._q_cmd = q.copy()
        lim0, _, _ = self.gc.compute_limits(q, 0.35)
        self._lim_prev_nm = lim0.copy()
        pos, _, _, _ = self.gc.compute_fk_jacobian(q)
        self._start_ee_fk = pos.copy()
        return ServoState(
            q_quantized=q,
            q_dot_quantized=np.zeros(5, dtype=float),
            tracking_error=np.zeros(5, dtype=float),
            ee_rel_start=np.zeros(3, dtype=float),
            ee_twist_fk=np.zeros(6, dtype=float),
            overload_triggered=False,
            timestamp_s=time.perf_counter(),
        )

    def read_state(self) -> ServoState:
        if self._q_prev is None or self._q_cmd is None or self._start_ee_fk is None:
            return self.reset()
        ticks = self.bus.read_present_positions_ticks()
        q = sts3215_ticks_to_rad(ticks, zero_ticks=self.zero_ticks, spec=self.spec)
        q_dot = (q - self._q_prev) / self.dt
        self._q_prev = q.copy()
        err = self._q_cmd - q
        overload = bool(np.any(np.abs(err) > self.wd.overload_err_rad))
        pos, _, jp, jr = self.gc.compute_fk_jacobian(q)
        return ServoState(
            q_quantized=q,
            q_dot_quantized=q_dot,
            tracking_error=err,
            ee_rel_start=pos - self._start_ee_fk,
            ee_twist_fk=np.concatenate([jp @ q_dot, jr @ q_dot]),
            overload_triggered=overload,
            timestamp_s=time.perf_counter(),
        )

    def command(self, q_target_rad: np.ndarray, alpha: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Write slew-limited GravComp-TL `0x30` ticks and clamped `0x2A` goal positions."""
        q_cur = self._q_prev if self._q_prev is not None else sts3215_ticks_to_rad(
            self.bus.read_present_positions_ticks(), zero_ticks=self.zero_ticks, spec=self.spec
        )
        q_in = np.asarray(q_target_rad, dtype=float).reshape(5)
        a_in = np.asarray(alpha, dtype=float)
        if not (np.all(np.isfinite(q_in)) and np.all(np.isfinite(a_in))):
            return self.emergency_soft_hold()

        # Overload auto-soften if tracking error exceeded threshold
        if self._q_cmd is not None and np.any(np.abs(self._q_cmd - q_cur) > self.wd.overload_err_rad):
            self.overload_count += 1
            a_in = np.minimum(a_in, self.wd.overload_soft_alpha)

        q_safe = np.clip(q_in, self.j_lo, self.j_hi)
        lim_raw_nm, _, _ = self.gc.compute_limits(q_cur, a_in)

        # Max torque-slew rate limiter (never block rapid softening on overload)
        if self._lim_prev_nm is not None:
            max_step = self.wd.max_torque_slew_nm_s * self.dt
            lim_nm = np.clip(lim_raw_nm, self._lim_prev_nm - max_step, self._lim_prev_nm + max_step)
        else:
            lim_nm = lim_raw_nm
        self._lim_prev_nm = lim_nm.copy()

        reg_0x30 = nm_to_sts3215_ticks(lim_nm, spec=self.spec)
        ticks = rad_to_sts3215_ticks(q_safe, zero_ticks=self.zero_ticks, spec=self.spec)
        self.bus.write_torque_limits_ticks(reg_0x30)
        self.bus.write_goal_positions_ticks(ticks)
        self._q_cmd = q_safe.copy()
        return q_safe, lim_nm, reg_0x30

    def emergency_soft_hold(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop all 5 joints immediately to gravity-compensated soft float at current pose."""
        ticks = self.bus.read_present_positions_ticks()
        q_cur = sts3215_ticks_to_rad(ticks, zero_ticks=self.zero_ticks, spec=self.spec)
        lim_nm, reg_0x30, _ = self.gc.compute_limits(q_cur, self.spec.alpha_min)
        self._lim_prev_nm = lim_nm.copy()
        self.bus.write_torque_limits_ticks(reg_0x30)
        self.bus.write_goal_positions_ticks(ticks)
        self._q_cmd = q_cur.copy()
        return q_cur, lim_nm, reg_0x30

    def shutdown(self) -> None:
        """Clean torque-release shutdown: soft hold then zero torque limit (0x30 = 0)."""
        self.emergency_soft_hold()
        self.bus.write_torque_limits_ticks(np.zeros(5, dtype=np.int32))
        if hasattr(self.bus, 'close'):
            self.bus.close()
