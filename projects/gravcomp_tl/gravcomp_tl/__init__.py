# SPDX-License-Identifier: MIT
"""GravComp-TL: Gravity-Compensated Servo Torque Limiting for low-cost SO-101 / STS3215 arms."""
from .core import (
    JOINT_NAMES,
    GravCompTL,
    STS3215Spec,
    compute_auroc,
    compute_tactile_snr,
    expand_alpha,
    nm_to_sts3215_ticks,
    rad_to_sts3215_ticks,
    sts3215_ticks_to_nm,
    sts3215_ticks_to_rad,
)
from .policy_runner import GravCompPolicyRunner, PolicyStepOutput
from .sts3215_driver import (
    DryRunSTS3215Bus,
    FeetechSCServoBus,
    STS3215Bus,
    STS3215GravCompController,
    ServoState,
    WatchdogConfig,
)

__all__ = [
    'JOINT_NAMES',
    'STS3215Spec',
    'GravCompTL',
    'expand_alpha',
    'nm_to_sts3215_ticks',
    'sts3215_ticks_to_nm',
    'rad_to_sts3215_ticks',
    'sts3215_ticks_to_rad',
    'compute_tactile_snr',
    'compute_auroc',
    'STS3215Bus',
    'DryRunSTS3215Bus',
    'FeetechSCServoBus',
    'ServoState',
    'WatchdogConfig',
    'STS3215GravCompController',
    'PolicyStepOutput',
    'GravCompPolicyRunner',
]
