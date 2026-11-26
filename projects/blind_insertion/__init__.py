"""Demo-free blind proprioceptive peg-in-hole insertion on the MuJoCo SO-101 arm."""
from .geometry import (
    ALL_SHAPES,
    SEEN_SHAPES,
    UNSEEN_SHAPES,
    AssemblyGeometrySpec,
    build_shape_polygon,
    run_all_drop_tests,
    run_drop_test,
)
from .env import (
    ALLOWED_OBS_KEYS,
    DT,
    ENCODER_QUANT_RAD,
    FORBIDDEN_PRIVILEGED_KEYS,
    PHYSICS_TIMESTEP,
    STS3215_MAX_TORQUE_NM,
    BlindInsertionEnv,
    InsertionTaskSpec,
    STS3215Config,
)
from .benchmarks import (
    build_regime_a_bolted,
    build_regime_b_unbolted,
    build_tight_clearance_suite,
)
from .controllers import (
    BlindSpiralController,
    NominalStraightDownController,
    PrivilegedOracleController,
    run_episode,
)

__all__ = [
    'ALL_SHAPES', 'SEEN_SHAPES', 'UNSEEN_SHAPES',
    'AssemblyGeometrySpec', 'build_shape_polygon', 'run_drop_test', 'run_all_drop_tests',
    'ALLOWED_OBS_KEYS', 'FORBIDDEN_PRIVILEGED_KEYS',
    'DT', 'PHYSICS_TIMESTEP', 'ENCODER_QUANT_RAD', 'STS3215_MAX_TORQUE_NM',
    'STS3215Config', 'InsertionTaskSpec', 'BlindInsertionEnv',
    'build_regime_a_bolted', 'build_regime_b_unbolted', 'build_tight_clearance_suite',
    'NominalStraightDownController', 'BlindSpiralController', 'PrivilegedOracleController',
    'run_episode',
]
