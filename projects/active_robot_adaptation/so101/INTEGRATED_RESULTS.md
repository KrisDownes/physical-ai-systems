# Integrated SO-101 pilot results

The [controller and observability assumptions](INTEGRATED_METHOD.md) define the camera model, marker geometry, tool disturbance and coordinate handling. Results below are simulated and deterministic. The [frozen configuration](frozen_integrated.json) specified 40 independent episodes before execution: two new starts and object positions, two camera translations, two wrist-frame tool offsets, five conditions and four methods. All start with reset adaptation memory and the same 200-action/8-rad movement budgets. Camera and tool values vary together by case, so this does not measure every cross-combination.

| Condition | Fixed | Calibration only | Motion only | Integrated |
|---|---:|---:|---:|---:|
| No change | 2/2 | 2/2 | 2/2 | 2/2 |
| Camera translation only | 1/2 | 2/2 | 1/2 | 2/2 |
| Tool offset only | 2/2 | 2/2 | 2/2 | 2/2 |
| Camera + tool, table references | 1/2 | 2/2 | 1/2 | 2/2 |
| Camera + tool, no table references | 1/2 | 2/2 | 1/2 | 2/2 |

The fixed and motion-only failures were grasp failures. With the available markers and tested offsets, **calibration alone matched the integrated controller**. Motion adaptation did not improve success in these cases; the nominal controller also handled both isolated tool offsets. With no table references, both calibrated methods used a gripper-marker fallback that explicitly assumes nominal geometry. Its position estimates were biased by the unknown offset, but remained within the grasp tolerance in these two cases. This is not evidence that the camera and mounting errors were separately identified.

The table markers were preferred when consistent. For the -5 mm tool-offset case, the gripper estimate differed from the table estimate by about 5.1–5.2 mm and was rejected as inconsistent. The +4 mm case differed by about 3.6–3.7 mm and was rejected only because the table was preferred; a threshold miss is not proof that the tool geometry is correct. If table references are unavailable, the selector has no such check and logs the assumption behind the fallback. The actual marker and jaws move together; no hidden tool offset or true camera transform is supplied to the controller.

The 40 trials used 3,144 actions and 334.4 simulated seconds in total; measured episode wall times summed to about 130.9 s, including rendering and evaluation. Timed controller computation summed to about 0.343 s, excluding rendering and other program overhead. The largest episode used 88 actions and 2.705 rad of accumulated joint movement. Per-condition median action counts, simulated time, movement, computation and FK-to-waypoint tracking error are in `summary.json`; per-episode grasp alignment, calibration decisions, accepted/rejected adaptation updates and traces are retained. No sampled joint-limit violations or prohibited contacts were recorded in the **66,880 audited physics steps**, including reset settling. These simulation checks are not hardware safety evidence.

Separate oracle-FK development diagnostics succeeded with both +6 mm and -6 mm wrist-frame offsets. Those diagnostics expose true geometry to the controller and are **excluded** from the comparison. The final comparison used +4 mm and -5 mm offsets and different starts/targets. Two cases per condition are far too few for a general reliability or method-ranking claim; marker visibility, exact camera orientation, known object height and approximate nominal kinematics remain idealized. The + and - camera and tool values are paired by case rather than fully crossed. No physical robot was used.

The existing online-versus-fixed case and the pitch-Jacobian implementation fix are described in `INTEGRATED_METHOD.md`. The final comparison was run with the corrected pitch response. After the run, calibration-decision logging was made explicit for accepted table estimates and failure exits, and run metadata was enhanced to snapshot sources. These changes do not alter action selection: one paired re-execution of the combined-condition integrated case had identical 73-action joint traces. The exact source actually used for the comparison is preserved in its `source_snapshot.json.gz`, with hashes and configuration in `metadata.json`. Historical table acceptance reasons in `calibration_decisions.json` were derived from the recorded selector branch, not retroactively inserted into raw episode records.

No active probe or new RL training was performed. A single gripper marker and nominal FK can confuse camera translation with tool mounting error when table references are absent, but that ambiguity did not produce a reproducible failure at the tested magnitudes. The next justified experiment is a predeclared observability check: use ordinary task trajectories at several wrist orientations and compare whether the table-calibrated marker residual changes as expected for a wrist-frame offset. Only if passive observations leave materially different grasp corrections plausible should a fixed, bounded orientation probe be compared with passive adaptation under the same budget. Do not infer that a probe is necessary from this two-case result.

## Reproduce

From `projects/active_robot_adaptation`, after setup described in `README.md`:

```sh
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/discrepancy.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/discrepancy_replay.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/geometry_oracle.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/integrated_evaluation.py --config so101/frozen_integrated.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test_*.py' -v
```

Each run writes a new unique directory. A stopped evaluation can resume with the same `--config` and `--resume RUN_DIR` after configuration and source-hash checks. Use `so101/summarize_integrated.py --run RUN_DIR` for compact per-episode logs, explicit calibration decisions and lossless compressed traces; `so101/plot_integrated.py --run RUN_DIR` uses the existing Matplotlib environment for the plot.
