# Site localization and inspection

Can a mobile robot map a changed indoor site, find a moved inspection asset, and verify an inspection without being given the asset's new position?

In simulation, the robot explores with lidar, hands its estimated pose and acquired map to localization, searches for an AprilTag on the asset, approaches it, and reads a gauge. A separate judge checks the physical outcome.

On one held-out connected layout, three search policies each completed all three target-present missions. All three withheld completion when the target was absent. Across the 12 missions there were no contacts or false completion claims. The camera-aware policy did not show a completion advantage over nearest or map-visibility search. These results cover one simulated layout; they do not establish field reliability.

The simulated camera uses geometric projection and a fixed gauge image. Real-camera detection and gauge-reading transfer remain unverified.

[Experiment and results](docs/inspection_aware/result.md) · [Camera audit and reproduction commands](docs/camera_audit/result.md)
