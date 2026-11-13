def forward_kinematics(joint_angles, link_lengths):
    """
    Calculate the forward kinematics for a planar robotic arm.

    Parameters:
    joint_angles (list): A list of joint angles in radians.
    link_lengths (list): A list of link lengths.

    Returns:
    tuple: The (x, y) position of the end effector.
    """
    x = 0
    y = 0
    theta = 0

    for i in range(len(joint_angles)):
        theta += joint_angles[i]
        x += link_lengths[i] * cos(theta)
        y += link_lengths[i] * sin(theta)

    return (x, y)