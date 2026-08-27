from setuptools import find_packages, setup

package_name = 'rover_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krisd',
    maintainer_email='krisdownes@yahoo.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'obstacle_guard = rover_control.obstacle_guard:main',
            'path_follower = rover_control.path_follower:main',
        ],
    },
)
