from setuptools import find_packages, setup


package_name = 'visual_rover_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['mcp>=1.25,<2', 'Pillow>=10', 'setuptools'],
    zip_safe=True,
    maintainer='krisd',
    maintainer_email='krisdownes@yahoo.com',
    description='Bounded robot-facing tools for the visual rover agent.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'agent_executor = visual_rover_agent.node:main',
            'rover_driver_mcp = visual_rover_agent.driver_interface:main',
        ],
    },
)
