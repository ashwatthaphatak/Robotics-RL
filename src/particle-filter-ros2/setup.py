from setuptools import setup, find_packages
import os
from glob import glob

package_name = "particle-filter-ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src"),          # <-- finds particle_filter_ros2
    package_dir={"": "src"},                      # <-- tells setuptools that code is in src/
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aaphatak",
    maintainer_email="ashwatthap@gmail.com",
    description="Particle filter project",
    license="TODO",
    entry_points={
        "console_scripts": [
            "motion_model = particle_filter_ros2.motion_model:main",
            "sensor_model = particle_filter_ros2.sensor_model:main",
            "particle_filter = particle_filter_ros2.particle_filter:main",
        ],
    },
)
