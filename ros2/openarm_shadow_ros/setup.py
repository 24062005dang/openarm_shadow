from glob import glob

from setuptools import setup

package_name = "openarm_shadow_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="To Minh Duc",
    maintainer_email="tominhducsp17@gmail.com",
    description="Teleop OpenArm bằng camera (openarm_shadow) qua ROS 2",
    license="Apache-2.0",
    entry_points={"console_scripts": ["teleop = openarm_shadow_ros.teleop_node:main"]},
)
