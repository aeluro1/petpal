from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

import xacro

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_arg = DeclareLaunchArgument("use_sim_time",
                                             default_value = "false",
                                             description = "Use simulation time")

    pkg_path = Path(get_package_share_directory("petpal_description"))
    xacro_path = pkg_path / "urdf" / "petpal.urdf.xacro"
    robot_description_config = xacro.process_file(xacro_path)

    params = {
        "robot_description": robot_description_config.toxml(),
        "use_sim_time": use_sim_time
    }
    robot_state_publisher_node = Node(
        package = "robot_state_publisher",
        executable = "robot_state_publisher",
        output = "screen",
        parameters = [params]
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher_node
    ])