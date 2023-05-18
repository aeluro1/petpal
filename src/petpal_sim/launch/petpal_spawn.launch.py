from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    pkg_path = Path(get_package_share_directory("petpal_sim"))

    petpal_load_launch = IncludeLaunchDescription(
        # PythonLaunchDescriptionSource([str(pkg_path), "/launch", "/petpal_load.launch.py"]),
        PythonLaunchDescriptionSource([FindPackageShare("petpal_sim"), "/launch", "/petpal_load.launch.py"]),
        launch_arguments = {"use_sim_time": "true"}.items()
    )

    gazebo_launch = IncludeLaunchDescription(
        # PythonLaunchDescriptionSource([str(Path(get_package_share_directory("gazebo_ros")) / "launch" / "gazebo.launch.py")])
        PythonLaunchDescriptionSource([FindPackageShare("gazebo_ros"), "/launch", "/gazebo.launch.py"])
    )

    spawn_node = Node(
        package = "gazebo_ros",
        executable = "spawn_entity.py",
        arguments = ["-topic", "robot_description", "-entity", "petpal"],
        output = "screen"
    )

    rviz_node = Node(
        package = "rviz2",
        executable = "rviz2",
        arguments = ["-d", f"{pkg_path / 'config' / 'config.rviz'}"],
        output = "screen"
    )

    # Start new shell to run teleop?
    #   ros2 run teleop_twist_keyboard teleop_twist_keyboard
    # Alternatively, use teleop_twist_joy w/ joy_node
    
    return LaunchDescription([
        petpal_load_launch,
        gazebo_launch,
        spawn_node,
        rviz_node
    ])