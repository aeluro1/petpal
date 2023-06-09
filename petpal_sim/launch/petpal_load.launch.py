from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_arg = DeclareLaunchArgument("use_sim_time",
                                             default_value = "false",
                                             description = "Use simulation time")
    
    use_depth_cam = LaunchConfiguration("use_depth_cam")

    use_depth_cam_arg = DeclareLaunchArgument("use_depth_cam",
                                              default_value = "false",
                                              description = "Use depth camera instead of 2D RGB camera")

    xacro_path = [get_package_share_directory("petpal_description"), "/urdf", "/petpal.urdf.xacro"]
    robot_description_cmd = Command(["xacro ", *xacro_path, " use_depth_cam:=", use_depth_cam])

    params = {
        "robot_description": robot_description_cmd,
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
        use_depth_cam_arg,
        robot_state_publisher_node
    ])