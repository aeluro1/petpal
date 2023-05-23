// #include <urdf/model.h>
// #include "ros/ros.h"

// int main(int argc, char **argv) {
//     ros::init(argc, argv, "my_parser");
//     if (argc != 2) {
//         ROS_ERROR("Missing URDF");
//         return -1;
//     }
//     std::string urdf_file = argv[1];

//     urdf::Model model;
//     if (!model.initFile(urdf-file)) {
//         ROS_ERROR("Failed to parse URDF");
//         return -1;
//     }
//     ROS_INFO("Successfully parsed URDF");
//     return 0;
// }