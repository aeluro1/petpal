cmake_minimum_required(VERSION 3.8)
project(petpal_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# add_library(${PROJECT_NAME}_components SHARED src/coordinate_transform_component.cpp)
# ament_target_dependencies(
#         ${PROJECT_NAME}_components
#         "rclcpp"
#         "rclcpp_components"
#         "tf2_ros"
#         "tf2_eigen"
#         "stsl_interfaces"
# )
# rclcpp_components_register_node(
#         ${PROJECT_NAME}_components
#         PLUGIN "coordinate_transform::CoordinateTransformComponent"
#         EXECUTABLE ${PROJECT_NAME}_node
# )

# install(
#         TARGETS ${PROJECT_NAME}_components
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# add_executable(parser src/parser.cpp)
# target_link_libraries(parser ${catkin_LIBRARIES})

install(
  DIRECTORY meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()