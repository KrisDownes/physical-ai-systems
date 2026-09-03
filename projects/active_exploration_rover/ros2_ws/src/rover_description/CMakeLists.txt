cmake_minimum_required(VERSION 3.8)
project(rover_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_urdf_imu
    test/test_urdf_imu.py
  )

  ament_add_pytest_test(
    test_world_imu_system
    test/test_world_imu_system.py
  )

  ament_add_pytest_test(
    test_camera_lidar_clearance
    test/test_camera_lidar_clearance.py
  )
endif()

install(
  DIRECTORY launch urdf rviz worlds
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
