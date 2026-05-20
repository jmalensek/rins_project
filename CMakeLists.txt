kappa@isildur:~/Documents/Task1_robot/src$ colcon build --symlink-install
Starting >>> dis_tutorial7
Starting >>> rins_project
Finished <<< dis_tutorial7 [0.08s]
--- stderr: rins_project                             
CMake Warning (dev) at /usr/lib/x86_64-linux-gnu/cmake/pcl/Modules/FindFLANN.cmake:45 (find_package):
  Policy CMP0144 is not set: find_package uses upper-case <PACKAGENAME>_ROOT
  variables.  Run "cmake --help-policy CMP0144" for policy details.  Use the
  cmake_policy command to set the policy and suppress this warning.

  CMake variable FLANN_ROOT is set to:

    /usr

  For compatibility, find_package is ignoring the variable, but code in a
  .cmake module might still use it.
Call Stack (most recent call first):
  /usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake:261 (find_package)
  /usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake:306 (find_flann)
  /usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake:570 (find_external_library)
  /opt/ros/jazzy/share/pcl_conversions/cmake/ament_cmake_export_dependencies-extras.cmake:21 (find_package)
  /opt/ros/jazzy/share/pcl_conversions/cmake/pcl_conversionsConfig.cmake:41 (include)
  CMakeLists.txt:29 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

/home/kappa/Documents/Task1_robot/src/rins_project/cplusplus/cylinder_segmentation.cpp:24:10: fatal error: cv_bridge/cv_bridge.h: No such file or directory
   24 | #include <cv_bridge/cv_bridge.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
gmake[2]: *** [CMakeFiles/cylinder_segmentation.dir/build.make:76: CMakeFiles/cylinder_segmentation.dir/cplusplus/cylinder_segmentation.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:137: CMakeFiles/cylinder_segmentation.dir/all] Error 2
gmake: *** [Makefile:146: all] Error 2
---
Failed   <<< rins_project [3.39s, exited with code 2]

Summary: 1 package finished [3.54s]
  1 package failed: rins_project
  1 package had stderr output: rins_project
kappa@isildur:~/Documents/Task1_robot/src$ colcon build --symlink-install

