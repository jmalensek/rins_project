# RINS_PROJECT

Komande za zagon nav2 za fizičnega robota:

1. ros2 launch turtlebot4_navigation localization.launch.py map:=/home/kappa/Documents/Task1/src/rins_project/maps/map_name.yaml
2. ros2 launch turtlebot4_navigation nav2.launch.py params_file:=/home/kappa/Documents/Task1/src/rins_project/config/nav2_careless_extreme.yaml # Po želji zamenjaj z drugim nav2 configom
3. ros2 launch turtlebot4_viz view_navigation.launch.py

Za task 2 zaženi: 

1. ros2 launch dis_tutorial7 sim_turtlebot_nav.launch.py params_file:=/home/kappa/Documents/Task1_robot/src/rins_project/config/nav2_careless_moderate.yaml use_sim_time:=true

2. ros2 run dis_tutorial7 arm_mover_actions.py

3. ros2 topic pub --once /arm_command std_msgs/msg/String "{data: look_for_qr}"

4. ros2 launch rins_project task2_movement_launch.py


Tuki so samo moji zapiski, kej si rabim še popravit:

red_green_cell_detection: premikanje - go to pose - tega jz ne znam, bom doma

ros2 topic pub --once /task_manager/color_of_the_cell std_msgs/msg/String "{data: 'red'}"

ros2 topic pub --once /task_manager/task_started std_msgs/msg/Bool "{data: True}"


tile_detection za popravit - zaenkrat sploh ne zazna pravilno ploščic

[ERROR] [1781600394.032165543] [detect_tiles]: Error processing tile 0: OpenCV(4.13.0) /io/opencv/modules/imgproc/src/imgwarp.cpp:3053: error: (-215:Assertion failed) src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4 in function 'getPerspectiveTransform'


