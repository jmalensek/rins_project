# RINS_PROJECT

Komande za zagon nav2 za fizičnega robota:

1. ros2 launch turtlebot4_navigation localization.launch.py map:=/home/kappa/Documents/Task1/src/rins_project/maps/map_name.yaml
2. ros2 launch turtlebot4_navigation nav2.launch.py params_file:=/home/kappa/Documents/Task1/src/rins_project/config/nav2_careless_extreme.yaml # Po želji zamenjaj z drugim nav2 configom
3. ros2 launch turtlebot4_viz view_navigation.launch.py

Za task 2 zaženi: ros2 launch dis_tutorial7 sim_turtlebot_nav.launch.py params_file:=/home/kappa/Documents/Task2/src/rins_project/config/nav2_careless_moderate.yaml use_sim_time:=true


Tuki so samo moji zapiski, kej si rabim še popravit:

red_green_cell_detection: premikanje - go to pose - tega jz ne znam, bom doma

ros2 topic pub --once /task_manager/color_of_the_cell std_msgs/msg/String "{data: 'red'}"

ros2 topic pub --once /task_manager/task_started std_msgs/msg/Bool "{data: True}"
