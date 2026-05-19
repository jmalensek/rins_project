kappa@isildur:~/Documents/Task1_robot$ ros2 run rins_project cylinder_detection.py 
Barrel detection node starting.
[WARN] [1779208313.275779208] [rcl]: ROS_LOCALHOST_ONLY is deprecated but still honored if it is enabled. Use ROS_AUTOMATIC_DISCOVERY_RANGE and ROS_STATIC_PEERS instead.
[WARN] [1779208313.275795830] [rcl]: 'localhost_only' is disabled, 'automatic_discovery_range' and 'static_peers' will be used.
[INFO] [1779208313.514866868] [detect_barrels]: Node initialized. Publishing markers to /barrels_marker.
[WARN] [1779208313.515533828] [detect_barrels]: python-pcl not found.
[INFO] [1779208313.943703408] [detect_barrels]: 
=== Barrel Report ===
Total barrels : 0
Vertical      : 0
Horizontal    : 0
Leaking       : 0
Colors        : 

Traceback (most recent call last):
  File "/home/kappa/Documents/Task1_robot/install/rins_project/lib/rins_project/cylinder_detection.py", line 909, in <module>
    main()
  File "/home/kappa/Documents/Task1_robot/install/rins_project/lib/rins_project/cylinder_detection.py", line 887, in main
    rclpy.spin(node)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 247, in spin
    executor.spin_once()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 926, in spin_once
    self._spin_once_impl(timeout_sec)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 918, in _spin_once_impl
    raise handler.exception()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/task.py", line 286, in _execute_coroutine_step
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 592, in handler
    await call_coroutine()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 480, in _execute
    await await_or_execute(sub.callback, *msg_tuple)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 115, in await_or_execute
    return callback(*args)
           ^^^^^^^^^^^^^^^
  File "/home/kappa/Documents/Task1_robot/install/rins_project/lib/rins_project/cylinder_detection.py", line 227, in pointcloud_callback
    pts_no_floor = self._remove_floor(pts_map)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kappa/Documents/Task1_robot/install/rins_project/lib/rins_project/cylinder_detection.py", line 371, in _remove_floor
    return pts[mask]
           ~~~^^^^^^
TypeError: only integer scalar arrays can be converted to a scalar index
[ros2run]: Process exited with failure 1
kappa@isildur:~/Documents/Task1_robot$ ^C
kappa@isildur:~/Documents/Task1_robot$ 
