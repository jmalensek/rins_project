from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _script_node(executable: str, arg_name: str, namespace: LaunchConfiguration) -> Node:
    return Node(
        package='rins_project',
        executable=executable,
        name=executable.replace('.py', ''),
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration(arg_name)),
    )

# DeclareLaunchArgument('run_robot_explorer', default_value='true', description='Run robot_explorer.py'),
def generate_launch_description() -> LaunchDescription:
    namespace = LaunchConfiguration('namespace')

    args = [
        DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
        DeclareLaunchArgument('run_detect_people2', default_value='true', description='Run DETECT_PEOPLE_2s.py'),
        DeclareLaunchArgument('run_detect_rings27', default_value='true', description='Run detect_rings27.py'),
        DeclareLaunchArgument('run_task_manager', default_value='true', description='Run task_manager.py'),
        DeclareLaunchArgument('run_red_green_cell_detection_better', default_value='true', description='Run red_green_cell_detection_better.py'),
        DeclareLaunchArgument('run_tile_detection', default_value='true', description='Run tile_detection.py'),
        DeclareLaunchArgument('run_cylinder_segmentation', default_value='true', description='Run cylinder_segmentation'),
        #DeclareLaunchArgument('run_arm_mover', default_value='true', description='Run arm_mover.py'),
    ]

    nodes = [
        _script_node('DETECT_PEOPLE_2s.py', 'run_detect_people2', namespace),
        _script_node('detect_rings27.py', 'run_detect_rings27', namespace),
        _script_node('task_manager.py', 'run_task_manager', namespace),
        _script_node('red_green_cell_detection_better.py', 'run_red_green_cell_detection_better', namespace),
        _script_node('tile_detection.py', 'run_tile_detection', namespace),
        _script_node('cylinder_segmentation', 'run_cylinder_segmentation', namespace),
        #_script_node('arm_mover.py', 'run_arm_mover', namespace),
    ]

    return LaunchDescription(args + nodes)
