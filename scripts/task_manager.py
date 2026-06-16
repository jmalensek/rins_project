from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

'''
Updejtano za Task2

Zalaufa:

- detect people
- detect rings
- detect cylinders
- red/green cell detection
- tile detection
- arm mover actions
- recognize people
'''


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
        DeclareLaunchArgument('run_DETECT_PEOPLE_2s', default_value='true', description='Run DETECT_PEOPLE_2s.py'),
        DeclareLaunchArgument('run_detect_rings27', default_value='true', description='Run detect_rings27.py'),
        DeclareLaunchArgument('run_cylinder_segmentation', default_value='true', description='Run cylinder_segmentation.cpp'),
        DeclareLaunchArgument('run_red_green_cell_detection', default_value='true', description='Run red_green_cell_detection.py'),
        DeclareLaunchArgument('run_tile_detection', default_value='true', description='Run tile_detection.py'),
        DeclareLaunchArgument('run_arm_mover_actions', default_value='true', description='Run arm_mover_actions.py'),
        DeclareLaunchArgument('run_recognize_people', default_value='true', description='Run recognize_people_2s.py'),
    ]

    nodes = [
        _script_node('detect_people2.py', 'run_DETECT_PEOPLE_2s', namespace),
        _script_node('detect_rings27.py', 'run_detect_rings27', namespace),
        _script_node('cylinder_segmentation.cpp', 'run_cylinder_segmentation', namespace),
        _script_node('red_green_cell_detection.py', 'run_red_green_cell_detection', namespace),
        _script_node('tile_detection.py', 'run_tile_detection', namespace),
        _script_node('arm_mover_actions.py', 'run_arm_mover_actions', namespace),
        _script_node('recognize_people_2s.py', 'run_recognize_people', namespace),
    ]

    return LaunchDescription(args + nodes)
