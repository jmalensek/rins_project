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
        #DeclareLaunchArgument('run_arm_mover_actions', default_value='true', description='Run arm_mover_actions.py'),
        DeclareLaunchArgument('run_detect_yellow_line', default_value='true', description='Run detect_yellow_line.py'),
        DeclareLaunchArgument('run_detect_obstacles', default_value='true', description='Run detect_obstacles.py'),
        DeclareLaunchArgument('run_robot_explorer', default_value='true', description='Run robot_explorer.py'),
    ]

    nodes = [
        #_script_node('arm_mover_actions.py', 'run_arm_mover_actions', namespace),
        _script_node('detect_yellow_line.py', 'run_detect_yellow_line', namespace),
        _script_node('detect_obstacles.py', 'run_detect_obstacles', namespace),
        _script_node('robot_explorer.py', 'run_robot_explorer', namespace),
    ]

    return LaunchDescription(args + nodes)
