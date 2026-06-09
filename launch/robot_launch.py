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
        DeclareLaunchArgument('run_detect_people1_robot', default_value='true', description='Run detect_people1_robot.py'),
        DeclareLaunchArgument('run_detect_rings1_robot', default_value='true', description='Run detect_rings1_robot.py'),
        DeclareLaunchArgument('run_greet_people1.robot', default_value='true', description='Run greet_people1_robot.py'),
    ]

    nodes = [
        _script_node('detect_people1_robot.py', 'run_detect_people1_robot', namespace),
        _script_node('detect_rings1_robot.py', 'run_detect_rings1_robot', namespace),
        _script_node('greet_people1_robot.py', 'run_greet_people1.robot', namespace),
    ]

    return LaunchDescription(args + nodes)
