from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam        = LaunchConfiguration('slam')
    params_file = LaunchConfiguration('params_file')
    map_yaml    = LaunchConfiguration('map')   # <-- MAP

    nav2_dir = get_package_share_directory('nav2_bringup')
    default_params = os.path.join(nav2_dir, 'params', 'nav2_params.yaml')
    default_map   = os.path.join(nav2_dir, 'maps', 'turtlebot3_world.yaml')

    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    declare_slam         = DeclareLaunchArgument('slam',        default_value='True')
    declare_params_file  = DeclareLaunchArgument('params_file', default_value=default_params)
    declare_map          = DeclareLaunchArgument('map',         default_value=default_map)  # <-- MAP

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'slam': slam,
            'params_file': params_file,
            'map': map_yaml,   # <-- MAP BURADA GEÇİLİYOR
        }.items()
    )

    perception = Node(
        package='hybrid_nav', executable='perception_node', name='perception_node',
        output='screen', parameters=[{'range_threshold': 1.5}],
    )
    adaptation = Node(
        package='hybrid_nav', executable='adaptation_node', name='adaptation_node',
        output='screen', parameters=[
            {'low_density_threshold': 0.05},
            {'high_density_threshold': 0.25},
            {'min_speed': 0.20},
            {'max_speed': 0.60},
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_slam,
        declare_params_file,
        declare_map,
        bringup,
        perception,
        adaptation,
    ])
