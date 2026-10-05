"""Chạy teleop openarm_shadow với backend OpenArm qua ROS 2.

    ros2 launch openarm_shadow_ros teleop.launch.py configs:="/abs/d455_wrist_real.yaml /abs/local_3cam.yaml"
    ros2 launch openarm_shadow_ros teleop.launch.py configs:="..." dry_run:=true      # chỉ đọc, không gửi lệnh

Launch không có bàn phím terminal nên không hỏi 'yes' (--yes); vẫn phải giữ READY (hoặc bấm SPACE trên cửa sổ hiển
thị) mới engage và bắt đầu gửi lệnh. Đường dẫn config nên là đường dẫn tuyệt đối.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(context):
    get = lambda k: LaunchConfiguration(k).perform(context).strip()  # noqa: E731
    args = ["--yes", "--source", get("source")]
    for c in get("configs").split():
        args += ["--config", c]
    if get("arms"):
        args += ["--arms", get("arms")]
    if get("record"):
        args += ["--record", get("record")]
    if get("dry_run").lower() in ("1", "true", "yes"):
        args.append("--dry-run")
    if get("headless").lower() in ("1", "true", "yes"):
        args.append("--headless")
    return [Node(package="openarm_shadow_ros", executable="teleop", name="openarm_shadow_teleop", output="screen",
                 emulate_tty=True, arguments=args)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("configs", default_value="", description="các file config, cách nhau bởi dấu cách"),
        DeclareLaunchArgument("source", default_value="multi", description="multi | realsense | chỉ số webcam"),
        DeclareLaunchArgument("arms", default_value="", description="vd right hoặc right,left (mặc định theo config)"),
        DeclareLaunchArgument("record", default_value="", description="file .npz để ghi --record"),
        DeclareLaunchArgument("dry_run", default_value="false", description="true: chỉ đọc joint_states, không gửi"),
        DeclareLaunchArgument("headless", default_value="false", description="true: không mở cửa sổ hiển thị"),
        OpaqueFunction(function=_node),
    ])
