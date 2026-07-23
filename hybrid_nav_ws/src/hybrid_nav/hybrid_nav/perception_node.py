import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.sub = self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.pub = self.create_publisher(Float32, '/crowd_density', 10)
        self.declare_parameter('range_threshold', 1.5)  # metre

    def cb_scan(self, msg: LaserScan):
        thr = float(self.get_parameter('range_threshold').value)
        arr = np.array(msg.ranges, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            density = 0.0
        else:
            close = np.sum(finite < thr)
            density = float(close) / float(finite.size)
        self.pub.publish(Float32(data=density))
        # self.get_logger().info(f'density={density:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
