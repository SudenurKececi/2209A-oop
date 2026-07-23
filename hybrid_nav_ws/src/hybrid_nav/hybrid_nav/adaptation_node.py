import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class AdaptationNode(Node):
    def __init__(self):
        super().__init__('adaptation_node')
        self.sub = self.create_subscription(Float32, '/crowd_density', self.cb_density, 10)
        self.pub = self.create_publisher(Float32, '/speed_limit', 10)
        self.declare_parameter('low_density_threshold', 0.05)
        self.declare_parameter('high_density_threshold', 0.25)
        self.declare_parameter('min_speed', 0.20)
        self.declare_parameter('max_speed', 0.60)
        self.current = float(self.get_parameter('max_speed').value)

    def cb_density(self, msg: Float32):
        low  = float(self.get_parameter('low_density_threshold').value)
        high = float(self.get_parameter('high_density_threshold').value)
        vmin = float(self.get_parameter('min_speed').value)
        vmax = float(self.get_parameter('max_speed').value)
        d = float(msg.data)

        if d <= low:   v = vmax
        elif d >= high:v = vmin
        else:
            a = (d - low) / max(1e-6, (high - low))
            v = vmax + a * (vmin - vmax)

        if abs(v - self.current) > 1e-3:
            self.current = v
            self.pub.publish(Float32(data=v))
            self.get_logger().info(f'speed_limit -> {v:.2f} m/s (density={d:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = AdaptationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
