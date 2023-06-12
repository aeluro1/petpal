import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

class ObjectTracker(Node):
    def __init__(self):
        super()._init__("object_tracker")
        self._publisher = self.create_publisher(Image, "/camera/image_raw/annotated", QoSPresetProfiles.SENSOR_DATA)
        self._subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._parse_image,
            QoSPresetProfiles.SENSOR_DATA
        )

    def _parse_image(self, msg: Image):
        self.get_logger().info(f"Image frame received")

        # add yolo/resnet call here


def main(args: list = None):
    rclpy.init(args = args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)

    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()