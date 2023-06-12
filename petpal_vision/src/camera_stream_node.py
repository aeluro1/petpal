import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2 as cv

class CameraBroadcaster(Node):
    def __init__(self):
        super()._init__("object_tracker")
        self._publisher = self.create_publisher(Image, "/camera/image_raw/annotated", QoSPresetProfiles.SENSOR_DATA)
        self._cv_bridge = CvBridge()
        self.create_timer(1/60, self._broadcast) # implement ROS parameter for camera FPS

    def _broadcast(self, msg: Image):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().fatal("Camera could not be opened")
        
        ret, frame = cap.read()

        if not ret:
            self.get_logger().error("Could not retrieve frame from camera buffer")

        image = self._cv_bridge.imgmsg_to_cv2(frame)
        self._publisher.publish(image)

        cap.release()
        cv.destroyAllWindows()


def main(args: list = None):
    rclpy.init(args = args)
    camera_broadcaster = CameraBroadcaster()
    rclpy.spin(camera_broadcaster)

    camera_broadcaster.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

    