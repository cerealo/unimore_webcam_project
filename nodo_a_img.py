import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Per ricevere la risposta di testo
import cv2
import os

class NodoA(Node):

    def __init__(self):
        super().__init__('nodo_a_talker_listener')
        
        # 1. TALKER: Pubblica la foto
        self.publisher_foto = self.create_publisher(Image, 'topic_foto', 10)
        
        # 2. LISTENER: Ascolta la risposta di testo dal Nodo B
        self.subscriber_risposta = self.create_subscription(
            String, 'topic_risposta', self.risposta_callback, 10)
        
        # Timer per avviare il botta e risposta ogni 2 secondi
        self.timer = self.create_timer(2.0, self.timer_callback)
        
        # Metti qui il percorso della tua foto
        self.image_path = 'src/py_pubsub/py_pubsub/psyduck.jpg'
        
        self.get_logger().info('Nodo A pronto per il botta e risposta!')

    def timer_callback(self):
        if not os.path.exists(self.image_path):
            self.get_logger().error(f"Foto non trovata in: {self.image_path}")
            return

        cv_image = cv2.imread(self.image_path)
        if cv_image is not None:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            height, width, channels = cv_image_rgb.shape
            
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height = height
            msg.width = width
            msg.encoding = 'rgb8'
            msg.step = width * channels
            msg.data = cv_image_rgb.tobytes()

            # "Botta": Invio la foto
            self.publisher_foto.publish(msg)
            self.get_logger().info(f"--> [Nodo A] Inviata foto ({width}x{height}) al Nodo B...")

    def risposta_callback(self, msg):
        # "Risposta": Ricevo il testo dal Nodo B
        self.get_logger().info(f"<-- [Nodo A] Risposta ricevuta dal Nodo B: '{msg.data}'\n")


def main(args=None):
    rclpy.init(args=args)
    node = NodoA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()