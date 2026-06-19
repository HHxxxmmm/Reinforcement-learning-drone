import socket
import numpy as np
import struct
import yaml

def get_packet(tcp_socket, packet_size):
    data = b''
    while len(data) < packet_size:
        packet = tcp_socket.recv(packet_size - len(data))
        data += packet
    return data


def send_packet(tcp_socket, packet_format, data):
    packed_data = struct.pack(packet_format, *data)
    tcp_socket.sendall(packed_data)

class NetworkAdaptor:
    """

    """
    # 与 MATLAB get_my_initial.m 一致：100 个 int32（400 字节）
    INITIAL_PACKET_FORMAT = "<100i"
    GETTING_PACKET_FORMAT = "=27d"
    SENDING_PACKET_FORMAT = "<5d"
    INITIAL_PACKET_SIZE = 400
    GETTING_PACKET_SIZE = 216
    SENDING_PACKET_SIZE = 40

    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.host = self.config['host']
        self.port = self.config['port']
        self.timeout = self.config.get('socket_timeout', 30)
        self.socket = None

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _open_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        return sock

    def connect(self):
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
        self.socket = self._open_socket()
        self.socket.connect((self.host, self.port))

    def reconnect(self):
        self.connect()

    def close(self):
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None

    def send_initial_packet(self, initial_data):
        send_packet(self.socket, self.INITIAL_PACKET_FORMAT, initial_data)

    def get_observation_packet(self):
        data = get_packet(self.socket, self.GETTING_PACKET_SIZE)
        unpacked_data = np.array(struct.unpack(self.GETTING_PACKET_FORMAT, data), dtype=np.float64)
        return unpacked_data

    def send_action_packet(self, action):
        send_packet(self.socket, self.SENDING_PACKET_FORMAT, action)