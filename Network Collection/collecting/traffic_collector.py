from scapy.all import sniff, IP
import logging

class NetworkTrafficCollector:
    def __init__(self, collection_point: str, protocol_filters: list = None, packet_callback=None):
        self.collection_point = collection_point
        self.protocol_filters = protocol_filters
        self.packet_callback = packet_callback
        logging.basicConfig(level=logging.DEBUG)

    def start_capture(self, timeout: int = None):
        try:
            logging.info(f"Starting traffic capture on {self.collection_point}...")
            sniff(iface=self.collection_point, prn=self.packet_handler, filter=self._create_bpf_filter(), store=False, timeout=timeout)
        except Exception as e:
            logging.error(f"Failed to start capture: {e}")
            raise RuntimeError("Failed to start packet capture") from e

    def packet_handler(self, packet):
        try:
            if self.packet_callback:
                self.packet_callback(packet)
        except Exception as e:
            logging.warning(f"Error processing packet: {e}")

    def _create_bpf_filter(self) -> str:
        try:
            protocol_map = {"TCP": "tcp", "UDP": "udp", "ICMP": "icmp", "IP": "ip"}
            filters = [protocol_map[p] for p in self.protocol_filters if p in protocol_map]
            return " or ".join(filters)
        except Exception as e:
            logging.error(f"Error creating BPF filter: {e}")
            raise
        
