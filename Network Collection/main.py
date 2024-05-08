from collecting.traffic_collector import NetworkTrafficCollector
from processing.data_processor import DataAggregator
import logging

def main():
    logging.basicConfig(level=logging.INFO)

    aggregator = DataAggregator("batch", ["error", "corrupt"], "network_data.json", batch_size=100)
    network_collector = NetworkTrafficCollector("Wi-Fi 2", ["TCP", "UDP"], packet_callback=aggregator.process_packet)
    network_collector.start_capture(timeout=10)

if __name__ == "__main__":
    main()
