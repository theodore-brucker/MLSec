import json
import numpy as np
import os
import hashlib
from scapy.all import IP, TCP, ICMP, ARP, Ether
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import logging

import matplotlib.pyplot as plt
from datetime import datetime

def ip_to_hash(ip: str) -> int:
    return int(hashlib.sha256(ip.encode()).hexdigest()[:8], 16)

def flags_to_int(flags: str) -> int:
    return sum((0x01 << i) for i, f in enumerate('FSRPAUEC') if f in flags)

def protocol_to_int(proto: int) -> int:
    # Simple mapping for common protocol numbers to a smaller range
    protocol_map = {1: 1,  # ICMP
                    6: 2,  # TCP
                    17: 3,  # UDP
                    2: 4}  # IGMP
    return protocol_map.get(proto, 0)  # Default to 0 if unknown

def translate_features(packet) -> List[int]:
    # Ensure to handle Ethernet frames to extract payload properly
    if Ether in packet:
        packet = packet[IP]

    ip_layer = packet[IP] if IP in packet else None
    tcp_layer = packet[TCP] if TCP in packet else None
    icmp_layer = packet[ICMP] if ICMP in packet else None

    features = [
        ip_to_hash(ip_layer.src) if ip_layer else ip_to_hash("0.0.0.0"),
        ip_to_hash(ip_layer.dst) if ip_layer else ip_to_hash("0.0.0.0"),
        ip_layer.ttl if ip_layer else 0,
        protocol_to_int(ip_layer.proto) if ip_layer else 0,
        tcp_layer.sport if tcp_layer else 0,
        tcp_layer.dport if tcp_layer else 0,
        flags_to_int(tcp_layer.flags) if tcp_layer and hasattr(tcp_layer, 'flags') else 0,
        len(packet)  # Using packet length as a feature
    ]

    # Add ICMP type and code as features if the packet is an ICMP packet
    if icmp_layer:
        features.extend([icmp_layer.type, icmp_layer.code])
    else:
        features.extend([0, 0])  # Default to 0 if not ICMP

    return features

class DataAggregator:
    def __init__(self, strategy: str, rules: List[str], output_file: str, batch_size: int = 100):
        self.strategy = strategy
        self.rules = rules
        self.output_file = output_file
        self.stats_file = output_file.replace('.json', '_stats.json')
        self.visualization_file = output_file.replace('.json', '_viz.png')
        self.batch_size = batch_size
        self.buffer = []
        self.scaler = StandardScaler()
        self.statistics = {}
        self.feature_names = [
                                'Source IP Hash',
                                'Destination IP Hash',
                                'TTL',
                                'Protocol Number',
                                'Source Port',
                                'Destination Port',
                                'TCP Flags',
                                'Packet Length',
                                'ICMP Type',
                                'ICMP Code'
                            ]
        logging.basicConfig(level=logging.DEBUG)

    def process_packet(self, packet):
        try:
            self.buffer.append(packet)
            if len(self.buffer) >= self.batch_size:
                self.process_and_write()
        except Exception as e:
            logging.error(f"Failed to process packet: {e}")

    def process_and_write(self):
        try:
            if self.buffer:
                cleaned_data = self._clean_data(self.buffer)
                scaled_data = self._scale_data(cleaned_data)
                self._update_statistics(scaled_data)
                self._generate_visualizations(scaled_data)
                self._write_to_file(scaled_data, mode='a')  # Append mode
                self.buffer = []  # Clear buffer after writing
        except Exception as e:
            logging.error(f"Error during batch processing and writing: {e}")

    def _update_statistics(self, data):
        data = np.array(data)
        if data.size == 0:
            logging.info("No data to update statistics.")
            return

        self.statistics = {name: {} for name in self.feature_names}
        for i, name in enumerate(self.feature_names):
            self.statistics[name]['mean'] = np.mean(data[:, i])
            self.statistics[name]['median'] = np.median(data[:, i])
            self.statistics[name]['std_dev'] = np.std(data[:, i])
            self.statistics[name]['min'] = np.min(data[:, i])
            self.statistics[name]['max'] = np.max(data[:, i])

        self.statistics['last_updated'] = datetime.now().isoformat()
        self.statistics['num_samples'] = len(data)
        with open(self.stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        logging.info("Updated statistics and metadata.")

    def _generate_visualizations(self, data):
        data_array = np.array(data)
        if data_array.size == 0:
            logging.info("No data to generate visualizations.")
            return
        
        num_features = data_array.shape[1]
        plt.figure(figsize=(10, 8))
        for i in range(num_features):
            plt.subplot(num_features, 1, i + 1)
            plt.hist(data_array[:, i], bins=20, color='skyblue', edgecolor='black')
            plt.title(self.feature_names[i])
        
        plt.tight_layout()
        plt.savefig(self.visualization_file)
        plt.close()
        logging.info("Generated visualizations.")

    def _clean_data(self, data: List[Dict]) -> List[Dict]:
        try:
            return [d for d in data if all(rule not in str(d) for rule in self.rules)]
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return []

    def _scale_data(self, data: List[Dict]) -> List[Dict]:
        try:
            data_matrix = np.array([translate_features(d) for d in data])
            scaled_data = self.scaler.fit_transform(data_matrix)
            return scaled_data.tolist()
        except Exception as e:
            logging.error(f"Error scaling data: {e}")
            return []

    def _write_to_file(self, new_data: List[Dict], mode='a'):
        try:
            # Check if the file exists and is not empty
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                with open(self.output_file, 'r+') as file:
                    # Read the current content of the file
                    file.seek(0)
                    data = json.load(file)
                    # Extend the existing array with new data
                    if isinstance(data, list):
                        data.extend(new_data)
                    else:
                        data = new_data  # If for some reason the file structure was corrupted
                    
                    # Rewrite the updated data
                    file.seek(0)
                    file.truncate()  # Clear the file
                    json.dump(data, file, indent=4)
            else:
                # If the file does not exist or is empty, write new data as a new array
                with open(self.output_file, 'w') as file:
                    json.dump(new_data, file, indent=4)
            
            logging.info(f"Wrote preprocessed packet data to {self.output_file}")
        except Exception as e:
            logging.error(f"Error writing to file: {e}")
            