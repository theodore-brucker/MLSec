from multiprocessing import Pool, cpu_count, Manager
from scapy.all import IP, TCP, UDP, rdpcap, PcapReader
import pandas as pd
import os

def extract_packet_features(packet):
    src_ip = dst_ip = src_port = dst_port = protocol = timing = None
    ttl = length = None

    # Check if the packet has an IP layer
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        ttl = packet[IP].ttl
        length = packet[IP].len  # Total length of the IP packet (header + data)
        
    # Check for TCP protocol
    if TCP in packet:
        protocol = 'TCP'
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    
    # Check for UDP protocol
    elif UDP in packet:
        protocol = 'UDP'
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    # Extract timing (epoch time)
    timing = packet.time

    # Create a dictionary with the packet's features
    features = {
        'src_ip': src_ip or 'N/A',
        'dst_ip': dst_ip or 'N/A',
        'src_port': src_port or 'N/A',
        'dst_port': dst_port or 'N/A',
        'protocol': protocol or 'N/A',
        'timing': timing or 'N/A',
        'ttl': ttl or 'N/A',
        'length': length or 'N/A',
    }
    return features

def process_batch(batch):
    # Process each batch of packets
    features = []
    for packet in batch:
        features.append(extract_packet_features(packet))
    return features

from scapy.all import PcapReader
import os

def read_pcap_in_batches(pcap_path, batch_size=1000):
    batches = []
    batch = []

    for file_name in os.listdir(pcap_path):
        full_file_path = os.path.join(pcap_path, file_name)  # Construct the full file path
        with PcapReader(full_file_path) as pcap_reader:  # Use the full file path
            for packet in pcap_reader:
                batch.append(packet)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
            # Add the last batch if it has packets
            if batch:
                batches.append(batch)
                batch = []  # Reset the batch for the next file (if needed)
    return batches


def init_main():
    pcap_directory = 'C:\AWS Target\Original Network Traffic and Log data\Thursday-15-02-2018\pcap'
    batch_size = 1000  # Define suitable batch size

    # Read pcap in batches
    batches = read_pcap_in_batches(pcap_directory, batch_size=batch_size)

    # Process batches in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_batch, batches)

    # Flatten the list of results
    all_features = [item for sublist in results for item in sublist]

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv('output.csv', index=False)
    print(f"Data saved to output.csv")

if __name__ == "__main__":
    init_main()