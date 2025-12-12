from reedsolo import RSCodec
from scrambler import Scrambler
import json

# Metadata class to hold video metadata information
# Exportable as JSON and with heavy parity
class Metadata:
    def __init__(self, max_frame_size=4096):
        self.max_frame_size = max_frame_size
        self.n_blocks_per_frame, self.n_data_bytes_per_block, self.n_parity_bytes_per_block = self.compute_rs_layout()
        self.rsc = RSCodec(self.n_parity_bytes_per_block)

    def prepare_encoder(self, data):
        self.data = data
        assert "filename" in data, "Metadata must include 'filename'"
        assert "filesize" in data, "Metadata must include 'filesize'"
        assert "filesize_real" in data, "Metadata must include 'filesize_real'"

    def compute_rs_layout(self):
        MAX_RS_BYTES = 255
        n_blocks_per_frame = self.max_frame_size // MAX_RS_BYTES
        # Hard code 127, 128 for now
        n_data_bytes_per_block = 127
        n_parity_bytes_per_block = 128
        
        return n_blocks_per_frame, n_data_bytes_per_block, n_parity_bytes_per_block
        
    def get_encoded_bytes(self):
        raw_meta_dic = self.data
        meta_json = json.dumps(raw_meta_dic)
        meta_bytes = meta_json.encode('utf-8')

        n_meta_blocks = (len(meta_bytes) + self.n_data_bytes_per_block - 1) // self.n_data_bytes_per_block
        meta_ecc_blocks = []
        for i in range(n_meta_blocks):
            start_idx = i * self.n_data_bytes_per_block
            end_idx = start_idx + self.n_data_bytes_per_block
            block_data = meta_bytes[start_idx:end_idx]
            if len(block_data) < self.n_data_bytes_per_block:
                block_data += bytes(self.n_data_bytes_per_block - len(block_data))
            ecc_block = self.rsc.encode(block_data)
            meta_ecc_blocks.append(ecc_block)
        
        # scramble metadata blocks
        scrambler = Scrambler()
        scrambled_meta_blocks = [scrambler.scramble(block) for block in meta_ecc_blocks]

        encoded_meta_bytes = b''.join(scrambled_meta_blocks)
        assert len(encoded_meta_bytes) <= self.max_frame_size, "Metadata exceeds maximum frame size."

        return encoded_meta_bytes
    
    def extract_metadata(self, decoded_bytes):
        n_total_blocks = len(decoded_bytes) // (self.n_data_bytes_per_block + self.n_parity_bytes_per_block)
        scrambler = Scrambler()

        meta_bytes = bytearray()
        for i in range(n_total_blocks):
            start_idx = i * (self.n_data_bytes_per_block + self.n_parity_bytes_per_block)
            end_idx = start_idx + (self.n_data_bytes_per_block + self.n_parity_bytes_per_block)
            block = decoded_bytes[start_idx:end_idx]
            descrambled_block = scrambler.descramble(block)
            decoded_block = self.rsc.decode(descrambled_block)[0]
            meta_bytes.extend(decoded_block)
        
        # Remove padding zeros
        meta_bytes = meta_bytes.rstrip(b'\x00')
        meta_json = meta_bytes.decode('utf-8')
        meta_dic = json.loads(meta_json)

        return meta_dic
    
if __name__ == "__main__":
    # Loopback test
    sample_metadata = {
        "filename": "example.txt",
        "filesize": 123456,
        "other_info": "test metadata"
    }
    max_frame_size = 4096  # bytes
    metadata = Metadata(sample_metadata, max_frame_size)
    encoded_bytes = metadata.get_encoded_bytes()
    print(f"Encoded metadata size: {len(encoded_bytes)} bytes")
    print(f"Encoded metadata bytes: {encoded_bytes}")
    decoded_metadata = metadata.extract_metadata(encoded_bytes)
    print(f"Decoded metadata: {decoded_metadata}")

    assert decoded_metadata == sample_metadata, "Decoded metadata does not match original."
    print ("Metadata loopback test passed.")
