from reedsolo import RSCodec, ReedSolomonError
import numpy as np

class DataStream:
    """
    Reads a file, scrambles it byte-by-byte, and yields 
    6-bit integer payloads for the video encoder.
    """
    def __init__(self, filepath, cfg, scrambler=None, bits_per_payload=6):
        assert cfg.ecc_bytes > 0, "ECC bytes must be greater than zero"
        assert cfg.ecc_data_bytes > 0, "ECC data bytes must be greater than zero"
        assert cfg.ecc_bytes % bits_per_payload == 0, "ECC symbols must align to byte boundaries"
        self.filepath = filepath
        self.scrambler = scrambler
        self.bits_per_payload = bits_per_payload
        self.bit_shift_mask = (1 << bits_per_payload) - 1
        self._generator = self._payload_generator()
        # We have cfg.ecc_bytes and cfg.ecc_data_bytes
        self.cfg = cfg
        self.rsc = RSCodec(cfg.ecc_bytes)
        with open(self.filepath, "rb") as f:
            self.filesize_real = self.get_file_size(f)
        self.filesize = ((self.filesize_real + self.cfg.ecc_data_bytes - 1) // self.cfg.ecc_data_bytes) * (self.cfg.ecc_data_bytes + self.cfg.ecc_bytes)
        
        print (f"DataStream initialized for file '{self.filepath}' of size {self.filesize_real} bytes.")
        print (f"Using ECC: {cfg.ecc_bytes} bytes per {cfg.ecc_data_bytes} data bytes.")
        print (f"Total size with ECC and padding: {self.filesize} bytes.")

    def get_file_size(self, file):
        file.seek(0, 2)  # Move to end of file
        size = file.tell()
        file.seek(0)     # Reset to start of file
        return size

    def _payload_generator(self):
        bit_buffer = 0
        bit_count = 0
        
        with open(self.filepath, "rb") as f:
            while True:
                # 1. Read a chunk of bytes
                chunk = f.read(self.cfg.ecc_data_bytes)
                # Pad chunk if needed
                if len(chunk) < self.cfg.ecc_data_bytes:
                    bytes_padded = self.cfg.ecc_data_bytes - len(chunk)
                    chunk += bytes(bytes_padded)
                chunk = self.rsc.encode(chunk) # Apply ECC
                if not chunk:
                    break
                
                # 2. Scramble the bytes immediately
                scrambled_chunk = chunk if self.scrambler is None else self.scrambler.scramble(chunk)

                # 3. Pack bits into a rolling buffer
                for byte in scrambled_chunk:
                    bit_buffer = (bit_buffer << 8) | byte
                    bit_count += 8

                    # 4. Extract 6-bit chunks whenever possible
                    while bit_count >= self.bits_per_payload:
                        # Shift so the target 6 bits are at the bottom
                        shift_amount = bit_count - self.bits_per_payload
                        payload = (bit_buffer >> shift_amount) & self.bit_shift_mask
                        
                        # Remove used bits from buffer (masking is optional but clean)
                        # We just lower the count; the old bits stay in 'high' positions 
                        # but will be pushed out eventually or ignored.
                        # For cleanliness, let's mask:
                        mask = (1 << shift_amount) - 1
                        bit_buffer &= mask
                        bit_count -= self.bits_per_payload
                        
                        yield payload
        
        # 5. Handle end of file (Padding)
        # If bits remain, pad them to 6 bits
        if bit_count > 0:
            shift_needed = self.bits_per_payload - bit_count
            payload = (bit_buffer << shift_needed) & self.bit_shift_mask
            yield payload
            
        # 6. Infinite padding for remaining frames
        while True:
            yield 0 # Padding with zero-value payloads

    def get_batch(self, count):
        """Returns a list of 'count' 6-bit integers."""
        return [next(self._generator) for _ in range(count)]
    
    def bytes_to_symbols(self, byte_data):
        """Convert byte data to list of 6-bit symbols."""
        bit_buffer = 0
        bit_count = 0
        symbols = []
        for byte in byte_data:
            bit_buffer = (bit_buffer << 8) | byte
            bit_count += 8
            while bit_count >= self.bits_per_payload:
                shift_amount = bit_count - self.bits_per_payload
                symbol = (bit_buffer >> shift_amount) & self.bit_shift_mask
                symbols.append(symbol)
                mask = (1 << shift_amount) - 1
                bit_buffer &= mask
                bit_count -= self.bits_per_payload
        # Handle remaining bits
        if bit_count > 0:
            shift_needed = self.bits_per_payload - bit_count
            symbol = (bit_buffer << shift_needed) & self.bit_shift_mask
            symbols.append(symbol)
        return symbols
    
    def symbols_to_bytes(self, symbols):
        """Convert list of 6-bit symbols back to byte data."""
        bit_buffer = 0
        bit_count = 0
        byte_array = bytearray()
        for symbol in symbols:
            bit_buffer = (bit_buffer << self.bits_per_payload) | symbol
            bit_count += self.bits_per_payload
            while bit_count >= 8:
                shift_amount = bit_count - 8
                byte = (bit_buffer >> shift_amount) & 0xFF
                byte_array.append(byte)
                mask = (1 << shift_amount) - 1
                bit_buffer &= mask
                bit_count -= 8
        return bytes(byte_array)

class ECCDecoder():
    """
    Decodes byte streams with Reed-Solomon ECC.
    """
    def __init__(self, cfg, descrambler=None):
        assert cfg.ecc_bytes > 0, "ECC bytes must be greater than zero"
        self.cfg = cfg
        self.rsc = RSCodec(cfg.ecc_bytes)
        self.scrambler = descrambler
        self._bytecounter = 0
        self._buf = []

    def add_bytes(self, byte_list):
        """Add bytes to the internal buffer for ECC decoding."""
        self._buf.extend(byte_list)
    
    def get_size_with_ecc(self, data_size):
        """Calculate total size including ECC bytes."""
        block_size = self.cfg.ecc_data_bytes + self.cfg.ecc_bytes
        num_blocks = (data_size + self.cfg.ecc_data_bytes - 1) // self.cfg.ecc_data_bytes
        return num_blocks * block_size

    def decode_buffer(self, length):
        print (f"Decoding {length} bytes (buffer size {len(self._buf)} bytes) with ECC...")
        decoded_data = bytearray()
        block_size = self.cfg.ecc_data_bytes + self.cfg.ecc_bytes
        print(f"Using block size: {block_size} bytes ({self.cfg.ecc_data_bytes} data + {self.cfg.ecc_bytes} ECC)")
        assert length % block_size == 0, "Length to decode must be multiple of ECC block size"
        errors = np.zeros(length // block_size, dtype=int)
        failures_mask = np.zeros(length // block_size, dtype=bool)
        for i in range(0, length, block_size):
            print(f"\rDecoding block {i // block_size + 1}/{length // block_size}", end='', flush=True)
            block = bytes(self._buf[i:i+block_size])
            # Descramble if needed
            if self.scrambler is not None:
                block = self.scrambler.descramble(block)
            try:
                decoded_block, _, erratas = self.rsc.decode(block)
                decoded_data.extend(decoded_block)
                errors[i // block_size] = len(erratas)
            except ReedSolomonError as e:
                print(f"Reed-Solomon decoding error at block starting byte {i}: {e}")
                # Use original data on failure
                decoded_data.extend(block[:self.cfg.ecc_data_bytes])
                failures_mask[i // block_size] = True
        self._buf = [] # Clear buffer after decoding
        print()
        print(f"ECC Decoding complete: {np.sum(errors)} total errors corrected across {len(errors)} blocks.")
        return decoded_data, errors, failures_mask

        