class LFSR:
    def __init__(self, seed: int, taps: list[int], length: int):
        self.length = length
        self.state = seed & ((1 << length) - 1)
        self.taps = taps

        if 0 not in self.taps:
            warning = f"Tap list {self.taps} does not include output tap (0). Adding it automatically."
            print("Warning:", warning)
            self.taps.append(0)
        
        # Pre-calculate tap mask for O(1) feedback calculation
        self.tap_mask = 0
        for t in self.taps:
            self.tap_mask |= (1 << t)

    def step(self) -> int:
        output_bit = self.state & 1

        masked = self.state & self.tap_mask

        # Requires Python 3.10+
        feedback = masked.bit_count() & 1

        
        self.state >>= 1
        self.state |= (feedback << (self.length - 1))
        
        return output_bit

    def generate(self, n: int) -> list[int]:
        return [self.step() for _ in range(n)]

    def generate_bytes(self, n_bytes: int) -> bytes:
        """
        Generates n_bytes of data from the LFSR efficiently.
        """
        state = self.state
        length = self.length
        mask = self.tap_mask
        feedback_shift = length - 1
        
        # Check availability of bit_count once outside the loop
        use_bit_count = hasattr(int, "bit_count")
        
        # Pre-allocate buffer
        result = bytearray(n_bytes)
        
        # Local variable optimization
        for i in range(n_bytes):
            byte_val = 0
            # Unroll 8 bits per byte
            for _ in range(8):
                output_bit = state & 1
                
                if use_bit_count:
                    feedback = (state & mask).bit_count() & 1
                else:
                    feedback = bin(state & mask).count('1') & 1
                
                state >>= 1
                state |= (feedback << feedback_shift)
                
                # MSB first packing to match original scrambler logic
                byte_val = (byte_val << 1) | output_bit
            
            result[i] = byte_val
        
        self.state = state
        return bytes(result)