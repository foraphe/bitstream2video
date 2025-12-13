# Scrambler module for encoding and decoding binary data using LFSR
from LFSR import LFSR
import numpy as np
from numba import jit

# JIT-compiled kernel for high performance
@jit(nopython=True)
def _scramble_jit(data, state, tap_mask, length):
    n = len(data)
    out = np.empty(n, dtype=np.uint8)
    feedback_shift = length - 1
    
    for i in range(n):
        byte_val = 0
        # Unroll 8 bits per byte
        for _ in range(8):
            output_bit = state & 1
            
            # Calculate feedback (popcount of masked state)
            masked = state & tap_mask
            # Manual popcount compatible with Numba
            c = 0
            v = masked
            while v > 0:
                v &= (v - 1)
                c += 1
            feedback = c & 1
            
            state >>= 1
            state |= (feedback << feedback_shift)
            
            # MSB first packing
            byte_val = (byte_val << 1) | output_bit
        
        out[i] = data[i] ^ byte_val
        
    return out, state

class Scrambler:
    def __init__(self, seed = 0b010101010, taps=[0, 4], length=9):
        self.lfsr = LFSR(seed, taps, length)

    def scramble(self, data):
        # Convert input to numpy array
        data_np = np.frombuffer(data, dtype=np.uint8)
        
        # Run the JIT compiled function
        # We pass the raw state integers to avoid object overhead
        result_np, new_state = _scramble_jit(
            data_np, 
            self.lfsr.state, 
            self.lfsr.tap_mask, 
            self.lfsr.length
        )
        
        # Update the LFSR object state
        self.lfsr.state = new_state
        
        return result_np.tobytes()

    def descramble(self, data):
        # Descrambling is identical to scrambling due to XOR properties
        return self.scramble(data)
    
if __name__ == "__main__":
    # Simple test
    scrambler = Scrambler()
    original_data = b"\xDE\xAD\xBE\xEF"
    scrambled_data = scrambler.scramble(original_data)
    scrambler = Scrambler()
    descrambled_data = scrambler.descramble(scrambled_data)
    assert original_data == descrambled_data
    print("Scrambling and descrambling successful!")

    # Check if LFSR falls into a short trivial cycle by testing using all-zero input
    zero_data = b"\x00" * 1000
    scrambler = Scrambler(9)
    scrambled_zero_data = scrambler.scramble(zero_data)
    # Try to find a period
    period = None
    for p in range(1, 500):
        if scrambled_zero_data[:p] * (len(scrambled_zero_data) // p) == scrambled_zero_data:
            period = p
            break
    if period:
        print(f"Warning: Detected a potentially short cycle with period {period} in scrambled zero data.")
    else:
        print("No short cycle detected in scrambled zero data.")
    
    from time import time
    # Performance test
    print("Starting performance test with 10 MB of data...")
    large_data = b"\xFF" * 10_000_000  # 10 MB of data
    scrambler = Scrambler()
    start_time = time()
    scrambled_large_data = scrambler.scramble(large_data)
    end_time = time()
    print(f"Scrambled 10 MB of data in {end_time - start_time:.2f} seconds.")
    scrambler = Scrambler()
    start_time = time()
    descrambled_large_data = scrambler.descramble(scrambled_large_data)
    end_time = time()
    print(f"Descrambled 10 MB of data in {end_time - start_time:.2f} seconds.")
    assert large_data == descrambled_large_data
    print("Large data scrambling and descrambling successful!")