# Scrambler module for encoding and decoding binary data using LFSR
from LFSR import LFSR
class Scrambler:
    def __init__(self, seed = 0b010101010, taps=[0, 4], length=9):
        self.lfsr = LFSR(seed, taps, length)

    def scramble(self, data):
        scrambled = bytearray()
        lfsr_bits = self.lfsr.generate(len(data) * 8)
        for i in range(len(data)):
            byte = data[i]
            scrambled_byte = 0
            for j in range(8):
                bit = (byte >> (7 - j)) & 1
                scrambled_bit = bit ^ lfsr_bits[i * 8 + j]
                scrambled_byte = (scrambled_byte << 1) | scrambled_bit
            scrambled.append(scrambled_byte)
        return bytes(scrambled)

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