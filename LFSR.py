class LFSR:
    def __init__(self, seed: int, taps: list[int], length: int):
        self.length = length
        self.state = seed & ((1 << length) - 1)
        self.taps = taps

        if 0 not in self.taps:
            warning = f"Tap list {self.taps} does not include output tap (0). Adding it automatically."
            print("Warning:", warning)
            self.taps.append(0)

    def step(self) -> int:
        output_bit = self.state & 1

        feedback = 0
        for tap_index in self.taps:
            feedback ^= (self.state >> tap_index) & 1
        
        self.state >>= 1
        

        self.state |= (feedback << (self.length - 1))
        
        return output_bit

    def generate(self, n: int) -> list[int]:
        return [self.step() for _ in range(n)]