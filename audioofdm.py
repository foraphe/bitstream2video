# Assuming 48kHz audio sample rate and 30fps video, we have a generous 1600 samples per frame for some parity
# We might as well compare PSK vs FSK here, using frequencies below 10kHz

import numpy as np

FRAME_RATE = 30
AUDIO_SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // FRAME_RATE  # 1600 samples per frame

class AudioOFDM:
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, frame_size=SAMPLES_PER_FRAME, min_freq=1200, max_freq=10000):
        """
        Initialize OFDM Modem.
        :param min_freq: Minimum frequency in Hz (avoid DC/low freq noise). 
                         Raised to 1200Hz to allow space for Sync Pulse (600Hz).
        :param max_freq: Maximum frequency in Hz (stay within audio bandwidth)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.freq_res = sample_rate / frame_size
        
        # Calculate usable frequency bins
        # Bin n corresponds to frequency n * sample_rate / frame_size
        self.start_bin = int(np.ceil(min_freq / self.freq_res))
        self.end_bin = int(np.floor(max_freq / self.freq_res))
        
        # Ensure we don't exceed Nyquist (frame_size // 2)
        self.end_bin = min(self.end_bin, frame_size // 2 - 1)
        
        self.carriers = np.arange(self.start_bin, self.end_bin + 1)
        self.num_carriers = len(self.carriers)

        # Align to byte boundary
        self.num_carriers = (self.num_carriers // 4) * 4 
        self.carriers = self.carriers[:self.num_carriers]
        
        # QPSK encodes 2 bits per carrier
        self.bits_per_frame = self.num_carriers * 2
        
        # Amplitude scaling to prevent clipping (0.5 leaves headroom)
        self.output_scale = 0.5
        
        # Sync Pulse Configuration
        self.sync_freq = 600  # Hz
        self.sync_duration = 0.005 # 5ms

    def get_capacity(self):
        return self.bits_per_frame

    def get_sync_pulse(self):
        """ 
        Generates the synchronization pulse.
        Uses a Blackman-windowed Cosine wave.
        """
        num_sync_samples = int(self.sample_rate * self.sync_duration)
        t = np.arange(num_sync_samples) / self.sample_rate
        
        # Blackman window to minimize spectral leakage into OFDM bins
        window = np.blackman(num_sync_samples)
        
        # Cosine wave as requested
        pulse = 0.5 * np.cos(2 * np.pi * self.sync_freq * t) * window
        return pulse

    def encode_frame(self, bits):
        """
        Encodes a list of bits (0s and 1s) into a time-domain audio frame.
        Returns: numpy array of shape (1600,) with float values between -1.0 and 1.0
        """
        # 1. Prepare Data
        # Pad with zeros if not enough bits
        if len(bits) < self.bits_per_frame:
            print("Warn: Not enough bits for full frame, check alignment")
            bits = list(bits) + [0] * (self.bits_per_frame - len(bits))
        elif len(bits) > self.bits_per_frame:
            print("Warn: Too many bits for full frame, truncating")
            bits = bits[:self.bits_per_frame]
            
        bits_arr = np.array(bits, dtype=int)
        
        # 2. Map bits to QPSK Symbols
        # Reshape to pairs of bits: (N, 2)
        pairs = bits_arr.reshape(-1, 2)
        
        # QPSK Mapping:
        # 00 -> 1+j, 01 -> -1+j, 10 -> 1-j, 11 -> -1-j
        # Logic: bit0 controls Real sign, bit1 controls Imag sign
        # 0 -> +1, 1 -> -1
        real_parts = 1.0 - 2.0 * pairs[:, 0]
        imag_parts = 1.0 - 2.0 * pairs[:, 1]
        symbols = real_parts + 1j * imag_parts
        
        # 3. Construct Frequency Spectrum
        # rfft expects size N//2 + 1
        spectrum = np.zeros(self.frame_size // 2 + 1, dtype=np.complex64)
        
        # Place symbols into assigned carrier bins
        spectrum[self.carriers] = symbols
        
        # 4. IFFT to Time Domain
        # irfft produces real-valued output from Hermitian symmetric input
        audio_frame = np.fft.irfft(spectrum, n=self.frame_size)
        
        # 5. Normalize
        # Normalize peak to self.output_scale to maximize dynamic range without clipping
        max_amp = np.max(np.abs(audio_frame))
        if max_amp > 0:
            audio_frame = audio_frame / max_amp * self.output_scale

        # 6. Inject Synchronization Pulse
        audio_frame = self.inject_sync_pulse(audio_frame)

        return audio_frame

    def inject_sync_pulse(self, audio_frame):
        """
        Injects a windowed cosine wave pulse at the beginning of the audio buffer.
        This acts as a frame synchronization marker.
        """
        pulse = self.get_sync_pulse()
        L = len(pulse)
        
        # Mix the sync signal into the beginning of the frame
        # Since we reserved low frequencies (0-1200Hz), this 600Hz pulse 
        # will not interfere with the OFDM data.
        if L <= len(audio_frame):
            audio_frame[:L] += pulse
        
        return audio_frame

    def find_stream_start(self, audio_stream, search_window_sec=2.0):
        """
        Finds the start index of the first frame by correlating with the sync pulse.
        This compensates for codec priming samples (silence at start).
        """
        pulse = self.get_sync_pulse()
        
        # Limit search to avoid processing huge files entirely if not needed
        search_len = int(search_window_sec * self.sample_rate)
        if len(audio_stream) > search_len:
            search_region = audio_stream[:search_len]
        else:
            search_region = audio_stream
            
        # Cross-correlation
        # 'valid' mode: returns correlation only where signals fully overlap
        correlation = np.correlate(search_region, pulse, mode='valid')
        
        # find all peaks above a threshold
        threshold = 0.95 * np.max(np.abs(correlation))
        peak_idx = np.argmax(np.abs(correlation) >= threshold)

        # take the first peak that exceeds the threshold
        peak_indices = np.where(np.abs(correlation) >= threshold)[0]
        if len(peak_indices) > 0:
            peak_idx = peak_indices[0]
        else:
            peak_idx = np.argmax(correlation)
        
        return peak_idx

    def decode_synchronized_stream(self, audio_stream, return_symbols=False):
        """
        Generator that yields decoded bits from a raw audio stream.
        Handles synchronization automatically.
        """
        # 1. Find the start of the first frame
        start_idx = self.find_stream_start(audio_stream)
        print(f"Stream synchronized. Start offset: {start_idx} samples")
        
        # 2. Iterate through frames
        current_idx = start_idx
        total_samples = len(audio_stream)
        
        while current_idx + self.frame_size <= total_samples:
            # Extract frame
            frame_audio = audio_stream[current_idx : current_idx + self.frame_size]
            
            # Decode
            bits, symbols = self.decode_frame(frame_audio, return_symbols=return_symbols)
            if return_symbols:
                yield bits, symbols
            else:
                yield bits
            
            # Advance
            current_idx += self.frame_size

    def decode_frame(self, audio_frame, return_symbols=False):
        """
        Decodes a time-domain audio frame into a list of bits.
        :param return_symbols: If True, returns (bits, symbols) tuple.
        """
        # 1. FFT to Frequency Domain
        # Handle size mismatches gracefully
        if len(audio_frame) != self.frame_size:
            if len(audio_frame) > self.frame_size:
                audio_frame = audio_frame[:self.frame_size]
            else:
                audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)))
                
        spectrum = np.fft.rfft(audio_frame)
        
        # 2. Extract Symbols
        received_symbols = spectrum[self.carriers]
        
        # 3. Demodulate QPSK
        # Real > 0 -> bit 0 is 0, Real < 0 -> bit 0 is 1
        # Imag > 0 -> bit 1 is 0, Imag < 0 -> bit 1 is 1
        
        # Vectorized demodulation
        b0 = (received_symbols.real < 0).astype(int)
        b1 = (received_symbols.imag < 0).astype(int)
        
        # Interleave bits: [b0[0], b1[0], b0[1], b1[1], ...]
        decoded_bits = np.column_stack((b0, b1)).flatten().tolist()
        
        if return_symbols:
            return decoded_bits, received_symbols
            
        return decoded_bits, None
    
    def get_preamble_frame(self):
        """
        Generates a full-frame chirp sweep for preamble synchronization.
        Sweeps from 500Hz to 15kHz.
        """
        t = np.arange(self.frame_size) / self.sample_rate
        
        # Linear chirp: f(t) = f0 + k*t
        f0 = 500
        f1 = 15000
        k = (f1 - f0) / (self.frame_size / self.sample_rate)
        
        # Phase = integral of frequency
        phase = 2 * np.pi * (f0 * t + (k / 2) * t**2)
        chirp = 0.5 * np.sin(phase)
        
        # Apply Blackman window to ensure smooth start/end (no clicks)
        window = np.blackman(self.frame_size)
        return chirp * window

    def find_preamble_start(self, audio_stream, threshold=0.5):
        """
        Finds the start of the preamble sequence.
        Returns the index of the FIRST chirp frame found.
        """
        preamble = self.get_preamble_frame()
        
        # Optimization: Don't search the whole file if it's huge
        # Search first 5 seconds (should be enough to find preamble)
        search_len = min(len(audio_stream), 48000 * 5)
        search_region = audio_stream[:search_len]
        
        # Cross-correlation
        correlation = np.correlate(search_region, preamble, mode='valid')
        
        if len(correlation) == 0:
            return -1

        abs_corr = np.abs(correlation)
        max_val = np.max(abs_corr)
        
        if max_val < 0.1: # Signal too weak
            return -1
            
        # Find all points that are "peaks" (above threshold)
        # Since we send multiple preamble frames, we will see multiple peaks.
        # We want the very first one.
        peak_indices = np.where(abs_corr > max_val * threshold)[0]
        
        if len(peak_indices) > 0:
            return peak_indices[0]
            
        return -1
    
    def calculate_phase_variance(self, audio_frame):
        """
        Calculates the variance of the phase error for a QPSK frame.
        Lower is better. A perfect lock has variance ~0.
        A ring has high variance.
        """
        if len(audio_frame) != self.frame_size:
            return float('inf')

        spectrum = np.fft.rfft(audio_frame)
        received_symbols = spectrum[self.carriers]
        
        # QPSK: Raise to 4th power to remove modulation (map all points to 0 degrees)
        # z^4 should be real and negative (angle = pi)
        z4 = received_symbols ** 4
        
        # Calculate how spread out these angles are
        # We use the circular variance: 1 - |mean(z4 / |z4|)|
        # If they are all tight, mean vector length is 1 -> variance 0
        # If they are spread around a ring, mean vector length is 0 -> variance 1
        
        # Normalize magnitude first
        z4_norm = z4 / (np.abs(z4) + 1e-9)
        mean_vector = np.mean(z4_norm)
        variance = 1.0 - np.abs(mean_vector)
        
        return variance

# Helper for quick testing
if __name__ == "__main__":
    import av
    class AudioEncoder:
        def __init__(self, sample_rate=48000, bitrate_k=192, output_file="output.aac"):
            self.sample_rate = sample_rate
            self.bitrate_k = bitrate_k
            self.output_file = output_file
            self.container = None
            self.audio_stream = None
            self.create_encoder()

        def create_encoder(self):
            self.container = av.open(self.output_file, mode='w')
            self.audio_stream = self.container.add_stream('aac', rate=self.sample_rate)
            self.audio_stream.bit_rate = self.bitrate_k * 1000
            self.audio_stream.format = 'flt'
            self.audio_stream.layout = 'mono'

        def encode_audio(self, audio_samples):
            samples = np.array(audio_samples).astype(np.float32)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format='flt', layout='mono')
            frame.sample_rate = self.sample_rate
            for packet in self.audio_stream.encode(frame):
                self.container.mux(packet)

        def close(self):
            # Flush encoder
            for packet in self.audio_stream.encode(None):
                self.container.mux(packet)
            self.container.close()

    ofdm = AudioOFDM()
    print(f"OFDM Configured: {ofdm.num_carriers} carriers")
    print(f"Capacity: {ofdm.bits_per_frame} bits per frame ({ofdm.bits_per_frame * FRAME_RATE / 1000:.2f} kbps)")
    
    # Test Loopback with Offset
    import random

    test_file = "text.txt"

    with open(test_file, "rb") as f:
        data = f.read()
    
    data = data[:1048576] # 8 mbits
    test_bits = []
    for byte in data:  # Limit size for test
        for i in range(8):
            test_bits.append((byte >> (7 - i)) & 1)

    encoder = AudioEncoder(output_file="test_output.aac", bitrate_k=160)

    # Encode
    
    stream = []
    n_frames = (len(test_bits) + ofdm.bits_per_frame - 1) // ofdm.bits_per_frame # not counting preamble
    
    class BitIterator:
        def __init__(self, bits):
            self.bits = bits
            self.idx = 0
            self.total = len(bits)
        
        def get_bit(self):
            if self.idx < self.total:
                b = self.bits[self.idx]
                self.idx += 1
                return b
            else:
                return 0  # Pad with zeros if out of bits
        
        def frame_generator(self, size):
            # generator that generates 1600 bits at a time
            return [self.get_bit() for _ in range(size)]

    bit_iter = BitIterator(test_bits)
    for i in range(n_frames):
        bits_frame = bit_iter.frame_generator(ofdm.bits_per_frame)
        audio_frame = ofdm.encode_frame(bits_frame)
        stream.extend(audio_frame)
        assert len(stream) == SAMPLES_PER_FRAME
        current_frame = stream[:SAMPLES_PER_FRAME]
        current_frame = ofdm.inject_sync_pulse(current_frame)
        encoder.encode_audio(current_frame)
        stream = stream[SAMPLES_PER_FRAME:]  # Remove encoded part
        print(f"\rEncoded {i + 1}/{n_frames} frames...", end='', flush=True)
    print(f"\rEncoded {n_frames}/{n_frames} frames")

    encoder.close()

    # Read back the encoded file
    container = av.open("test_output.aac", mode='r')
    stream_audio = container.streams.audio[0]
    stream = []
    for frame in container.decode(stream_audio):
        data = frame.to_ndarray()
        stream.extend(data[0])  # Mono
    container.close()
    stream = np.array(stream, dtype=np.float32)

    # Decode using synchronization
    rough_data_start = ofdm.find_stream_start(stream)
    print(f"Correlated start offset: {rough_data_start} samples")
    best_offset = rough_data_start
    best_score = float('inf')
                    
    scan_range = range(-5, 5) 
                    
    print("Fine-tuning synchronization...")
    for delta in scan_range:
        test_idx = rough_data_start + delta

        test_frame = stream[test_idx : test_idx + 1600]
        score = ofdm.calculate_phase_variance(test_frame)
                        
        print(f"  Offset {delta}: Score {score:.4f}")
                        
        if score < best_score:
            best_score = score
            best_offset = test_idx

    print(f"Best offset found: {best_offset} (Score: {best_score:.4f})")
    # Decode from best offset
    decoded_stream = []
    received_symbols = []
    for bits, symbols in ofdm.decode_synchronized_stream(stream[best_offset:], return_symbols=True):
        decoded_stream.append(bits)
        received_symbols.append(symbols)

    # plot symbol constellation scatter
    # received_symbols is a list of complex64
    import matplotlib.pyplot as plt
    all_symbols = np.concatenate(received_symbols)
    plt.figure(figsize=(6,6))
    plt.scatter(all_symbols.real, all_symbols.imag, s=1, alpha=0.1)
    plt.title("Decoded QPSK Symbol Constellation")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid()
    plt.savefig("test_symbol_constellation.png")

    with open(test_file, "rb") as f:
        original_data = f.read()
    
    original_data = original_data[:1048576]  # 8 mbits
    original_bits = []
    for byte in original_data:
        for i in range(8):
            original_bits.append((byte >> (7 - i)) & 1)
    
    errors = np.zeros(len(decoded_stream), dtype=int)
    if decoded_stream:
        for i, bits in enumerate(decoded_stream):
            # Use numpy for easy comparison
            bits_arr = np.array(bits, dtype=int)
            start_bit = i * ofdm.bits_per_frame
            end_bit = start_bit + ofdm.bits_per_frame if start_bit + ofdm.bits_per_frame <= len(original_bits) else len(original_bits)
            bits_slice = original_bits[start_bit:end_bit]
            if len(bits_slice) < ofdm.bits_per_frame:
                bits_slice = bits_slice + [0] * (ofdm.bits_per_frame - len(bits_slice))
            errors[i] = np.sum(bits_arr != bits_slice)
            # print(f"Frame {i}: Decoded {len(bits)} bits, Errors: {errors[i]}")
        print("Sync and decode complete.")
        print(f"Total frames decoded: {len(decoded_stream)}")
        total_errors = np.sum(errors)
        total_bits = len(decoded_stream) * ofdm.bits_per_frame
        ber = total_errors / total_bits if total_bits > 0 else 0.0
        print(f"Total Bit Errors: {total_errors} / {total_bits} bits ({ber:.6e} BER)")
    else:
        print("Sync failed, no frames found.")