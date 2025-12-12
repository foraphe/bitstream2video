import cv2
import numpy as np
import math
import concurrent.futures
from scrambler import Scrambler
from reedsolo import RSCodec, ReedSolomonError
from datastream_ecc import DataStream, ECCDecoder

class Config:
    def __init__(self):
        self.premable_frames = 5
        assert self.premable_frames >= 1, "There must be at least one premable frame"

        self.video_width = 1920
        self.video_height = 960
        self.frame_rate = 30

        # 8 is basically the only grid size that works well
        # could be because encoder uses 8x8 DCT blocks internally
        self.grid_w = 8
        self.grid_h = 8

        # Even values almost all work, but 4x4 works best
        self.patch_w = 4
        self.patch_h = 4

        # Offsets within each grid cell
        # Anything non-zero makes decoding worse
        self.patch_offset_x = 0
        self.patch_offset_y = 0

        self.frames_per_symbol = 1

        # Since we don't have a pixel that's exactly at the peak of the sine wave
        # due to even patch size, we can have a bit more SNR by scaling the amplitude
        self.amplitude = 254 / math.sin(math.pi * ((self.patch_w - 1) // 2) / (self.patch_w - 1))
        print (f"Using amplitude: {self.amplitude}, highest pixel phase {((self.patch_w - 1) // 2)}/{(self.patch_w - 1)} ({self.amplitude * math.sin(math.pi * ((self.patch_w - 1) // 2) / (self.patch_w - 1)) :.4f})")
        # self.amplitude = 254
        self.dc_bias = 0

        # ECC parameters
        # They need to divide the number of bits per symbol
        # so that symbol boundaries align to ECC block boundaries
        self.ecc_bytes = 30
        self.ecc_data_bytes = 222

        assert self.video_width % self.grid_w == 0, "Video width must be divisible by grid width"
        assert self.video_height % self.grid_h == 0, "Video height must be divisible by grid height"
        
        self.grid_per_row = self.video_width // self.grid_w
        self.grid_per_col = self.video_height // self.grid_h

        # half-pixel and quarter-pixel ME could work
        # but we don't have enough resolution to distinguish the symbols
        # going for integer pixels shifts only
        self.codebook = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx, dy) != (0, 0):
                    self.codebook.append((dx, dy))

        assert math.log2(len(self.codebook)) % 1 == 0, "Codebook size must be a power of 2"

        self.bits_per_symbol = int(math.log2(len(self.codebook)))
        self.bits = []
        for i in range(len(self.codebook)):
            bits = []
            for b in reversed(range(self.bits_per_symbol)):
                bits.append((i >> b) & 1)
            self.bits.append(bits)

        print (f"Video Resolution: {self.video_width}x{self.video_height}")
        
        self.symbol_per_frame = self.grid_per_col * self.grid_per_row
        self.payload_per_frame = self.symbol_per_frame * self.bits_per_symbol # in bits

        assert self.payload_per_frame % 8 == 0, "Payload per frame must be a multiple of 8 bits (1 byte)"

        self.symbol_rate = self.frame_rate * self.symbol_per_frame
        self.payload_rate = self.symbol_rate * math.log2(len(self.codebook))

        print (f"Grid: {self.grid_per_row}x{self.grid_per_col} ({self.symbol_per_frame} symbols/frame)")
        print (f"Codebook Size: {len(self.codebook)} symbols ({int(math.log2(len(self.codebook)))} bits/symbol)")

        print(f"Symbol Rate: {self.symbol_rate} symbols/second")
        print(f"Payload Rate: {self.payload_rate / 1000} kbits/second")

cfg = Config()

'''
# We have to zone a frame into multiple blocks, and for simplicity just throw RS codes onto even frames
# at the exact same positions
# So effectively 100% parity overhead
def compute_rs_layout(cfg):
    # We need to find RS block size such that:
    # 1. The amount of bits within a block is divisible by 8 (1 byte)
    # 2. The number of blocks per frame is an integer
    # 3. Data bits in each block is smaller than 255/2 = 127 bytes (for RS(255, k) codes)
    bits_per_block = None
    blocks_per_frame = None
    for bpb in range(1024, 0, -8):
        if (cfg.payload_per_frame % bpb) != 0:
            continue
        num_blocks = cfg.payload_per_frame // bpb
        bytes_per_block = bpb // 8
        if bytes_per_block <= 127:
            bits_per_block = bpb
            blocks_per_frame = num_blocks
            break
    assert bits_per_block is not None, "Failed to compute RS block size"
    print(f"RS Layout: {blocks_per_frame} blocks/frame, {bits_per_block} bits/block ({bits_per_block // 8} bytes/block)")
    return bits_per_block, blocks_per_frame
cfg.rs_bits_per_block, cfg.rs_blocks_per_frame = compute_rs_layout(cfg)
assert cfg.rs_blocks_per_frame * cfg.rs_bits_per_block == cfg.payload_per_frame, "RS layout mismatch"
assert cfg.rs_bits_per_block % 8 == 0, "RS bits per block must be multiple of 8"
assert cfg.rs_bits_per_block > 0, "RS bits per block must be positive"
rsc = RSCodec(cfg.rs_bits_per_block // 8)
# Find the width and height for ECC blocks, trying to keep them as square as possible
def compute_ecc_block_size(cfg):
    total_blocks = cfg.rs_blocks_per_frame
    aspect_ratio = cfg.video_width / cfg.video_height
    best_w = 1
    best_h = total_blocks
    best_diff = float('inf')
    for h in range(1, total_blocks + 1):
        if total_blocks % h != 0:
            continue
        w = total_blocks // h
        current_aspect = w / h
        diff = abs(current_aspect - aspect_ratio)
        if diff < best_diff:
            best_diff = diff
            best_w = w
            best_h = h
    print(f"ECC Block Grid: {best_w}x{best_h} blocks")
    return best_w, best_h
cfg.ecc_blocks_w, cfg.ecc_blocks_h = compute_ecc_block_size(cfg)
assert cfg.ecc_blocks_w * cfg.ecc_blocks_h == cfg.rs_blocks_per_frame, "ECC block grid size mismatch"
'''

# For simplicity, append 30 bytes (80 symbols) of parity
# after every 222 bytes of data (592 symbols)
rsc = RSCodec(30)

import numpy as np
import cv2
import math
import random

def build_sinusoidal_patch(cfg, phase_x = 0, phase_y = 0):
    """
    Generate a 2D sinusoidal luma patch that fades to zero at the edges.
    """
    w = cfg.patch_w
    h = cfg.patch_h
    A = cfg.amplitude
    DC = cfg.dc_bias

    x = np.linspace(0, math.pi, w, dtype=np.float32)
    y = np.linspace(0, math.pi, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    patch = A * np.sin(X + phase_x * math.pi / (cfg.patch_w - 1)) * np.sin(Y + phase_y * math.pi / (cfg.patch_h - 1))
    patch = patch + DC

    return patch.astype(np.float32)

# Tried different patches, sine is still the best
'''
def build_sinusoidal_patch(cfg):
    """
    Generate a 2D sinusoidal luma patch that fades to zero at the edges.
    Using a 2D Gaussian-modulated sinusoid for better frequency characteristics.
    """
    _patches = [
        'gaussian_pulse',
        'windowed_sine',
        'windowed_lfm_chirp',
        'sine'
    ]
    
    patch_type = _patches[3]
    
    w = cfg.patch_w
    h = cfg.patch_h
    A = cfg.amplitude
    DC = cfg.dc_bias
    
    x = np.linspace(-1, 1, w, dtype=np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    if patch_type == 'gaussian_pulse':
        sigma = 0.5
        patch = A * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    elif patch_type == 'windowed_sine':
        freq = 5  # cycles across the patch
        window = np.hanning(w).reshape(1, w) * np.hanning(h).reshape(h, 1)
        patch = A * np.sin(2 * math.pi * freq * X) * window
    elif patch_type == 'windowed_lfm_chirp':
        f0 = 2  # start frequency
        f1 = 8  # end frequency
        t = (X + 1) / 2  # normalize to [0, 1]
        chirp_signal = np.sin(2 * math.pi * (f0 * t + (f1 - f0) / 2 * t**2))
        window = np.hanning(w).reshape(1, w) * np.hanning(h).reshape(h, 1)
        patch = A * chirp_signal * window
    elif patch_type == 'sine':
        patch = A * np.sin(math.pi * (X + 1) / 2) * np.sin(math.pi * (Y + 1) / 2)
    else:
        raise ValueError("Unknown patch type")
    
    patch = patch + DC

    return patch.astype(np.float32)
'''

class SinusoidalPatchEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.patch = build_sinusoidal_patch(cfg)

    def encode_frame(self, symbols):
        """
        Encode one frame’s grid of symbols into a Y plane.
        symbols is a list/array of length = cfg.symbol_per_frame
        where each symbol is an index into cfg.codebook.
        """
        cfg = self.cfg

        H = cfg.video_height
        W = cfg.video_width
        
        rows = cfg.grid_per_col
        cols = cfg.grid_per_row
        
        gh = cfg.grid_h
        gw = cfg.grid_w
        
        ph = cfg.patch_h
        pw = cfg.patch_w

        # Ensure symbols is a numpy array
        symbols = np.array(symbols, dtype=int)
        
        # Reshape to grid
        sym_grid = symbols[:rows*cols].reshape(rows, cols)

        # Create grid of cells: (rows, cols, gh, gw)
        Y_grid = np.zeros((rows, cols, gh, gw), dtype=np.float32)

        half_w = (gw - pw) // 2
        half_h = (gh - ph) // 2

        for code_idx, (dx, dy) in enumerate(cfg.codebook):
            # Find all cells that use this symbol
            mask = (sym_grid == code_idx)
            if not np.any(mask):
                continue
                
            # Phase in pixels
            # equal to fractional part of offsets
            phase_x = dx - round(dx)
            phase_y = dy - round(dy)
                
            # phase is in units of pixels, convert to radians
            patch_shifted = build_sinusoidal_patch(cfg, phase_x, phase_y)
                
            dx = round(dx)
            dy = round(dy)
                
            px = half_w + cfg.patch_offset_x + dx
            py = half_h + cfg.patch_offset_y + dy
                
            # Clip (safety)
            px = int(max(0, min(px, gw - pw)))
            py = int(max(0, min(py, gh - ph)))
                
            # Assign patch to all matching cells
            Y_grid[mask, py:py+ph, px:px+pw] += patch_shifted

        # Reshape to full image
        # (rows, cols, gh, gw) -> (rows, gh, cols, gw) -> (rows*gh, cols*gw)
        Y = Y_grid.transpose(0, 2, 1, 3).reshape(H, W)

        # Clip to video range
        Y = np.clip(Y, 0, 255).astype(np.uint8)

        # Create dummy UV planes for writing
        U = np.full((H // 2, W // 2), 128, dtype=np.uint8)
        V = np.full((H // 2, W // 2), 128, dtype=np.uint8)

        return Y, U, V
    
    def get_symbols_from_bytes(self, data_bytes):
        """
        Convert raw bytes into a list of symbols (3-bit values)
        """
        bytes_arr = np.frombuffer(data_bytes, dtype=np.uint8)
        bits = np.unpackbits(bytes_arr)
        
        # Pad bits if not divisible by bits_per_symbol
        remainder = len(bits) % self.cfg.bits_per_symbol
        if remainder != 0:
            print("Warn: Encoder: Padding bits to fit symbols")
            padding = self.cfg.bits_per_symbol - remainder
            bits = np.pad(bits, (0, padding), mode='constant')

        # Reshape to (num_symbols, bits_per_symbol)
        bits_reshaped = bits.reshape(-1, self.cfg.bits_per_symbol)
        
        # Convert bits to integer symbols
        # Powers of 2: [2^(n-1), ..., 1]
        powers = 1 << np.arange(self.cfg.bits_per_symbol - 1, -1, -1)
        symbols = bits_reshaped.dot(powers)

        return symbols.tolist()

    def get_bytes_from_symbols(self, symbols):
        """
        Convert a list of symbols (3-bit values) back into raw bytes
        """
        symbols = np.array(symbols, dtype=int)
        
        # Lookup bits
        # (num_symbols, bits_per_symbol)
        bits = self.bits_table[symbols]
        
        # Flatten and pack
        bytes_data = np.packbits(bits.flatten())
        
        return bytes_data.tobytes()

    def encode_ecc_frames(self, scrambled_data_bytes):
        """
        Encode the payload bytes along with their RS parity into two frames.

        Returns:
            (data_frame, parity_frame) where each frame is (Y, U, V).
        """
        cfg = self.cfg
        payload_bytes = cfg.payload_per_frame // 8
        block_bytes = cfg.rs_bits_per_block // 8
        blocks_per_frame = cfg.rs_blocks_per_frame

        if len(scrambled_data_bytes) != payload_bytes:
            raise ValueError(
                f"Expected {payload_bytes} bytes, got {len(scrambled_data_bytes)}"
            )

        data_arr = np.frombuffer(scrambled_data_bytes, dtype=np.uint8, count=payload_bytes)
        data_blocks = data_arr.reshape(cfg.ecc_blocks_h, cfg.ecc_blocks_w, block_bytes)

        parity_blocks = np.empty_like(data_blocks)
        data_blocks_flat = data_blocks.reshape(blocks_per_frame, block_bytes)
        parity_blocks_flat = parity_blocks.reshape(blocks_per_frame, block_bytes)

        for idx in range(blocks_per_frame):
            codeword = rsc.encode(bytes(data_blocks_flat[idx]))
            parity_blocks_flat[idx] = np.frombuffer(codeword[-block_bytes:], dtype=np.uint8)

        data_symbols = self.get_symbols_from_bytes(data_arr.tobytes())
        parity_symbols = self.get_symbols_from_bytes(parity_blocks_flat.reshape(-1).tobytes())

        data_frame = self.encode_frame(data_symbols)
        parity_frame = self.encode_frame(parity_symbols)

        return data_frame, parity_frame

class SinusoidalPatchDecoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.match_kernel, self.kernel_energy, self.offsets = self.create_kernels()
        # Pre-convert bits table to numpy for fast lookup
        self.bits_table = np.array(cfg.bits, dtype=np.uint8)
        
    def create_kernels(self):
        """
        Create matching kernels for each codebook entry.
        Returns a list of kernels.
        """
        cfg = self.cfg
        kernels = []
        kernel_energies = []
        offsets = []
        for dx, dy in cfg.codebook:
            phase_x = dx - round(dx)
            phase_y = dy - round(dy)
            dx_int = round(dx)
            dy_int = round(dy)
            kernel = build_sinusoidal_patch(cfg, phase_x, phase_y) - cfg.dc_bias
            energy = np.sum(kernel ** 2) + 1e-6
            kernels.append(kernel)
            kernel_energies.append(energy)
            offsets.append((dx_int, dy_int))
            
        return kernels, kernel_energies, offsets

    def decode_frame(self, Y, save_scores=False):
        """
        Decode one frame’s Y plane into a list of symbols.
        Y is a 2D numpy array of shape (H, W)
        """
        cfg = self.cfg
        
        rows = cfg.grid_per_col
        cols = cfg.grid_per_row
        gh = cfg.grid_h
        gw = cfg.grid_w
        ph = cfg.patch_h
        pw = cfg.patch_w
        
        # Reshape Y to isolate grid cells
        # (H, W) -> (rows, gh, cols, gw) -> (rows, cols, gh, gw)
        Y_cells = Y.reshape(rows, gh, cols, gw).transpose(0, 2, 1, 3)
        
        # Flatten to (num_cells, gh, gw)
        num_cells = rows * cols
        Y_cells_flat = Y_cells.reshape(num_cells, gh, gw).astype(np.float32)
        
        # Subtract DC bias (optional if we normalize patches, but good practice)
        Y_cells_flat -= cfg.dc_bias
        
        num_symbols = len(cfg.codebook)
        scores = np.zeros((num_cells, num_symbols), dtype=np.float32)
        
        half_w = (gw - pw) // 2
        half_h = (gh - ph) // 2
        
        # Iterate over codebook to compute scores
        for i in range(len(cfg.codebook)):
            kernel = self.match_kernel[i]
            # kernel_energy = self.kernel_energy[i] # Not needed for NCC
            dx_int, dy_int = self.offsets[i]

            # normalize kernel
            kernel = (kernel - np.mean(kernel)) / (np.std(kernel) + 1e-6)
            
            px = half_w + dx_int + cfg.patch_offset_x
            py = half_h + dy_int + cfg.patch_offset_y
            
            # Clip
            px = int(max(0, min(px, gw - pw)))
            py = int(max(0, min(py, gh - ph)))
            
            # Extract patches for this codebook entry from all cells
            # (num_cells, ph, pw)
            patches = Y_cells_flat[:, py:py+ph, px:px+pw]

            # Normalize patches locally
            p_mean = np.mean(patches, axis=(1, 2), keepdims=True)
            p_std = np.std(patches, axis=(1, 2), keepdims=True) + 1e-6
            patches_norm = (patches - p_mean) / p_std
            
            # Compute correlation
            # patches * kernel -> sum over (1, 2)
            # Divide by N (number of pixels) to get Pearson correlation coefficient [-1, 1]
            N = ph * pw
            dot_prod = np.sum(patches_norm * kernel, axis=(1, 2)) / N
            
            scores[:, i] = dot_prod
            
        # Find best symbol
        best_symbols = np.argmax(scores, axis=1)
        
        if save_scores:
            best_scores = np.max(scores, axis=1)
            return best_symbols.tolist(), best_scores.tolist()
        else:
            return best_symbols.tolist(), None
    
    def get_bytes_from_symbols(self, symbols):
        """
        Convert a list of symbols (3-bit values) back into raw bytes
        """
        symbols = np.array(symbols, dtype=int)
        
        # Lookup bits
        # (num_symbols, bits_per_symbol)
        bits = self.bits_table[symbols]
        
        # Flatten and pack
        bytes_data = np.packbits(bits.flatten())
        
        return bytes_data.tobytes()

# -------------------------------
# Example standalone usage
# -------------------------------

def bytes_generator(seed):
    random.seed(seed)
    while True:
        # Generate random byte
        yield random.getrandbits(8)

if __name__ == "__main__":
    import time 
    from videoencoder import VideoEncoder, VideoDecoder

    FILENAME='random.txt'

    encoder = SinusoidalPatchEncoder(cfg)
    decoder = SinusoidalPatchDecoder(cfg)
    scrambler = Scrambler()
    dec_scrambler = Scrambler()
    datasource = DataStream(FILENAME, cfg, scrambler = scrambler, bits_per_payload=cfg.bits_per_symbol)
    
    def get_frame_score_stats(scores):
        scores = np.array(scores)
        nmin = np.min(scores)
        nmax = np.max(scores)
        nmean = np.mean(scores)
        nstd = np.std(scores)
        return nmin, nmax, nmean, nstd, nmin - 2 * nstd, nmax + 2 * nstd
    
    def get_real_video_bitrate(cfg, filename, n_frames):
        with open(filename, "rb") as f:
            f.seek(0, 2)
            filesize = f.tell()
        print(f"File size: {filesize} bytes")
        duration_seconds = n_frames / cfg.frame_rate
        bitrate_kbps = (filesize * 8) / duration_seconds / 1000
        return bitrate_kbps
    
    # We are using CRF, so basically uncap the bitrate
    target_bitrate_kbps = -1
    videncoder = VideoEncoder(
        width=cfg.video_width,
        height=cfg.video_height,
        frame_rate=cfg.frame_rate,
        output_file='test_output.mp4',
        options={'preset': 'veryslow', 'crf': '24'},
        vcodec = 'libx264',
        bitrate_k=0
    )
    # print(f"Using target bitrate: {target_bitrate_kbps} kbps")
    # print(f"Ratio: {target_bitrate_kbps / (cfg.payload_rate / 1000) :.2f}x ({(cfg.payload_rate / 1000) / target_bitrate_kbps * 100 :.2f}%)")
    # cfg.real_data_rate = cfg.payload_rate / ((cfg.ecc_data_bytes + cfg.ecc_bytes) / cfg.ecc_data_bytes)
    # print(f"Effective data rate after ECC: {cfg.real_data_rate / 1000 :.2f} kbps ({cfg.real_data_rate / target_bitrate_kbps / 10 :.2f}%)")
    frame_count = datasource.filesize * 8 // cfg.bits_per_symbol // cfg.symbol_per_frame + 1
    RUN_CNT = frame_count
    print(f"Total frames to encode: {RUN_CNT}")

    
    all_scores = []

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(RUN_CNT):
            print(f"\rDispatching frame {_+1}/{RUN_CNT}", end='', flush=True)
            symbols = datasource.get_batch(cfg.symbol_per_frame)

            # Submit encoding to thread pool
            future = executor.submit(encoder.encode_frame, symbols)
            futures.append(future)

            if len(futures) >= 8:
                future = futures.pop(0)
                Y, U, V = future.result()
                videncoder.encode_frame(Y, U, V)
        
        for future in futures:
            Y, U, V = future.result()
            videncoder.encode_frame(Y, U, V)
    print("\nEncoding completed.")
    
    all_scores = []
    decoded_data = []
    videncoder.finalize_encoder()
    viddecoder = VideoDecoder('test_output.mp4')
    decoder = SinusoidalPatchDecoder(cfg)
    eccdecoder = ECCDecoder(cfg, descrambler=dec_scrambler)
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for (idx, frame) in enumerate(viddecoder.decode_frames()):
            print(f"\rDispatching decode frame {idx+1}/{RUN_CNT}", end='', flush=True)
            Y, U, V = frame
            Y = Y[:cfg.video_height * cfg.video_width].reshape(cfg.video_height, cfg.video_width)
            future = executor.submit(decoder.decode_frame, Y, True)
            futures.append(future)

            if len(futures) >= 8:
                future = futures.pop(0)
                decoded_symbols, scores = future.result()
                all_scores.append(get_frame_score_stats(scores)) 
                decoded_bytes = decoder.get_bytes_from_symbols(decoded_symbols)
                eccdecoder.add_bytes(decoded_bytes)
        
        for future in futures:
            decoded_symbols, scores = future.result()
            all_scores.append(get_frame_score_stats(scores)) 
            decoded_bytes = decoder.get_bytes_from_symbols(decoded_symbols)
            eccdecoder.add_bytes(decoded_bytes)
        print()

    # Before we have a proper sideband for metadata, send the exact expected length
    data, err_mask, fail_mask = eccdecoder.decode_buffer(datasource.filesize)
    decoded_data.extend(data)
    print("\nDecoding completed.")
    print(f"Data size: {len(decoded_data)} bytes")
    print(f"Errors corrected: {np.sum(err_mask)}, unrecoverable: {np.sum(fail_mask)}")

    # Compare
    idx = 0
    errors = 0
    error_frames = np.zeros(RUN_CNT, dtype=int)
    err_heat_map = np.zeros((cfg.grid_per_col, cfg.grid_per_row), dtype=int)

    with open('decoded_output.bin', 'wb') as f:
        f.write(bytearray(decoded_data))
    
    with open(FILENAME, 'rb') as f:
        all_input_data = f.read()
    
    for frame_idx in range(RUN_CNT):
        frame_data_bytes = decoded_data[frame_idx * (cfg.payload_per_frame // 8):(frame_idx + 1) * (cfg.payload_per_frame // 8)]
        input_data_bytes = all_input_data[frame_idx * (cfg.payload_per_frame // 8):(frame_idx + 1) * (cfg.payload_per_frame // 8)]
        frame_errors = 0
        for bidx in range(len(frame_data_bytes)):
            if idx >= len(all_input_data):
                break
            if frame_data_bytes[bidx] != input_data_bytes[bidx]:
                errors += 1
                frame_errors += 1
                byte_pos = bidx * 8
                for bit_pos in range(8):
                    bit_idx = byte_pos + bit_pos
                    symbol_idx = bit_idx // cfg.bits_per_symbol
                    grid_row = symbol_idx // cfg.grid_per_row
                    grid_col = symbol_idx % cfg.grid_per_row
                    err_heat_map[grid_row, grid_col] += 1
            idx += 1
        error_frames[frame_idx] = frame_errors

    if errors == 0:
        print("Final data verification passed!") 
    else:
        print(f"Final data verification failed with {errors} errors. BER = {errors / len(all_input_data) * 100 :.6f}%")
        print(f"Worse frame: {np.max(error_frames)} errors ({np.max(error_frames) / (cfg.payload_per_frame // 8) * 100 :.6f}%)")
    # Plot scores scatter plot
    import matplotlib.pyplot as plt
    frame_indices = list(range(len(all_scores)))
    mean_scores = np.array([s[2] for s in all_scores])
    min_scores = np.array([s[0] for s in all_scores])
    max_scores = np.array([s[1] for s in all_scores])
    trust_min = np.array([s[4] for s in all_scores])
    trust_max = np.array([s[5] for s in all_scores])
    std_scores = np.array([s[3] for s in all_scores])

    # Combine 
    
    # Plot mean scores with min/max and trust intervals
    plt.fill_between(frame_indices, min_scores, max_scores, color='lightgray', label='Min/Max Scores')
    plt.fill_between(frame_indices, mean_scores - 3 * std_scores, mean_scores + 3 * std_scores, color='lightblue', label='Trust Interval (±3σ)')
    plt.plot(frame_indices, mean_scores, color='blue', label='Mean Score')
    plt.xlabel('Frame Index')
    plt.ylabel('Correlation Score')
    plt.title('Frame Decoding Correlation Scores')
    plt.legend()
    plt.grid()
    plt.savefig('test_frame_scores.png')
    
    # plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(err_heat_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of Errors')
    plt.title('Error Heatmap (Row vs Column)')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.savefig('test_error_heatmap.png')

    # Plot errors per frame
    plt.figure()
    plt.bar(range(len(error_frames)), error_frames)
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Errors')
    plt.title('Number of Errors per Frame')
    plt.savefig('test_errors_per_frame.png')

    # Plot corrected error over blocks
    plt.figure()
    plt.bar(range(len(err_mask)), err_mask, label='Corrected Errors')
    plt.bar(range(len(fail_mask)), fail_mask, label='Unrecoverable Errors', bottom=err_mask)
    plt.xlabel('ECC Block Index')
    plt.ylabel('Number of Errors')
    plt.title('Errors per ECC Block')
    plt.legend()
    plt.savefig('test_ecc_block_errors.png')
    print("Test completed. Plots saved as PNG files.")

