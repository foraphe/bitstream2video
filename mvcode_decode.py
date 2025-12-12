from mvcode import Config, SinusoidalPatchDecoder
from scrambler import Scrambler
from datastream_ecc import ECCDecoder
from videoencoder import VideoDecoder
from concurrent.futures import ThreadPoolExecutor
from metadata import Metadata
import numpy as np
import sys

assert len(sys.argv) == 2 or len(sys.argv) == 3, "Usage: python mvcode_decode.py <input_video_file> [original_data_file]"

def get_frame_score_stats(scores):
    scores = np.array(scores)
    nmin = np.min(scores)
    nmax = np.max(scores)
    nmean = np.mean(scores)
    nstd = np.std(scores)
    return nmin, nmax, nmean, nstd, nmin - 2 * nstd, nmax + 2 * nstd

cfg = Config()
decoder = SinusoidalPatchDecoder(cfg)
scrambler = Scrambler()
ecc_decoder = ECCDecoder(cfg, descrambler=scrambler)
vid_decoder = VideoDecoder(sys.argv[1])

# This decides whether we compare original data and report metrics
running_test = len(sys.argv) == 3
if running_test:
    with open(sys.argv[2], "rb") as f:
        original_data = f.read()
    total_bytes = len(original_data)
    print("=== Running in test mode, will not save file ===")
    print(f"Original data size: {total_bytes} bytes")

decoded_data = bytearray()
futures = []
all_scores = []

# frames is a generator
frames = vid_decoder.decode_frames()

# decode premable frames
for idx in range(cfg.premable_frames):
    print(f"\rDecoding premable frame {idx+1}/{cfg.premable_frames}", end='', flush=True)
    frame = next(frames)
    Y, U, V = frame
    Y = Y[:cfg.video_height * cfg.video_width].reshape((cfg.video_height, cfg.video_width))
    symbols, scores = decoder.decode_frame(Y, running_test)
    if running_test:
        all_scores.append(get_frame_score_stats(scores))
    chunk = decoder.get_bytes_from_symbols(symbols)
    if (idx == 0):
        # First premable frame contains metadata
        meta_data = chunk
        # Just assume we don't end with zeros for simplicity
        # So strip trailing zeros
        meta_data = meta_data.rstrip(b'\x00')
        meta = Metadata(max_frame_size=cfg.payload_per_frame // 8)
        meta_dic = meta.extract_metadata(meta_data)
        print("\nExtracted metadata:")
        for k, v in meta_dic.items():
            print(f"  {k}: {v}")
        if running_test:
            assert meta_dic["filesize"] == ecc_decoder.get_size_with_ecc(total_bytes), "Extracted filesize does not match original data size."
print("\nPremable frames decoded.")

with ThreadPoolExecutor() as executor:
    for idx, frame in enumerate(frames):
        print(f"\rDecoding frame {idx+1}", end='', flush=True)
        Y, U, V = frame
        n_frames = idx + 1
        futures.append(executor.submit(decoder.decode_frame, Y, True))
        if len(futures) >= 10:
            future = futures.pop(0)
            chunk, scores = future.result()
            all_scores.append(get_frame_score_stats(scores))
            chunk = decoder.get_bytes_from_symbols(chunk)
            ecc_decoder.add_bytes(chunk)
    # Process remaining futures
    for future in futures:
        chunk, scores = future.result()
        all_scores.append(get_frame_score_stats(scores))
        chunk = decoder.get_bytes_from_symbols(chunk)
        ecc_decoder.add_bytes(chunk)
    
    print()
    n_total_bytes = meta_dic["filesize"]
    data, err_mask, fail_mask = ecc_decoder.decode_buffer(n_total_bytes)
    decoded_data.extend(data)

print(f"Decoded total of {len(decoded_data)} bytes.")
real_file_size = meta_dic["filesize_real"]
if real_file_size < len(decoded_data):
    decoded_data = decoded_data[:real_file_size]
    print(f"Removed padding zeros to real file size: {real_file_size} bytes.")

if meta_dic['filename']:
    output_filename = f"decoded_{meta_dic['filename']}"
else:
    output_filename = "decoded_output.bin"
if not running_test:
    with open(output_filename, "wb") as f:
        f.write(decoded_data)
    print(f"Decoded data written to {output_filename}")

if running_test:
    decoded_bytes = bytes(decoded_data[:total_bytes])
    
    def get_real_video_bitrate(cfg, filename, n_frames):
        with open(filename, "rb") as f:
            f.seek(0, 2)
            filesize = f.tell()
        duration_seconds = n_frames / cfg.frame_rate
        bitrate_kbps = (filesize * 8) / duration_seconds / 1000
        return bitrate_kbps
    
    real_bitrate_kbps = get_real_video_bitrate(cfg, sys.argv[1], n_frames)
    print(f"Measured video bitrate: {real_bitrate_kbps:.2f} kbps")
    print(f"Ratio: {real_bitrate_kbps / (cfg.payload_rate / 1000) :.2f}x ({(cfg.payload_rate / 1000) / real_bitrate_kbps * 100 :.2f}%)")
    cfg.real_data_rate = cfg.payload_rate / ((cfg.ecc_data_bytes + cfg.ecc_bytes) / cfg.ecc_data_bytes)
    print(f"Effective data rate after ECC: {cfg.real_data_rate / 1000 :.2f} kbps (channel util% {cfg.real_data_rate / real_bitrate_kbps / 10 :.2f}%)")
    
    # Compare and report metrics
    bit_errors = 0  
    for b1, b2 in zip(original_data, decoded_bytes):
        diff = b1 ^ b2
        bit_errors += bin(diff).count('1')
    total_bits = total_bytes * 8
    ber = bit_errors / total_bits
    print(f"Total bit errors: {bit_errors} out of {total_bits} bits")
    print(f"Bit Error Rate (BER): {ber:.6e}")
    print(f"ECC corrected: {err_mask.sum()} bytes, failed: {fail_mask.sum()} blocks")
    
    idx = 0

    import matplotlib.pyplot as plt
    frame_indices = list(range(len(all_scores)))
    mean_scores = np.array([s[2] for s in all_scores])
    min_scores = np.array([s[0] for s in all_scores])
    max_scores = np.array([s[1] for s in all_scores])
    trust_min = np.array([s[4] for s in all_scores])
    trust_max = np.array([s[5] for s in all_scores])
    std_scores = np.array([s[3] for s in all_scores])
    
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
    print("Saved decoding metrics to test_frame_scores.png")
    