from mvcode import Config, SinusoidalPatchEncoder
from scrambler import Scrambler
from datastream_ecc import DataStream
from videoencoder import VideoEncoder
from concurrent.futures import ThreadPoolExecutor
# from audioofdm import AudioOFDM
from metadata import Metadata
import numpy as np
import sys

assert len(sys.argv) == 2 or len(sys.argv) == 3, "Usage: python mvcode_encode.py <input_data_file> [output_video_file]"

cfg = Config()
input_file = sys.argv[1]

print("Creating video encoder, x264, crf=20, preset=veryslow")
output_filename = sys.argv[2] if len(sys.argv) == 3 else "output_video.mp4"
print(f"Output video file: {output_filename}")
vid_encoder = VideoEncoder(
    width=cfg.video_width,
    height=cfg.video_height,
    frame_rate=cfg.frame_rate,
    bitrate_k=0,
    output_file=output_filename,
    options={'crf': '20', 'preset': 'veryslow'},
)

# audio_ofdm = AudioOFDM(sample_rate=48000)
scrambler = Scrambler()
data_stream = DataStream(input_file, cfg, scrambler=scrambler, bits_per_payload=cfg.bits_per_symbol)
encoder = SinusoidalPatchEncoder(cfg)

filename_safe = input_file.split("/")[-1]
meta = Metadata(max_frame_size=cfg.payload_per_frame // 8)
meta.prepare_encoder({
    "filename": filename_safe,
    "filesize": data_stream.filesize,
    "filesize_real": data_stream.filesize_real,
})

# The first premable frame encodes metadata
meta_bytes = meta.get_encoded_bytes()
# Pad meta_bytes to fill a full frame
if len(meta_bytes) < cfg.payload_per_frame // 8:
    meta_bytes += bytes(cfg.payload_per_frame // 8 - len(meta_bytes))
meta_symbols = data_stream.bytes_to_symbols(meta_bytes)

print(f"Encoding metadata into first preamble frame, {len(meta_bytes)} bytes.")
Y, U, V = encoder.encode_frame(meta_symbols)
vid_encoder.encode_frame(Y, U, V)

# Then add random data premable frames
for i in range(cfg.premable_frames - 1):
    random_bytes = np.random.randint(0, 256, size=(cfg.payload_per_frame // 8,), dtype=np.uint8).tobytes()
    random_symbols = data_stream.bytes_to_symbols(random_bytes)
    Y, U, V = encoder.encode_frame(random_symbols)
    vid_encoder.encode_frame(Y, U, V)

print("Preamble frames encoded.")

frame_count = data_stream.filesize * 8 // cfg.bits_per_symbol // cfg.symbol_per_frame + 1
print(f"Encoding {data_stream.filesize} bytes into {frame_count} frames.")

futures = []
with ThreadPoolExecutor() as executor:
    for frame_idx in range(frame_count):
        print(f"\rEncoding frame {frame_idx+1}/{frame_count}", end='', flush=True)
        symbols = data_stream.get_batch(cfg.symbol_per_frame)

        future = executor.submit(encoder.encode_frame, symbols)
        futures.append(future)

        if len(futures) >= 10:
            future = futures.pop(0)
            Y_plane, U_plane, V_plane = future.result()
            vid_encoder.encode_frame(Y_plane, U_plane, V_plane)
        
    # Process remaining futures
    for future in futures:
        Y_plane, U_plane, V_plane = future.result()
        vid_encoder.encode_frame(Y_plane, U_plane, V_plane)

print("\nVideo encoding complete.")
vid_encoder.finalize_encoder()
