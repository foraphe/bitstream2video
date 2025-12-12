# Bitstream Video Encoder

**This project is still in early development. The code and documentation may change frequently without notice, optimization and code quality are not of primary concern at this stage, and backwards-compatibility is not guaranteed.**

This project implements a software modem that encodes arbitrary binary files into a video stream. Unlike traditional barcodes or QR codes, this system uses shifting sinusoidal patches to store data, which makes it more efficiently use the available video channel bandwidth.

The main difference of this project from similar projects is that instead of trying to encode more data into frames, it tries to maximize the video channel utilization (defined as the ratio of actual data bits transmitted per second to the average bit rate of the video), and therefore increasing the reliability of data transmission even after a few lossy compression passes. See the "Technical Details" section for more information.

The goal of this project is to create a robust and efficient method for storing and transmitting binary data through video streams, in the hopes that we can create a target data rate for a given (arbitrary, black-box) video codec like ones used by media streaming services or social media platforms, and have the data decode reliably after compression.

## Usage

### Prerequisites
*   Python 3.x
*   `av` (PyAV)
*   `numpy`
*   `opencv-python`
*   `reedsolo`

`opencv-python` is currently imported by not used, this will be fixed

### Encoding
To encode a file into a video:

```bash
python mvcode_encode.py <input_file> <output_video_file>
```

If `<output_video_file>` is not specified, it defaults to `output_video.mp4`.
The video file name should contain an extension, and any container format supporting the used codecs (H.264 in the current implementation) and supported by `PyAV` can be used.

### Decoding
To decode a video back into the original file:

```bash
python mvcode_decode.py <input_video_file> [original_input_file]
```

In the current implementation, the output file is named `decoded_<original_file_name>` where the original file name is retrieved from metadata stored within the video.

When an original input file is provided, the script will run in a "test mode", and instead of creating a decoded file, it runs a byte-wise comparison between the original and decoded data, and reports the number of errors. A graph of average decoder correlation values per frame will also be saved to `test_frame_scores.png`.

### Testing individual modules
Some modules can be tested individually by running their respective scripts, e.g.:

```bash
python scrambler.py
python metadata.py
python audioofdm.py
python mvcode.py
```

The `mvcode.py` test will generate several metrics, including correlation plots, error coordinate heatmaps, errors per frame, and errors per ECC block. These will be saved as PNG files in the current directory.


## TODO
* Store configuration parameters in the video metadata and use them when possible instead of assuming defaults.
* Make use of the audio channel for side-band, synchronization, or additional data.
* Optimize memory footprint during encoding and decoding.
* Add multi-threading to ECC block decoding, as this is currently taking a significant amount of time.
* Make the encoder accept more flexible data rates than the current implementation where we have to change video resolution or frame rate.

## Technical Details
The core idea of this project is to first divide the video frame into a grid of channels, and encode a sinusoidal patch within each channel. The position of the patch within the channel can carry data bits.
By creating small movements (+-1 across frames) of the patches, these information presumably map to inter-frame Motion Vectors (MVs) during video compression, which is shown to be more efficient than storing intra-frame symbols like PSK-encoded grid cells or PAM-encoded luma/chroma blocks.

Before encoding, we first add Reed-Solomon symbols to each chunk of data (currently, it's 30 RS bytes after every 222 data bytes). The resulting byte stream is then scrambled using a linear-feedback shift register (LFSR)-based scrambler to create a Pseudo-Noise (PN)-like sequence. This helps balancing the stream across channels, which _might_ cause issues for the codec, but this needs further investigation.

The choice of RS parameters is made so that both ECC and data bytes within a single ECC chunk would fit into the 3-bit-per-symbol payload scheme and align to boundaries. ECC is done in a "filter" fashion, where the core encoder isn't aware of ECC, and stream length is adjusted accordingly.

Currently, we use block size of 8x8 pixels (to align to DCT blocks) with a codebook of 8 positions around the center position of the block encoding 3 bits per symbol. This gives us a gross symbol rate of 2592 kbits per second (28800 blocks per frame, 3 bits per block, and 30 frames per second) into a 1920x960 30FPS video. With an initial compression run with `libx264` on raw data at CRF=20, followed by another run targeting 5.5Mbps, one such video could be compressed down to around 6Mbps, and channel utilization of around 38% (around 43% if ECC symbols are considered part of input data) is achieved. With a single `SVT-AV1` compression run, peak channel utilization of around 40% can be achieved, but `SVT-AV1` seems to be less friendly and causes large blocks of data to be encoded significantly darker than the rest, especially around the edges, when we run multiple compression passes, usually alongside loss of details and preventing full recovery of the data when targeting channel utilization above 40%.

Currently, a fixed-length premable of 5 frames will be inserted at the beginning of the video. The first preable frame also stored a serialized JSON object containing the metadata (currently, we store the file name, original size in bytes, and size after ECC/padding). The metadata is protected by its own, long RS symbols, and is also scrambled using the same scrambler as the main data stream.

## License
This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.
