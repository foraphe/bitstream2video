import av
import numpy as np

class VideoEncoder:
    def __init__(self, width, height, frame_rate, output_file, options, audio_sample_rate=48000, bitrate_k=8000, abitrate_k=160, vcodec='libx264'):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.output_file = output_file
        self.options = options
        self.audio_sample_rate = audio_sample_rate
        self.container = None
        self.video_stream = None
        self.audio_stream = None
        self.bitrate_k = bitrate_k
        self.vcodec = vcodec
        self.abitrate_k = abitrate_k
        self.create_encoder()

    def create_encoder(self):
        self.container = av.open(self.output_file, mode='w')
        
        # Video Stream
        self.video_stream = self.container.add_stream(self.vcodec, rate=self.frame_rate)
        self.video_stream.width = self.width
        self.video_stream.height = self.height
        self.video_stream.pix_fmt = 'yuv420p'
        self.video_stream.options = self.options
        if self.bitrate_k > 0:
            self.video_stream.bit_rate = self.bitrate_k * 1000  # Convert kbps to bps
        else:
            self.video_stream.bit_rate = 0  # Let encoder decide bitrate based on CRF
        
        # Audio Stream (AAC)
        self.audio_stream = self.container.add_stream('aac', rate=self.audio_sample_rate)
        self.audio_stream.bit_rate = self.abitrate_k * 1000  # Convert kbps to bps
        self.audio_stream.format = 'flt'
        self.audio_stream.layout = 'mono'
        
        return self.container, self.video_stream
        # return self.container, self.video_stream, self.audio_stream
    
    def encode_frame(self, Y_plane, U_plane, V_plane):
        frame = av.VideoFrame(self.width, self.height, 'yuv420p')
        frame.planes[0].update(Y_plane)
        frame.planes[1].update(U_plane)
        frame.planes[2].update(V_plane)

        for packet in self.video_stream.encode(frame):
            self.container.mux(packet)

    def encode_audio(self, audio_samples):
        """
        Encodes audio samples.
        :param audio_samples: Numpy array of audio samples (float).
        """
        # Ensure float32 for PyAV
        samples = audio_samples.astype(np.float32)
        
        # Reshape for PyAV (channels, samples) - Mono
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            
        frame = av.AudioFrame.from_ndarray(samples, format='flt', layout='mono')
        frame.sample_rate = self.audio_sample_rate
        
        for packet in self.audio_stream.encode(frame):
            self.container.mux(packet)
    
    def finalize_encoder(self):
        # Flush video encoder
        for packet in self.video_stream.encode():
            self.container.mux(packet)
            
        # Flush audio encoder
        for packet in self.audio_stream.encode():
            self.container.mux(packet)
            
        self.container.close()

class VideoDecoder:
    def __init__(self, input_file):
        self.input_file = input_file
        self.container = av.open(self.input_file)
        self.video_stream = self.container.streams.video[0]
        self.audio_stream = self.container.streams.audio[0] if self.container.streams.audio else None

    def decode_frames(self):
        for frame in self.container.decode(self.video_stream):
            Y = np.array(frame.planes[0])
            U = np.array(frame.planes[1])
            V = np.array(frame.planes[2])
            yield Y, U, V

    def decode_audio(self):
        if not self.audio_stream:
            return
        for frame in self.container.decode(self.audio_stream):
            yield frame
