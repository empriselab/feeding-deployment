import pyaudio

class EStopManager:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.streams = []
        self.device_indices = self.get_device_indices()
        print(f"Found audio devices: {self.device_indices}")

        # Open streams for each device
        for index in self.device_indices:
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,  # The e-stop button is mono
                    rate=48000,
                    input=True,
                    frames_per_buffer=4800,
                    input_device_index=9,
                    stream_callback=self.__audio_callback,
                )
                self.streams.append(stream)
            except OSError as exc:
                raise RuntimeError(
                    (
                        f"Error opening audio device {index}. "
                        f"{EStopManager.PYAUDIO_STREAM_TROUBLESHOOTING}\n\n"
                        f"Exception: {exc}"
                    )
                )

    def get_device_indices(self):
        device_indices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # Only consider input devices
                device_indices.append(i)
        return device_indices

    def __audio_callback(self, in_data, frame_count, time_info, status):
        # Handle the audio data from the e-stop button here
        print(f"Received data from device...")
        return (in_data, pyaudio.paContinue)

    def close(self):
        for stream in self.streams:
            stream.stop_stream()
            stream.close()
        self.audio.terminate()

# Example usage
estop_manager = EStopManager()

# Make sure to close the streams when you're done
# estop_manager.close()
