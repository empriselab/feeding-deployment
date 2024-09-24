"""Soft E-stop with a button."""

import time

import numpy as np
import numpy.typing as npt
import pyaudio
from threading import Lock
import argparse

class Button:
    """Physical button that connects over audio jack."""

    PYAUDIO_STREAM_TROUBLESHOOTING = (
        "The Pyaudio stream not opening error is often caused by another process using "
        "the microphone and/or audio device. To address this, terminate the code and "
        "try the following:\n"
        "  1. Close all applications (e.g., System Settings) that may be accessing "
        "audio devices.\n"
        "  2. If that still doesn't address it, run `sudo alsa force-reload`.\n"
        "     Wait a few (~5) secs after running this command to restart the node,\n"
        "     and note that you may have to run this command multiple times.\n"
        "Note that until this is addressed, the e-stop button will not be working."
    )

    def __init__(self, input_device_index) -> None:

        self.start_time = time.time()
        self.prev_data_arr: npt.NDArray | None = None
        self.detection_time: float | None = None
        self.max_threshold = 10000
        self.min_threshold = -10000

        self.audio = pyaudio.PyAudio()

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,  # The e-stop button is mono
                rate=48000,
                input=True,
                frames_per_buffer=4800,
                input_device_index=input_device_index,
                stream_callback=self.__audio_callback,
            )
        except OSError as exc:
            raise RuntimeError(
                (
                    f"Error opening audio device {0}. "
                    f"{Button.PYAUDIO_STREAM_TROUBLESHOOTING}\n\n"
                    f"Exception: {exc}"
                ),
            )
        
        self.button_lock = Lock()
        self.button_value = False

    def close(self) -> None:
        """Close the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def check(self) -> bool:
        """Check if the e-stop button has been pressed."""
        with self.button_lock:
            return self.button_value
        
    def reset(self) -> None:
        """Reset the e-stop button."""
        with self.button_lock:
            self.button_value = False

    def __audio_callback(
        self, data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple[bytes, int]:
        del frame_count, time_info, status  # unused

        # Skip the first few seconds of data, to avoid initial noise
        if time.time() - self.start_time < 2:
            return (data, pyaudio.paContinue)

        data_arr = np.frombuffer(data, dtype=np.int16)

        # Check if the e-stop button has been pressed
        if Button.rising_edge_detector(
            data_arr,
            self.prev_data_arr,
            self.max_threshold,
        ) or Button.falling_edge_detector(
            data_arr,
            self.prev_data_arr,
            self.min_threshold,
        ):
            if self.detection_time is None or time.time() - self.detection_time > 2:
                self.detection_time = time.time()
                with self.button_lock:
                    self.button_value = True

        # Return the data
        self.prev_data_arr = data_arr

        # print("In audio callback: ", data_arr)
        return (data, pyaudio.paContinue)

    @staticmethod
    def rising_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: npt.NDArray | None,
        threshold: int | float,
    ) -> bool:
        """Detects whether there is a rising edge in `curr_data_arr` that
        exceeds `threshold`. In other words, this function returns True if
        there is a point in `curr_data_arr` that is greater than `threshold`
        and the previous point is less than `threshold`.

        Although this method of detecting a rising edge is suceptible to noise
        (since it only requires two points to determine an edge), in practice
        the e-stop button's signal has little noise. If noise is an issue
        moving forward, we can add a filter to smoothen the signal, and then
        continue using this detector.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a rising edge

        Returns
        -------
        is_rising_edge: bool
            True if a rising edge was detected, False otherwise
        """
        is_above_threshold = curr_data_arr > threshold
        if np.any(is_above_threshold):
            first_index_above_threshold = np.argmax(is_above_threshold)
            # Get the previous value
            if first_index_above_threshold == 0:
                if prev_data_arr is None:
                    # If the first datapoint is above the threshold, it's not a
                    # rising edge
                    return False
                prev_value = prev_data_arr[-1]
            else:
                prev_value = curr_data_arr[first_index_above_threshold - 1]
            # If the previous value is less than the threshold, it is a rising edge
            return prev_value < threshold
        # If no point is above the threshold, there is no rising edge
        return False

    @staticmethod
    def falling_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: npt.NDArray | None,
        threshold: int | float,
    ) -> bool:
        """Detects whether there is a falling edge in `curr_data_arr` that
        exceeds `threshold`. In other words, this function returns True if
        there is a point in `curr_data_arr` that is less than `threshold` and
        the previous point is greater than `threshold`.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a falling edge

        Returns
        -------
        is_falling_edge: bool
            True if a falling edge was detected, False otherwise
        """
        # Flip all signs and call the rising edge detector
        return Button.rising_edge_detector(
            -curr_data_arr,
            None if prev_data_arr is None else -prev_data_arr,
            -threshold,
        )


if __name__ == "__main__":

    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="The index of the input device")
    args = parser.parse_args()

    # -1 is user emergency stop button
    # 9 is experimentor emergency stop button
    # 7 is transfer button

    if args.id is None:
        audio = pyaudio.PyAudio()
        device_indices = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # Only consider input devices
                device_indices.append(i)
                print(f"Device {i}: {device_info['name']}")
        raise ValueError("Please provide the input device index")

    button = Button(args.id)
    while True:
        if button.check():
            print("E-stop pressed!")
            break
        time.sleep(0.01)
    
    button.close()
