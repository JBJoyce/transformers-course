from math import floor, ceil
from typing import Optional, Union
from torch import tensor, int64

def time_to_sample_conv(start: float, stop: Optional[float]=None, *, sample_rate: Optional[int]=None, span: bool=False):
    """Converts timepoint(s) to audio sample index given sample rate

    Args:
        start (float): start time in seconds
        stop (Optional[float], optional): stop time in seconds
        sample_rate (int, optional): sample rate of audio file. Defaults to None.
        span (bool, optional): If given two timepoints, should return include all in-between values. Defaults to False.

    Returns:
        tensor: Returns either a single sample index, two sample indices, or a span depending on the arguments passed 
    """
    
    start_sample = floor(start * sample_rate)
    if stop is not None:
        stop_sample = ceil(stop * sample_rate)
        if span:
            return tensor([i for i in range(start_sample, stop_sample)], dtype=int64)
        return tensor([start_sample, stop_sample], dtype=int64)
    return tensor(start_sample, dtype=int64)

if __name__ == "__main__":
    audio_span_2 = (0.123, 0.4567)
    audio_span_1 = 52.5
    print(time_to_sample_conv(*audio_span_2, sample_rate=16000))
    print(time_to_sample_conv(audio_span_1, sample_rate=44100))
    print(time_to_sample_conv(*(0, 1), sample_rate=10, span=True))              
    