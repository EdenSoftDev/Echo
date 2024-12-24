import os
import re
import yaml
import pysrt
import moviepy


def write_captions(srt_path: str) -> None:
    os.makedirs("../output_srt", exist_ok=True)
    os.makedirs("../output_video", exist_ok=True)
    pass