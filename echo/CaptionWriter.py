import os
import re
import yaml
import pysrt

from moviepy.video import fx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy import VideoFileClip
from moviepy.video.VideoClip import TextClip


class CaptionWriter:
    def __init__(self, video_path: str):
        self.caption_config = yaml.load(
            open("./conf/caption_config.yaml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader
        )
        self.video_path = video_path
        self.video_clip = VideoFileClip(video_path)
        self.frame_height = self.video_clip.h
        self.frame_width = self.video_clip.w

        self.caption_clips = []

    def generate_caption_clips(self, start: float, end: float, text: str) -> None:
        text_clip = TextClip(
            text=text,
            font=self.caption_config["font"],
            font_size=self.caption_config["font_size"],
            color=self.caption_config["color"],
            stroke_color=self.caption_config["stroke_color"],
            stroke_width=self.caption_config["stroke_width"],
            duration=end - start,
        )

        text_clip = text_clip.with_start(start).with_end(end)
        text_clip = text_clip.with_effects([
            fx.FadeIn(duration=self.caption_config["fade_in"]),
            fx.FadeOut(duration=self.caption_config["fade_out"])
        ])
        text_clip = text_clip.with_position(("center", self.frame_height - self.caption_config["y_position"]), relative=True)

        self.caption_clips.append(text_clip)

    def generate_caption_clips_and_srt(self) -> None:
        txt_path = os.path.join("./output_txt", self.video_path.split("/")[-1].replace(".mp4", ".txt"))

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            subs = []
            line_count = 0
            for line in lines:
                line_count += 1

                if not re.match(r"^(\d+\.\d+)-(\d+\.\d+): (.+)$", line):
                    continue

                start = float(line.split("-")[0])
                end = float(line.split(":")[0].split("-")[1])
                text = line.split(": ")[1]

                if end - start < self.caption_config["fade_in"] + self.caption_config["fade_out"]:
                    continue

                self.generate_caption_clips(start, end, text)

                subs.append(pysrt.SubRipItem(index=len(subs), start=pysrt.srttime.SubRipTime(seconds=start),
                                             end=pysrt.srttime.SubRipTime(seconds=end), text=text))

            subs = pysrt.SubRipFile(subs)

            subs.save(os.path.join("./output_srt", self.video_path.split("/")[-1].replace(".mp4", ".srt")))

    def write_captions(self, video_path: str) -> None:
        os.makedirs("./output_srt", exist_ok=True)
        os.makedirs("./output_video", exist_ok=True)

        print(f"Writing captions to video {video_path}...")

        txt_path = os.path.join("./output_txt", video_path.split("/")[-1].replace(".mp4", ".txt"))
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Captions file not found. Please run parse mode first.")

        self.generate_caption_clips_and_srt()

        self.caption_clips.insert(0, self.video_clip)

        final = CompositeVideoClip(self.caption_clips)
        final.write_videofile(os.path.join("./output_video", video_path.split("/")[-1]),
                              audio_codec="aac", preset="ultrafast")


if __name__ == "__main__":
    caption_writer = CaptionWriter("./example.mp4")
    caption_writer.write_captions("./example.mp4")
