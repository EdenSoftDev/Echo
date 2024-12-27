import os
import yaml
import warnings

from torch.cuda import is_available as cuda_is_available
from whisper import load_model
from transformers import pipeline
from moviepy.video.io.VideoFileClip import VideoFileClip

model_config = yaml.load(open("./conf/model_config.yaml", "r"), Loader=yaml.FullLoader)


class CaptionParser:
    def __init__(self, video_path: str, model_name: str, model_path: str, model_type: str, language: str):
        self.video_path = video_path
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.language = language

        self.model = None
        self.audio_path = self.get_audio()

    def write_captions(self, sentenses: dict) -> None:
        os.makedirs("./output_txt", exist_ok=True)

        with open(os.path.join("./output_txt", self.video_path.split("/")[-1].replace(".mp4", ".txt")), "w",
                  encoding="utf-8") as f:
            for k, v in sentenses.items():
                f.write(f"{k[0]}-{k[1]}: {v}\n")

    def get_audio(self) -> str:
        assert os.path.exists(self.video_path), "Video path does not exist."

        _audio_path = self.video_path.replace(".mp4", ".wav")
        if not os.path.exists(_audio_path):
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(_audio_path)
        return _audio_path

    def do_whisper_transcribe(self) -> dict:
        return self.model.transcribe(self.audio_path, language=self.language, word_timestamps=True)

    def parse_captions_with_whisper(self) -> dict:
        self.model = load_model(os.path.join(model_config["default_model_path"], "whisper", self.model_name + ".pt"))

        if cuda_is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            warnings.warn("CUDA is not available, using CPU. Highly recommend using a GPU for faster inference.")

        print(f"Transcribing {self.video_path} with {self.model_name} model")
        output = self.do_whisper_transcribe()
        print(f"Transcription complete with {len(output['segments'])} segments")

        sentences = {}
        for segment in output['segments']:
            sentences[(round(segment['start'], 2), round(segment['end'], 2))] = segment['text']

        self.write_captions(sentences)
        print(f"Captions written to ./output_txt/{self.video_path.split('/')[-1].replace('.mp4', '.txt')}")

        return sentences

    def parse_captions_with_huggingface(self) -> dict:
        if not cuda_is_available():
            warnings.warn("CUDA is not available, using CPU. Highly recommend using a GPU for faster inference.")

        model = pipeline(
            task="automatic-speech-recognition",
            model=self.model_path,
            chunk_length_s=5,
            stride_length_s=1,
            model_kwargs={
                "attn_implementation": "sdpa",
            },
            device=0 if cuda_is_available() else -1
        )

        print(f"Transcribing {self.video_path} with {self.model_name} model")
        output = model(
            inputs=self.audio_path,
            return_timestamps=True,
            generate_kwargs={
                "language": self.language,
                "task": "transcribe"
            }
        )
        print(f"Transcription complete with {len(output['chunks'])} segments")

        sentences = {}
        for segment in output['chunks']:
            sentences[(round(segment['timestamp'][0], 2), round(segment['timestamp'][1], 2))] = segment['text']

        self.write_captions(sentences)
        print(f"Captions written to ./output_txt/{self.video_path.split('/')[-1].replace('.mp4', '.txt')}")

        return sentences

    def parse_captions(self) -> dict:
        assert self.video_path is not None, "Video path is required"

        if self.model_type == "whisper":
            return self.parse_captions_with_whisper()
        elif self.model_type == "huggingface":
            return self.parse_captions_with_huggingface()


if __name__ == "__main__":
    parser = CaptionParser(
        video_path="./example.mp4",
        model_name="whisper",
        model_path="facebook/wav2vec2-base-960h",
        model_type="huggingface",
        language="Chinese"
    )
    parser.parse_captions()