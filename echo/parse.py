import os
import yaml
import warnings

import torch
from whisper import load_model
from moviepy.video.io.VideoFileClip import VideoFileClip

settings = yaml.load(open("./conf/config.yaml", "r"), Loader=yaml.FullLoader)


def write_captions(video_path: str, sentenses: dict) -> None:
    os.makedirs("./output_txt", exist_ok=True)

    with open(os.path.join("./output_txt", video_path.split("/")[-1].replace(".mp4", ".txt")), "w") as f:
        for k, v in sentenses.items():
            f.write(f"{k[0]}-{k[1]}: {v}\n")


def get_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    if not os.path.exists(audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    return audio_path


def do_transcribe(model, audio_path, language) -> dict:
    output = model.transcribe(audio_path, word_timestamps=True, language=language)
    return output


def parse_captions(video_path: str, model_name: str, language: str) -> str:
    model = load_model(os.path.join(settings["default_model_path"], model_name + ".pt"))

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        warnings.warn("CUDA is not available, using CPU. Highly recommend using a GPU for faster inference.")

    audio_path = get_audio(video_path)

    print(f"Transcribing {video_path} with {model_name} model")
    output = do_transcribe(model, audio_path, language)
    print(f"Transcription complete with {len(output['segments'])} segments")

    sentences = {}
    for segment in output['segments']:
        sentences[(round(segment['start'], 2), round(segment['end'], 2))] = segment['text']

    write_captions(video_path, sentences)
    print(f"Captions written to ./output_txt/{video_path.split('/')[-1].replace('.mp4', '.txt')}")

    return sentences


if __name__ == "__main__":
    model_name = "turbo"
    model = load_model(os.path.join(settings["default_model_path"], model_name + ".pt"))
    video_path = "./example.mp4"
    language = "Chinese"
    audio_path = get_audio(video_path)
    text = do_transcribe(model, audio_path, language)
    print(text)