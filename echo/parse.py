import os
import yaml
import warnings

import torch
from whisper import load_model
from moviepy.video.io.VideoFileClip import VideoFileClip

settings = yaml.load(open("./conf/model_config.yaml", "r"), Loader=yaml.FullLoader)


def write_captions(video_path: str, sentenses: dict) -> None:
    os.makedirs("./output_txt", exist_ok=True)

    with open(os.path.join("./output_txt", video_path.split("/")[-1].replace(".mp4", ".txt")), "w") as f:
        for k, v in sentenses.items():
            f.write(f"{k[0]}-{k[1]}: {v}\n")


def get_audio(video_path: str) -> str:
    _audio_path = video_path.replace(".mp4", ".wav")
    if not os.path.exists(_audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(_audio_path)
    return _audio_path


def do_whisper_transcribe(model, audio_path, language) -> dict:
    return model.transcribe(audio_path, word_timestamps=True, language=language)


def parse_captions_with_whisper(video_path: str, model_info: dict, language: str) -> dict:
    model = load_model(os.path.join(settings["default_model_path"], model_name + ".pt"))

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        warnings.warn("CUDA is not available, using CPU. Highly recommend using a GPU for faster inference.")

    audio_path = get_audio(video_path)

    print(f"Transcribing {video_path} with {model_name} model")
    output = do_whisper_transcribe(model, audio_path, language)
    print(f"Transcription complete with {len(output['segments'])} segments")

    sentences = {}
    for segment in output['segments']:
        sentences[(round(segment['start'], 2), round(segment['end'], 2))] = segment['text']

    write_captions(video_path, sentences)
    print(f"Captions written to ./output_txt/{video_path.split('/')[-1].replace('.mp4', '.txt')}")

    return sentences


def parse_captions_with_huggingface(video_path: str, model_info: dict, language: str) -> dict:
    assert model_info["filename"] is not None, "Filename is required for huggingface model"


def parse_captions(video_path: str, model_info: dict, language: str) -> dict:
    assert video_path is not None, "Video path is required"

    if model_info["model_type"] == "whisper":
        return parse_captions_with_whisper(video_path, model_info, language)
    elif model_info["model_type"] == "huggingface":
        return parse_captions_with_huggingface(video_path, model_info, language)


if __name__ == "__main__":
    model_name = "turbo"
    model = load_model(os.path.join(settings["default_model_path"], model_name + ".pt"))
    video_path = "./example.mp4"
    language = "Chinese"
    audio_path = get_audio(video_path)
    text = do_whisper_transcribe(model, audio_path, language)
    print(text)