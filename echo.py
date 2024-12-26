print("Importing. Warning messages from side_packages are normal.")

import argparse
import os
import yaml

from model_management.whisper import whisper_download, whisper_model_address
from model_management.huggingface import huggingface_download
from echo.parse import parse_captions
from echo.write import write_captions

model_settings = yaml.load(open("./conf/model_config.yaml", "r"), Loader=yaml.FullLoader)


def attempt_download_model(model_name: str, force_download: bool = False, args: argparse.Namespace = None) -> dict:
    assert model_name is not None, "Model name is required."
    os.makedirs(model_settings["default_model_path"], exist_ok=True)

    if model_name in whisper_model_address:
        whisper_download(model_name, force_download)
        model_info = {
            "model_name": model_name,
            "model_path": os.path.join(model_settings["default_model_path"], "whisper", model_name + ".pt"),
            "model_type": "whisper"
        }
    else:
        huggingface_download(model_name, args)

        model_info = {
            "model_name": model_name,
            "model_path": os.path.join(model_settings["default_model_path"], model_name),
            "filename": args.filename,
            "model_type": "huggingface"
        }

    return model_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, choices=["parse", "write", "download"],
                        default="parse", required=True)
    parser.add_argument("-video_path", type=str)
    parser.add_argument("-model_name", type=str)
    parser.add_argument("-filename", type=str)
    parser.add_argument("-token", type=str)
    parser.add_argument("-force_download", action="store_true")
    parser.add_argument("-language", type=str, default="Chinese")

    args = parser.parse_args()
    model_info = attempt_download_model(args.model_name, args.force_download, args)

    if args.mode == "parse":
        parse_captions(args.video_path, model_info, args.language)
    elif args.mode == "write":
        write_captions(args.video_path)


if __name__ == "__main__":
    main()