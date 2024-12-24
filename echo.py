print("Importing. Warning messages from side_packages are normal.")

import argparse

from echo.models_management import download_model
from echo.parse import parse_captions
from echo.write import write_captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-video_path", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True)
    parser.add_argument("-force_download", action="store_true")
    parser.add_argument("-mode", type=str, choices=["parse", "write"],
                        default="parse", required=True)
    parser.add_argument("-language", type=str, default="Chinese")

    args = parser.parse_args()
    download_model(args.model_name, args.force_download)

    if args.mode == "parse":
        parse_captions(args.video_path, args.model_name, args.language)
    elif args.mode == "write":
        write_captions(args.video_path)


if __name__ == "__main__":
    main()