import argparse
import os
import yaml
import warnings

from huggingface_hub import snapshot_download, hf_hub_download

model_settings = yaml.load(open("./conf/model_config.yaml", "r"), Loader=yaml.FullLoader)


def huggingface_download_file(repo: str, filename: str, args: argparse.Namespace, retry: bool = False) -> None:
    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            force_download=args.force_download,
            use_auth_token=args.token,
            local_dir=os.path.join(model_settings["default_model_path"], "huggingface", repo)
        )
        print(f"Downloaded {filename} from {repo}")
    except Exception as e:
        if retry:
            raise ValueError(f"Failed to download {filename} with error {e}")
        warnings.warn(f"Failed to download {filename} with error {e}. Retrying download.")
        huggingface_download_file(repo, filename, args, retry=True)


def huggingface_download_repo(repo: str, args: argparse.Namespace, retry: bool = False) -> None:
    try:
        snapshot_download(
            repo_id=repo,
            revision="main",
            use_auth_token=args.token,
            force_download=args.force_download,
            local_dir=os.path.join(model_settings["default_model_path"], "huggingface", repo)
        )
        print(f"Downloaded {repo}")
    except Exception as e:
        if retry:
            raise ValueError(f"Failed to download {repo} with error {e}")
        warnings.warn(f"Failed to download {repo} with error {e}. Retrying download.")
        huggingface_download_repo(repo, args, retry=True)


def huggingface_download(model_name: str, args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(model_settings["default_model_path"], "huggingface"), exist_ok=True)
    os.makedirs(os.path.join(model_settings["default_model_path"], "huggingface", model_name), exist_ok=True)

    if args.filename:
        huggingface_download_file(model_name, args.filename, args)
    else:
        huggingface_download_repo(model_name, args)