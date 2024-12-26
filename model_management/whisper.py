import os
import requests
from tqdm import tqdm
import warnings
import hashlib
import yaml

whisper_model_address = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

model_settings = yaml.load(open("./conf/model_config.yaml", "r"), Loader=yaml.FullLoader)


def download_model_from_url(model_name: str, url: str, local_download_model_path: str, retry: bool = False) -> None:
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise ValueError(f"Failed to download model {model_name} from {url} \n with status code {response.status_code}")

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    total_size_in_mb = total_size_in_bytes / (1024 * 1024)

    try:
        with open(local_download_model_path, "wb") as handle:
            with tqdm(total=total_size_in_mb, unit='MB', desc="Downloading",
                      unit_scale=False, unit_divisor=1024,
                      mininterval=0.1,
                      miniters=None,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}MB/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                downloaded_size = 0
                for data in response.iter_content(chunk_size=4096):
                    if data:
                        handle.write(data)
                        chunk_size_mb = len(data) / (1024 * 1024)
                        downloaded_size += chunk_size_mb
                        remaining = max(0, min(chunk_size_mb, total_size_in_mb - pbar.n))
                        pbar.update(remaining)
                pbar.close()
    except Exception as e:
        if retry:
            raise ValueError(f"Failed to download model {model_name} from {url} with error {e}")

        warnings.warn(f"Failed to download model {model_name} from {url} with error {e}."
                      f"Retrying download.")
        os.remove(local_download_model_path)
        download_model_from_url(model_name, url, local_download_model_path, retry=True)


def check_sha256(sha256: str, file_path: str):
    return hashlib.sha256(open(file_path, "rb").read()).hexdigest() == sha256


def whisper_download(model_name: str, force_download: bool = False, retry: bool = False) -> None:
    os.makedirs(os.path.join(model_settings["default_model_path"], "whisper"), exist_ok=True)
    local_download_model_path = os.path.join(model_settings["default_model_path"], "whisper", model_name + ".pt")
    url = whisper_model_address[model_name]

    if os.path.exists(local_download_model_path):
        if force_download:
            print(f"Model detected, but force downloading model {model_name} to {local_download_model_path}")
            os.remove(local_download_model_path)
        else:
            if check_sha256(url.split("/")[-2], local_download_model_path):
                print(f"Model {model_name} already exists.")
                return
            else:
                print(f"Model {model_name} already exists, but SHA256 checksum does not match. "
                      f"Re downloading to {local_download_model_path}")
                os.remove(local_download_model_path)
    else:
        print(f"No model named {model_name}. Downloading to {local_download_model_path}")

    download_model_from_url(model_name, url, local_download_model_path)

    if not check_sha256(url.split("/")[-2], local_download_model_path):
        if retry:
            raise ValueError(f"Downloaded model {model_name} SHA256 checksum still does not match the expected value. "
                             f"Exiting.")

        warnings.warn(f"Downloaded model {model_name} SHA256 checksum does not match the expected value. "
                      f"Retrying download.")
        os.remove(local_download_model_path)
        whisper_download(model_name, force_download=True, retry=True)

    print(f"Model {model_name} successfully downloaded to {local_download_model_path}")


if __name__ == "__main__":
    whisper_download("turbo", force_download=True)