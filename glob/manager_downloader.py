import os
from urllib.parse import urlparse
import urllib
import sys
import logging
import requests
from huggingface_hub import HfApi
from tqdm.auto import tqdm
import shutil

aria2 = os.getenv('COMFYUI_MANAGER_ARIA2_SERVER')
HF_ENDPOINT = os.getenv('HF_ENDPOINT')
dir_remote = os.getenv('COMFYUI_MANAGER_DIR_REMOTE')
dir_net = os.getenv('COMFYUI_MANAGER_DIR_NET')
print(f"aria2: {aria2}")
print(f"dir_remote: {dir_remote}")
print(f"dir_net: {dir_net}")


if aria2 is not None:
    secret = os.getenv('COMFYUI_MANAGER_ARIA2_SECRET')
    url = urlparse(aria2)
    port = url.port
    host = url.scheme + '://' + url.hostname
    import aria2p

    aria2 = aria2p.API(aria2p.Client(host=host, port=port, secret=secret))


def basic_download_url(url, dest_folder: str, filename: str):
    '''
    Download file from url to dest_folder with filename
    using requests library.
    '''
    import requests

    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Full path to save the file
    dest_path = os.path.join(dest_folder, filename)

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        raise Exception(f"Failed to download file from {url}")


def download_url(model_url: str, model_dir: str, filename: str):
    if HF_ENDPOINT:
        model_url = model_url.replace('https://huggingface.co', HF_ENDPOINT)
        logging.info(f"model_url replaced by HF_ENDPOINT, new = {model_url}")
    if aria2:
        return aria2_download_url(model_url, dir_remote, dir_net, model_dir, filename)
    else:
        from torchvision.datasets.utils import download_url as torchvision_download_url
        return torchvision_download_url(model_url, model_dir, filename)


def aria2_find_task(dir: str, filename: str):
    target = os.path.join(dir, filename)

    downloads = aria2.get_downloads()

    for download in downloads:
        for file in download.files:
            if file.is_metadata:
                continue
            if str(file.path) == target:
                return download


def aria2_download_url(model_url: str, dir_remote: str, dir_net: str, model_dir: str, filename: str):
    import manager_core as core
    import tqdm
    import time
    import json

    if model_dir.startswith(core.comfy_path):
        model_dir = model_dir[len(core.comfy_path) :]

    download_dir = model_dir if model_dir.startswith('/') else os.path.join('/models', model_dir)
    print(f"download_dir: {download_dir}")
    # 如果download_dir是绝对路径，则删除开头与comfy_base_path相同的部分，忽略大小写
    if download_dir.lower().startswith(core.comfy_path.lower()):
        download_dir_rel = download_dir[len(core.comfy_path):]
    else :
        download_dir_rel = download_dir
    print(f"download_dir_rel: {download_dir_rel}")
    download_dir_remote = os.path.join(dir_remote, download_dir_rel[1:])
    print(f"download_dir_remote: {download_dir_remote}")

    download = aria2_find_task(download_dir_remote, filename)
    if download is None:
        token_path = os.environ['TOKEN_PATH']
        # 读取token.json
        with open(token_path, 'r') as f:
            token = json.load(f)
        if model_url.startswith('https://huggingface.co'):
            headers = ["Authorization: Bearer " + token['huggingface']]
        options = {'dir': download_dir_remote, 'out': filename, 'header': headers}
        download = aria2.add(model_url, options)[0]

    if download.is_active:
        with tqdm.tqdm(
            total=download.total_length,
            bar_format='{l_bar}{bar}{r_bar}',
            desc=filename,
            unit='B',
            unit_scale=True,
        ) as progress_bar:
            while download.is_active:
                if progress_bar.total == 0 and download.total_length != 0:
                    progress_bar.reset(download.total_length)
                progress_bar.update(download.completed_length - progress_bar.n)
                time.sleep(1)
                download.update()

    # 下载完成行为
    if download.is_complete:
        download_dir_net = os.path.join(dir_net, download_dir_rel[1:])
        download_dir_net = download_dir_net.replace("\\", "/")
        print(f"download_dir_net: {download_dir_net}")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        file_net = os.path.join(download_dir_net, filename)
        file_local = os.path.join(download_dir, filename)
        print(f"file_net: {file_net}")
        print(f"file_local: {file_local}")
        shutil.copy2(file_net, file_local)



def download_url_with_agent(url, save_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)
        data = response.read()

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, 'wb') as f:
            f.write(data)

    except Exception as e:
        print(f"Download error: {url} / {e}", file=sys.stderr)
        return False

    print("Installation was successful.")
    return True

# NOTE: snapshot_download doesn't provide file size tqdm.
def download_repo_in_bytes(repo_id, local_dir):
    api = HfApi()
    repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)

    os.makedirs(local_dir, exist_ok=True)

    total_size = 0
    for file_info in repo_info.siblings:
        if file_info.size is not None:
            total_size += file_info.size

    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")

    for file_info in repo_info.siblings:
        out_path = os.path.join(local_dir, file_info.rfilename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if file_info.size is None:
            continue

        download_url = f"https://huggingface.co/{repo_id}/resolve/main/{file_info.rfilename}"

        with requests.get(download_url, stream=True) as r, open(out_path, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    pbar.close()


