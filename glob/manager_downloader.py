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
print(f"aria2: {aria2}")


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
    max_retry = 5
    retry = 0
    if aria2:
        while retry < max_retry:
            try:
                dir_remote = os.getenv('COMFYUI_MANAGER_DIR_REMOTE')
                dir_net = os.getenv('COMFYUI_MANAGER_DIR_NET')
                print(f"dir_remote: {dir_remote}")
                print(f"dir_net: {dir_net}")
                return aria2_download_url(model_url, dir_remote, dir_net, model_dir, filename)
            except Exception as e:
                logging.error(f"Download error: {model_url} / {e}")
                retry += 1
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

def aria2_download_add(model_url: str, download_dir: str, filename: str):
    import json

    token_path = os.environ['TOKEN_PATH']
    # 读取token.json
    with open(token_path, 'r') as f:
        token = json.load(f)
    headers = []
    if model_url.startswith('https://huggingface.co'):
        headers = ["Authorization: Bearer " + token['huggingface']]
    elif model_url.startswith('https://civitai.com'):
        headers = {"Authorization": f"Bearer {token['civitai']}"}
        model_url = requests.head(model_url, headers=headers, allow_redirects=True).url
        headers = []
    options = {'dir': download_dir, 'out': filename, 'header': headers}
    
    return aria2.add(model_url, options)[0]

def aria2_download_update(download, filename: str):
    import tqdm
    import time

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

def aria2_download_complete(dir_remote: str, dir_net :str, model_dir: str, filename: str):
    print(f"{filename} download complete, copy start")
    download_dir_net = get_download_path("download_dir_net", dir_remote, dir_net, model_dir)
    download_dir = get_download_path(path_id="download_dir", model_dir=model_dir)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    file_net = os.path.normpath(os.path.join(download_dir_net, filename))
    file_local = os.path.normpath(os.path.join(download_dir, filename))
    shutil.copy2(file_net, file_local)
    print(f"copy2: {file_net} -> {file_local}")
    return file_local

def aria2_download_url(model_url: str, dir_remote: str, dir_net: str, model_dir: str, filename: str):

    download_dir_remote = get_download_path(path_id="download_dir_remote", dir_remote=dir_remote, model_dir=model_dir)

    download = aria2_find_task(download_dir_remote, filename)

    if download is None or download.has_failed:
        download = aria2_download_add(model_url, download_dir_remote, filename)

    if download.is_active:
        aria2_download_update(download, filename)

    if download.is_complete:
        return aria2_download_complete(dir_remote, dir_net, model_dir, filename)

    if download.has_failed:
        raise Exception(f"Download failed: {model_url}")


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

def get_download_path(path_id: str, dir_remote: str = None, dir_net: str = None, model_dir: str = None):
    """ 
    dir_remote: E:/MCS/ComfyUI
    dir_net: //192.168.0.100/mcs/ComfyUI
    model_dir: D:/ComfyUI/models/model_type/base; /models/model_type/base; /model_type/base
    download_dir: D:/ComfyUI/models/model_type/base; /models/model_type/base
    download_dir_rel: /models/model_type/base
    download_dir_remote: E:/MCS/ComfyUI/models/model_type/base
    download_dir_net: //192.168.0.100/mcs/ComfyUI/models/model_type/base
    """

    import manager_core as core

    if model_dir.lower().startswith(core.comfy_path.lower()):
        download_dir_rel = model_dir[len(core.comfy_path) :]
    download_dir_rel = os.path.normpath(download_dir_rel)

    download_dir_rel = download_dir_rel if download_dir_rel.startswith('/') else os.path.join('/models', download_dir_rel)
    download_dir_rel = os.path.normpath(download_dir_rel)

    if path_id == "download_dir":
        return os.path.normpath(model_dir)
    if path_id == "download_dir_rel":
        return download_dir_rel
    if path_id == "download_dir_remote":
        return os.path.join(dir_remote, download_dir_rel[1:])
    if path_id == "download_dir_net":
        return os.path.normpath(os.path.join(dir_net, download_dir_rel[1:]))