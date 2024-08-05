import os
import copy
import time
import requests
from urllib.parse import unquote

import main_poster
from main import *
from main_poster import *

def download(url, save_folder, info=True):
    if info: print("Downloading file:", url)
    r = requests.get(url)
    file_name = get_download_file_name(url, r.headers)
    file_path = os.path.join(save_folder, file_name)
    file = open(file_path, "wb")
    file.write(r.content)
    file.close()
    if info: print("Finished download:", str(file_path))
    return file_path

def get_download_file_name(url, headers):
    filename = ''
    if 'Content-Disposition' in headers and headers['Content-Disposition']:
        disposition_split = headers['Content-Disposition'].split(';')
        if len(disposition_split) > 1:
            if disposition_split[1].strip().lower().startswith('filename='):
                file_name = disposition_split[1].split('=')
                if len(file_name) > 1:
                    filename = unquote(file_name[1]).replace("\"", "").replace("\'", "")
    if not filename and os.path.basename(url):
        filename = os.path.basename(url).split("?")[0]
    if not filename:
        return time.time()
    return filename

def post(url, data, info=True):
        # 开新线程进行post
        def post_func(url, data, info=True):
            r = requests.post(url, data)
            if info:
                print(r)
                print(r.text)
        post_thread = threading.Thread(target=post_func, args=(url, data, info))
        post_thread.start()

def post_file(url, file_path, data=None, info=True):
    # 开新线程进行post
    def post_func(url, data, files, info=True):
        print("Uploading:", str(files["file"]))
        r = requests.post(url, data=data, files=files)
        print("Finished upload")
        if info:
            print(r)
            print(r.text)

    files = {"file": open(file_path, "rb")}

    post_thread = threading.Thread(target=post_func, args=(url, data, files, info))
    post_thread.start()
