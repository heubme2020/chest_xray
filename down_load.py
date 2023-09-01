import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def download_files_from_url(url, base_folder = ''):
    folder = url.split('/')[-2]
    folder = base_folder + folder
    os.makedirs(folder, exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        for link in links:
            file_url = link['href']
            if not file_url.startswith('http'):
                file_url = urljoin(url, file_url)
            if file_url.endswith('index.html'):
                base = folder + '/'
                download_files_from_url(file_url, base)
            else:
                # 如果是文件，下载到指定路径
                file_name = file_url.split('/')[-1]
                file_path = os.path.join(folder, file_name)
                if os.path.exists(file_path):
                    print(f'已下载: {file_path}')
                    continue
                else:
                    download_file(file_url, file_path)
    else:
        print('无法获取网页内容。')


def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f'下载成功: {local_path}')
    else:
        print(f'无法下载文件: {url}')


def download_chexPert_data():
    # Chexpert数据集下载链接
    chexpert_url = https://aimistanforddatasets01.blob.core.windows.net/chexpertchestxrays-u20210408/CheXpert-v1.0.zip?sv=2019-02-02&sr=b&sig=50lvz7El%2FMXTGmjZZHw3B5Ttx7ElH2bT50PNyiBhlU8%3D&st=2023-08-06T05%3A25%3A24Z&se=2023-09-05T05%3A30%3A24Z&sp=r"

    # 本地目标文件路径
    local_file_path = "D:/Chexpert"

    # 下载Chexpert数据集
    response = requests.get(chexpert_url, stream=True)
    print(response)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("下载完成！")
    else:
        print("无法下载数据集。请检查链接是否有效。")
if __name__ == '__main__':
    # 替换为您要下载的网址和本地保存路径
    download_chexPert_data()