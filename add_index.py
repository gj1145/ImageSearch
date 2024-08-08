import os
import base64
import torch
from io import BytesIO
from sentence_transformers import SentenceTransformer
from PIL import Image
from PIL.ImageSequence import all_frames
from opensearchpy import OpenSearch
from tqdm import tqdm

img_path = '/root/autodl-tmp/wxdata/emoji_data' #图片路径
img_model = SentenceTransformer('/models/clip-ViT-B-32')

def get_pic_base64(filename, is_gif=False):
    img = Image.open(filename)  # 访问图片路径
    if is_gif:
        frames = all_frames(img)

        total_frames = len(frames)
        # 计算中间帧的索引
        middle_frame_index = total_frames // 2
        # 读取中间帧
        img.seek(middle_frame_index)
        img = img.convert('RGB')
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
    else:
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str

def get_pic(filename, is_gif=False):
    img = Image.open(filename)  # 访问图片路径
    if is_gif:
        frames = all_frames(img)
        total_frames = len(frames)
        # 计算中间帧的索引
        middle_frame_index = total_frames // 2
        # 读取中间帧
        img.seek(middle_frame_index)
        img = img.convert('RGB')
    return img

#创建client
client = OpenSearch(
    'https://localhost:9200',
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    http_auth=('admin', '114514Aa@')
)
index = 'image-search-sentence-transformers'

#添加索引
cnt = 0
if os.path.exists('./cnt.pth'):
    cnt = torch.load('./cnt.pth')
    
for root, dirs, files in os.walk(img_path, topdown=False):
    for pos, name in enumerate(tqdm(files)):
        if pos >= cnt:
            file_path = os.path.join(root, name)
            #img = get_pic_base64(file_path, file_path.endswith('gif'))
            img = get_pic(file_path, file_path.endswith('gif'))
            img_vector = img_model.encode(img)
            client.index(index=index, body={
                "name": name,
                "img_vector": img_vector,
            })
            cnt += 1
            torch.save(cnt, './cnt.pth')