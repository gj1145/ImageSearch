import torch
from PIL import Image
from util import *
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models


# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

class Cn_Clip_Model:
    def __init__(self, model_name, device='cpu', download_root='./'):
        # https://github.com/OFA-Sys/Chinese-CLIP
        """
        初始化
        :param model_name:选择一个'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'
        :param device: 'cpu' or 'cuda'
        :param download_root: 模型下载地址
        """
        self.device = device
        self.model, self.preprocess = load_from_name(model_name, device=device, download_root=download_root)

    def get_img_embedding(self, img):
        """
        img embedding
        :param img: img地址或者PIL.Image.Image对象
        :return: img embedding
        """
        self.model.eval()
        if isinstance(img, str):
            image = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
        if isinstance(img, Image.Image):
            image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def get_text_embedding(self, txt):
        """
        :param txt：可以是list 也可以是一个str
        :return: embedding
        """
        self.model.eval()
        text = clip.tokenize(txt).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features

    def get_similarity(self, img, text):
        """
        图片和text的相似度
        :param img:
        :param text:
        :return:
        """
        self.model.eval()
        if isinstance(img, str):
            image = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
        if isinstance(img, Image.Image):
            image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model.get_similarity(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
            return probs


if __name__ == '__main__':
    print("Available models:", available_models())
    # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    model = Cn_Clip_Model('RN50', "cuda" if torch.cuda.is_available() else "cpu", './')
    # 如果是gif可以传入PIL的Image对象 或者是 地址
    # embedding = model.get_img_embedding('../vx_emoji/datasets/img/1.gif')
    embedding = model.get_img_embedding('../vx_emoji/datasets/img/2.png')
    print(embedding.shape)
    print(embedding)
    text_embedding = model.get_text_embedding(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"])#可以是列表也可以是str
    print(len(text_embedding))
    print(text_embedding[0].shape)
    txt_embedding_2 = model.get_text_embedding("杰尼龟")
    print(txt_embedding_2.shape)
    similarity = model.get_similarity('../vx_emoji/datasets/img/2.png', ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"])
    print(similarity)
