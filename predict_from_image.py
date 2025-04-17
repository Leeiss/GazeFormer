import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from os.path import join
import warnings
import os

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_predict
from tqdm import tqdm

warnings.filterwarnings("ignore")


def preprocess_image(image_path, image_size=(512, 320)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # стандартные значения для ImageNet
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # добавим batch dim


def extract_features_resnet(img_tensor, device):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-2])  # убираем последние слои (FC и pooling)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = model(img_tensor.to(device))
    return features.squeeze(0)  # [2048, H, W]


def run_model(model, src, task, device="cuda:0", im_h=20, im_w=32, patch_size=16, num_samples=1):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h // 2) * patch_size, (im_w // 2) * patch_size]).unsqueeze(0).repeat(num_samples, 1)
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src=src, tgt=firstfix, task=task)
    token_prob = token_prob.detach().cpu().numpy()
    ys = ys.cpu().detach().numpy()
    xs = xs.cpu().detach().numpy()
    ts = ts.cpu().detach().numpy()
    scanpaths = []
    for i in range(num_samples):
        ys_i = [(im_h // 2) * patch_size] + list(ys[:, i, 0])[1:]
        xs_i = [(im_w // 2) * patch_size] + list(xs[:, i, 0])[1:]
        ts_i = list(ts[:, i, 0])
        token_type = [0] + list(np.argmax(token_prob[:, i, :], axis=-1))[1:]
        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(im_h * patch_size - 2, y), min(im_w * patch_size - 2, x), t])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths


def test_single_image(args, image_path):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Загружаем и обрабатываем изображение
    print(f"📷 Preprocessing image: {image_path}")
    img_tensor = preprocess_image(image_path, image_size=(512, 320))
    image_ftrs = extract_features_resnet(img_tensor, device).view(2048, -1).permute(1, 0)  # -> [H*W, C]

    # Загружаем эмбеддинг
    embedding_dict = np.load(open(join(args.dataset_dir, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()
    task_emb = embedding_dict[args.target_task]

    # Инициализируем модель
    transformer = Transformer(
        num_encoder_layers=args.num_encoder,
        nhead=args.nhead,
        d_model=args.hidden_dim,
        num_decoder_layers=args.num_decoder,
        dim_feedforward=args.hidden_dim,
        img_hidden_dim=args.img_hidden_dim,
        lm_dmodel=args.lm_hidden_dim,
        device=device
    ).to(device)

    model = gazeformer(
        transformer=transformer,
        spatial_dim=(args.im_h, args.im_w),
        max_len=args.max_len,
        device=device
    ).to(device)

    print("📦 Loading GazeFormer model...")
    model.load_state_dict(torch.load(args.trained_model, map_location=device)['model'])
    model.eval()

    print(f"🧪 Running inference for {args.target_task} | {args.target_condition}")
    scanpaths = run_model(
        model=model,
        src=image_ftrs.unsqueeze(0),
        task=task_emb,
        device=device,
        num_samples=args.num_samples,
        im_h=args.im_h,
        im_w=args.im_w,
        patch_size=args.patch_size
    )

    for i, scanpath in enumerate(scanpaths):
        print(f"\n Scanpath {i + 1}:")
        for fix in scanpath:
            print(f"x: {fix[1]}, y: {fix[0]}, t: {fix[2]}")

    return scanpaths


def main(args):
    seed_everything(args.seed)

    # путь к своему изображению
    image_path = args.input_image  # задаётся через аргумент
    test_single_image(args, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test (Image File)', parents=[get_args_parser_predict()])
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')

    args = parser.parse_args()

   

    main(args)
