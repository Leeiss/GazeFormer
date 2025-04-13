import argparse
import torch
import numpy as np
from models import Transformer
from gazeformer import gazeformer
from tqdm import tqdm
import os
import json

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


def generate_scanpaths(args):
    device = torch.device(f'cuda:{args.cuda}')

    # Создаем трансформер и модель
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

    # Загружаем веса модели
    model.load_state_dict(torch.load(args.trained_model, map_location=device)['model'])
    model.eval()

    # Загружаем эмбеддинги задач
    embedding_dict = np.load(open(join(args.dataset_dir, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()

    # Загружаем признаки изображения
    image_ftrs = torch.load(join(args.img_ftrs_dir, args.task_name.replace(' ', '_'), args.image_name.replace('jpg', 'pth'))).unsqueeze(0)
    task_emb = embedding_dict[args.task_name]

    # Генерируем сканпафы
    scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)

    # Сохраняем или выводим результаты
    for idx, scanpath in enumerate(scanpaths):
        print(f"Scanpath {idx + 1} for {args.task_name} and {args.image_name}:")
        print(scanpath)

    # Можно также сохранить сканпафы в файл, если нужно
    if args.save_scanpaths:
        output_filename = f"scanpaths_{args.image_name}_{args.task_name}.npy"
        np.save(output_filename, scanpaths)
        print(f"Scanpaths saved to {output_filename}")

def main(args):
    generate_scanpaths(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Scanpaths', description="Generate gaze scanpaths for a given image and task")
    parser.add_argument('--trained_model', required=True, help="Path to the trained model (.pth file)")
    parser.add_argument('--dataset_dir', required=True, help="Path to the dataset directory")
    parser.add_argument('--img_ftrs_dir', required=True, help="Directory containing image feature files")
    parser.add_argument('--task_name', required=True, help="Name of the task")
    parser.add_argument('--image_name', required=True, help="Name of the image (without extension)")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of scanpaths to generate")
    parser.add_argument('--cuda', type=int, default=-1, help="CUDA device number (-1 for CPU)")
    parser.add_argument('--save_scanpaths', type=bool, default=False, help="Whether to save generated scanpaths to a file")
    parser.add_argument('--im_h', type=int, default=20, help="Height of the image")
    parser.add_argument('--im_w', type=int, default=32, help="Width of the image")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patches")
    parser.add_argument('--max_len', type=int, default=100, help="Maximum length of the sequence")
    parser.add_argument('--num_encoder', type=int, default=6, help="Number of encoder layers")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Dimension of the hidden layers")
    parser.add_argument('--num_decoder', type=int, default=6, help="Number of decoder layers")
    parser.add_argument('--img_hidden_dim', type=int, default=512, help="Hidden dimension for image processing")
    parser.add_argument('--lm_hidden_dim', type=int, default=512, help="Hidden dimension for language model")

    args = parser.parse_args()
    main(args)

