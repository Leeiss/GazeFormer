import argparse
from os.path import join
import json
import numpy as np
import torch
import pickle
import warnings
from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_predict
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")


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


def test(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

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

    print("[✓] Loading the model...")
    model.load_state_dict(torch.load(args.trained_model, map_location=device)['model'])
    model.eval()

    # === Dataset settings ===
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir

    # === Target from args ===
    target_task = args.target_task
    target_image = args.target_image
    target_condition = args.target_condition
    target_key = f"{target_task}_{target_image}_{target_condition}"

    # === Load fixations ===
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if target_condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')

    print(f"[✓] Loading fixations from {fixation_path}...")
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)

    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition'] == target_condition, human_scanpaths))
    test_task_img_pairs = np.unique([
        traj['task'] + '_' + traj['name'] + '_' + traj['condition']
        for traj in test_target_trajs
    ])

    if target_key not in test_task_img_pairs:
        print(f"[!] Target '{target_key}' not found in test set.")
        return

    print(f"[✓] Found target '{target_key}', running inference...")

    # === Run model ===
    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()
    image_ftrs = torch.load(join(img_ftrs_dir, target_task.replace(' ', '_'), target_image.replace('jpg', 'pth')), map_location=device).unsqueeze(0)
    task_emb = embedding_dict[target_task]

    print(f"[✓] Running model for task: {target_task}, image: {target_image} on {device}...")
    scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)

    for idx, scanpath in enumerate(scanpaths):
        print(f"\n[✓] Predicted Scanpath #{idx + 1}:")
        for y, x, t in scanpath:
            print(f"x: {x:.2f}, y: {y:.2f}, t: {t:.2f}")



def main(args):
    seed_everything(args.seed)
    print("[✓] Starting test process...")
    test(args)  # запускаем тестовую функцию


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test', parents=[get_args_parser_predict()])
    args = parser.parse_args()
    main(args)

# python3 predict_scanpath.py \
#     --trained_model ./checkpoints/gazeformer_cocosearch_TP.pkg \
#     --dataset_dir ./dataset \
#     --img_ftrs_dir ./dataset/image_features \
#     --target_task car \
#     --target_image 000000491881.jpg \
#     --target_condition present \
#     --num_samples 5
