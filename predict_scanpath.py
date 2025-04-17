import argparse
import torch
import numpy as np
from os.path import join

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_predict
from tqdm import tqdm
import warnings

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


def test_single_case(args):
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

    print("Loading model...")
    model.load_state_dict(torch.load(args.trained_model, map_location=device)['model'])
    model.eval()

    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir

    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()
    task_name = args.target_task
    image_name = args.target_image
    condition = args.target_condition

    image_ftrs_path = join(img_ftrs_dir, task_name.replace(' ', '_'), image_name.replace('jpg', 'pth'))
    image_ftrs = torch.load(image_ftrs_path).unsqueeze(0)
    task_emb = embedding_dict[task_name]

    print(f"🧪 Running inference for {task_name} | {image_name} | {condition}")
    scanpaths = run_model(
        model=model,
        src=image_ftrs,
        task=task_emb,
        device=device,
        num_samples=args.num_samples,
        im_h=args.im_h,
        im_w=args.im_w,
        patch_size=args.patch_size
    )

    for i, scanpath in enumerate(scanpaths):
        print(f"\n Scanpath {i+1}:")
        for fix in scanpath:
            print(f"x: {fix[1]}, y: {fix[0]}, t: {fix[2]}")

    return scanpaths


def main(args):
    seed_everything(args.seed)
    test_single_case(args)


if __name__ == '__main__':
    parser = get_args_parser_predict()

    args = parser.parse_args()

    main(args)
