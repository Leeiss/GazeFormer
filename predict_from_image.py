import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from os.path import join
import warnings
import os

from models import Transformer, ResNetCOCO
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_predict
from tqdm import tqdm

warnings.filterwarnings("ignore")

def resize_and_pad(img, target_size=(512, 320), pad_color=(0, 0, 0)):
    original_w, original_h = img.size
    target_w, target_h = target_size

    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", target_size, pad_color)
    upper_left_x = (target_w - new_w) // 2
    upper_left_y = (target_h - new_h) // 2
    new_img.paste(resized_img, (upper_left_x, upper_left_y))

    return new_img


def preprocess_image(image_path, image_size=(512, 320)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')  # Открытие изображения
    img = resize_and_pad(img, image_size)  # Изменение размера и паддинг
    return transform(img).unsqueeze(0)


def extract_features_resnetcoco(img_tensor, device):
    model = ResNetCOCO(device=device).to(device)
    model.eval()
    with torch.no_grad():
        features = model(img_tensor.to(device)).squeeze().detach().cpu()
    return features


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

    print(f"📷 Preprocessing image: {image_path}")

    # Загружаем изображение
    fixed_img = preprocess_image(image_path, image_size=(512, 320))  # Предобработка изображения

    img_tensor = preprocess_image(image_path, image_size=(512, 320))  # Предобработка изображения
    image_ftrs = extract_features_resnetcoco(img_tensor, device)

    # Загружаем эмбеддинг
    embedding_dict = np.load(open(join(args.dataset_dir, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()
    task_emb = embedding_dict[args.target_task]

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

    # === ВИЗУАЛИЗАЦИЯ ===
    fixation_list = []
    for path in scanpaths:
        xs = [float(x[1]) for x in path]
        ys = [float(x[0]) for x in path]
        ts = [float(x[2]) for x in path]
        fixation_list.append({"X": xs, "Y": ys, "T": ts})

    fixed_img = resize_and_pad(Image.open(image_path).convert("RGB"), target_size=(512, 320))
    fixed_img_np = np.array(fixed_img)
    height, width, _ = fixed_img_np.shape

    # Построение тепловых карт
    def build_heatmaps(fixations_data, width, height):
        from scipy.ndimage import gaussian_filter
        heatmap_simple = np.zeros((height, width))
        heatmap_weighted = np.zeros((height, width))

        for fixation in fixations_data:
            for x, y, t in zip(fixation["X"], fixation["Y"], fixation["T"]):
                x_scaled = int(round(x))
                y_scaled = int(round(y))
                if 0 <= x_scaled < width and 0 <= y_scaled < height:
                    heatmap_simple[y_scaled, x_scaled] += 1
                    heatmap_weighted[y_scaled, x_scaled] += t / 100.0

        def normalize(hmap):
            hmap = np.clip(hmap, 0, np.max(hmap))
            if np.max(hmap) > 0:
                hmap = (hmap / np.max(hmap)) * 255
            return np.uint8(hmap)

        heatmap_simple = normalize(gaussian_filter(heatmap_simple, sigma=10))
        heatmap_weighted = normalize(gaussian_filter(heatmap_weighted, sigma=10))
        return heatmap_simple, heatmap_weighted

    def save_heatmap(base_img, heatmap, filename):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(base_img)
        plt.imshow(heatmap, cmap='jet', alpha=0.6)
        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

    def normalize_time(T):
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('Blues')
        norm = Normalize(vmin=min(T), vmax=max(T))
        return [cmap(norm(t)) for t in T]

    def save_gaze_trajectory_with_time_gradient(base_img, fixation_list, filename):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(base_img)

        for fixation in fixation_list:
            T = fixation['T']
            X = fixation['X']
            Y = fixation['Y']
            colors = normalize_time(T)

            for i in range(1, len(X)):
                plt.plot([X[i-1], X[i]], [Y[i-1], Y[i]], color=colors[i-1], lw=2)

            for i, (x, y) in enumerate(zip(X, Y)):
                plt.scatter(x, y, color=colors[i], s=100, zorder=10)

        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

    heatmap_simple, heatmap_weighted = build_heatmaps(fixation_list, width, height)
    save_heatmap(fixed_img_np, heatmap_simple, 'heatmap_simple.png')
    save_heatmap(fixed_img_np, heatmap_weighted, 'heatmap_weighted.png')
    save_gaze_trajectory_with_time_gradient(fixed_img_np, fixation_list, 'gaze_trajectory.png')

    print("✅ Визуализации сохранены: heatmap_simple.png, heatmap_weighted.png, gaze_trajectory.png")


def main(args):
    seed_everything(args.seed)
    image_path = args.input_image
    test_single_image(args, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test (Image File)', parents=[get_args_parser_predict()])
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    args.im_h = 10
    args.im_w = 16
    args.patch_size = 16

    args.target_task = 'car'
    args.target_condition = 'present'
    args.target_image = os.path.basename(args.input_image)

    main(args)
