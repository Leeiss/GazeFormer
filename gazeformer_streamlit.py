import argparse
import torch
import numpy as np
import os
from os.path import join, isdir, isfile
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import torchvision.transforms as T
from models import Transformer, ResNetCOCO
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_streamlit
from tqdm import tqdm
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


task_map = {
    'bottle': 'Бутылка',
    'bowl': 'Тарелка',
    'car': 'Автомобиль',
    'chair': 'Стул',
    'clock': 'Часы',
    'cup': 'Чашка',
    'fork': 'Вилка',
    'keyboard': 'Kлавиатура',
    'knife': 'Нож',
    'laptop': 'Ноутбук',
    'microwave': 'Микроволновка',
    'mouse': 'Мышь',
    'oven': 'Духовка',
    'potted plant': 'Растение в горшке',
    'sink': 'Раковина',
    'stop sign': 'Стоп знак',
    'toilet': 'Туалет',
    'tv': 'Телевизор'
}

def resize_with_padding(img, target_size=(512, 320), fill_color=(0, 0, 0)):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)

    img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


def extract_features_from_image(image_path, device, resize_dim=(640, 1024)):
    model = ResNetCOCO(device=device).to(device)
    model.eval()

    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(resize_dim),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    return features.squeeze(0)


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


def process_fixations(fixations_data, img, original_width, original_height):
    height, width, _ = img.shape
    scale_x = width / original_width
    scale_y = height / original_height

    heatmap_simple = np.zeros((height, width))
    heatmap_weighted = np.zeros((height, width))

    for fixation in fixations_data:
        X = fixation["X"]
        Y = fixation["Y"]
        T = fixation["T"]
        print(f"X - {X}")
        print(f"Y - {Y}")
        print(f"T - {T}")

        for x, y, t in zip(X, Y, T):
            x_scaled = int(round(x * scale_x))
            y_scaled = int(round(y * scale_y))
            if 0 <= x_scaled < width and 0 <= y_scaled < height:
                heatmap_simple[y_scaled, x_scaled] += 1
                heatmap_weighted[y_scaled, x_scaled] += t / 100.0

    heatmap_simple = gaussian_filter(heatmap_simple, sigma=10)
    heatmap_weighted = gaussian_filter(heatmap_weighted, sigma=10)

    def normalize(hmap):
        hmap = np.clip(hmap, 0, np.max(hmap))
        hmap = (hmap / np.max(hmap)) * 255
        return np.uint8(hmap)

    heatmap_simple = normalize(heatmap_simple)
    heatmap_weighted = normalize(heatmap_weighted)

    return heatmap_simple, heatmap_weighted


def normalize_time(T):
    cmap = plt.cm.get_cmap('Blues')
    norm = Normalize(vmin=min(T), vmax=max(T))
    return [cmap(norm(t)) for t in T]


def save_gaze_trajectory_with_time_gradient(base_img, fixation_list, scale_x, scale_y, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(base_img)

    for fixation in fixation_list:
        X = [x * scale_x for x in fixation['X']]
        Y = [y * scale_y for y in fixation['Y']]
        T = fixation['T']
        colors = normalize_time(T)

        for i in range(1, len(X)):
            plt.plot([X[i-1], X[i]], [Y[i-1], Y[i]], color=colors[i-1], lw=2)

        for i, (x, y) in enumerate(zip(X, Y)):
            plt.scatter(x, y, color=colors[i], s=100, zorder=10)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_heatmap(base_img, heatmap, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(base_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def test_single_case(args, image_path, selected_task_name):
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

    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle=True).item()



    if image_path.endswith('.jpg') or image_path.endswith('.png'):
        print(f"Extracting features from {image_path}...")
        image_ftrs = extract_features_from_image(image_path, device=device)
    elif image_path.endswith('.pth'):
        print(f"Loading features from {image_path}...")
        image_ftrs = torch.load(image_path).unsqueeze(0)
    else:
        raise ValueError("Unsupported file type. Only .jpg, .png, and .pth are supported.")

    task_emb = embedding_dict[selected_task_name]

    print(f"Running inference for {selected_task_name} | {os.path.basename(image_path)} ")
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

    formatted_scanpaths = [{"X": scanpath[:, 1], "Y": scanpath[:, 0], "T": scanpath[:, 2]} for scanpath in scanpaths]

    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    height, width, _ = img.shape

    original_width = 512
    original_height = 320

    heatmap_simple, heatmap_weighted = process_fixations(formatted_scanpaths, img, original_width, original_height)

    save_heatmap(img, heatmap_simple, 'heatmap_simple.png')
    save_heatmap(img, heatmap_weighted, 'heatmap_weighted.png')
    save_gaze_trajectory_with_time_gradient(img, formatted_scanpaths, 1, 1, 'gaze_trajectory.png')

    return scanpaths


def main(args):
    st.title("GazeFormer: Предсказание взгляда")

    uploaded_file = st.file_uploader("Загрузите изображение (.jpg/.png)", type=["jpg", "png"])

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        image = resize_with_padding(original_image, target_size=(512, 320))

        st.image(image, caption="Исходное изображение", use_container_width=True)

        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        st.sidebar.subheader("Опции отображения")

        show_heatmap_simple = st.sidebar.checkbox("Тепловая карта (простая)", value=False)
        show_heatmap_weighted = st.sidebar.checkbox("Тепловая карта (по времени)", value=True)
        show_trajectory = st.sidebar.checkbox("Траектория движения глаз", value=False)
        show_coords = st.sidebar.checkbox("Показать предсказанные координаты", value=False)
        show_all = st.sidebar.checkbox("Показать всё", value=False)

        task_list = list(task_map.values())
        selected_task = st.sidebar.selectbox("Что находится на изображении?", task_list)

        selected_task_name = [task for task, ru in task_map.items() if ru == selected_task][0]

        seed_everything(args.seed)
        scanpaths = test_single_case(args, temp_image_path, selected_task_name)

        st.subheader("Результаты моделирования взгляда:")

        if show_all or show_heatmap_simple:
            st.image("heatmap_simple.png", caption="Простая тепловая карта")

        if show_all or show_heatmap_weighted:
            st.image("heatmap_weighted.png", caption="Взвешенная тепловая карта (по времени)")

        if show_all or show_coords:
            st.subheader("Таблица предсказанных данных")
            for i, path in enumerate(scanpaths):
                df = path.copy()
                df_swapped = np.stack([df[:, 1], df[:, 0], df[:, 2]], axis=1)  # x, y, t
                df_display = pd.DataFrame(df_swapped, columns=["x", "y", "t"])
                st.markdown(df_display.to_html(index=False), unsafe_allow_html=True)

        if show_all or show_trajectory:
            st.image("gaze_trajectory.png", caption="Траектория взгляда с градиентом времени")


if __name__ == '__main__':
    parser = get_args_parser_streamlit()
    args = parser.parse_args()
    main(args)
