"""
Продолжение препроцессинга ПОСЛЕ первого прогона 1_prepare_dataset.py, который
упёрся в 12-часовой лимит Kaggle. Маски (agnostic-mask) для train И test уже
досчитались (видно в логах первого прогона — процесс дошёл до DensePose,
значит маски позади) и DensePose для train успел обработать ~6200 из 11647
фото. Этот скрипт НЕ начинает всё заново — он забирает уже посчитанное из
output первого прогона и продолжает именно DensePose (agnostic-mask вообще
не трогает — они уже готовы).

⚠️ ПЕРЕД ЗАПУСКОМ — обязательно подключите output первого ноутбука как Input:
   1. В новом ноутбуке (или в этом же) нажмите "Add Input"
   2. Вкладка "Notebook Output" (не "Datasets")
   3. Найдите notebookcde6045e8e (1_prepare_dataset.py), добавьте
   Он примонтируется в /kaggle/input/notebookcde6045e8e/_output_.zip —
   именно оттуда этот скрипт возьмёт уже готовые файлы.

СТАТУС (2026-08-25): выполнен успешно. Итог — весь VITON-HD предобработан:
  [train] agnostic-mask: 11647, image-densepose: 11647
  [test]  agnostic-mask: 2032,  image-densepose: 2032
Результат лежит в Output этого ноутбука (notebookd4fbf8b5b7) — именно его
нужно подключать как Input в 3_train_lora.py (PREV_DATASET_NOTEBOOK).
"""

import os
import shutil
import subprocess
import sys
import zipfile

IDM_VTON_REPO = "/kaggle/working/IDM-VTON"
OUTPUT_DIR    = "/kaggle/working/viton-hd-idm"

# путь к output первого прогона — поменяйте имя, если подключили input под другим названием
PREV_OUTPUT_ZIP_CANDIDATES = [
    "/kaggle/input/notebookcde6045e8e/_output_.zip",
]

CATEGORY   = "upper_body"
MASK_WIDTH  = 384
MASK_HEIGHT = 512


def _find_prev_zip():
    for p in PREV_OUTPUT_ZIP_CANDIDATES:
        if os.path.exists(p):
            return p
    # на случай другого имени входа — ищем любой _output_.zip под /kaggle/input
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "_output_.zip":
                return os.path.join(root, f)
    raise RuntimeError(
        "Не нашёл _output_.zip первого прогона в /kaggle/input — "
        "проверьте, что вы подключили Notebook Output через 'Add Input'."
    )


def restore_from_previous_run():
    """Распаковывает результат первого прогона обратно в /kaggle/working."""
    prev_zip = _find_prev_zip()
    print(f"Нашёл: {prev_zip}")
    os.makedirs("/kaggle/working", exist_ok=True)
    with zipfile.ZipFile(prev_zip) as zf:
        zf.extractall("/kaggle/working")
    print("✓ Распаковал предыдущий output в /kaggle/working")

    # быстрая проверка — сколько масок и densepose уже готово
    for split in ("train", "test"):
        for sub in ("agnostic-mask", "image-densepose", "image", "cloth"):
            d = os.path.join(OUTPUT_DIR, split, sub)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            broken = 0
            if os.path.isdir(d):
                broken = sum(1 for f in os.listdir(d) if os.path.islink(os.path.join(d, f)) and not os.path.exists(os.path.join(d, f)))
            print(f"  [{split}] {sub}: {n} файлов" + (f" (⚠️ {broken} битых симлинков)" if broken else ""))


def _repair_image_cloth_symlinks_if_needed():
    """
    image/ и cloth/ в первом прогоне были симлинками на распакованный архив
    VITON-HD (чтобы не копировать лишний раз). Если после zip/unzip они
    превратились в битые ссылки — перекачиваем сам VITON-HD (это самая
    быстрая часть, пара минут) и переcоздаём ссылки.
    """
    broken_found = False
    for split in ("train", "test"):
        for sub in ("image", "cloth"):
            d = os.path.join(OUTPUT_DIR, split, sub)
            if not os.path.isdir(d) or len(os.listdir(d)) == 0:
                broken_found = True

    if not broken_found:
        print("✓ image/ и cloth/ на месте, перекачивать VITON-HD не нужно")
        return

    print("⚠️ image/ или cloth/ пустые/битые — перекачиваю VITON-HD заново (только для восстановления ссылок)")
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    import glob
    import gdown

    zip_path = "/kaggle/working/viton_hd.zip"
    extract_dir = "/kaggle/working/viton_hd_raw"
    if not os.path.exists(zip_path):
        gdown.download(id="1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w", output=zip_path, quiet=False)
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        subprocess.run(["unzip", "-q", zip_path, "-d", extract_dir], check=True)

    train_src = glob.glob(os.path.join(extract_dir, "**", "train"), recursive=True)[0]
    test_src = glob.glob(os.path.join(extract_dir, "**", "test"), recursive=True)[0]
    for split, src in (("train", train_src), ("test", test_src)):
        for sub in ("image", "cloth"):
            src_sub = os.path.join(src, sub)
            dst_sub = os.path.join(OUTPUT_DIR, split, sub)
            if os.path.islink(dst_sub) or os.path.isdir(dst_sub):
                if os.path.islink(dst_sub):
                    os.remove(dst_sub)
                elif len(os.listdir(dst_sub)) == 0:
                    os.rmdir(dst_sub)
            if not os.path.exists(dst_sub):
                os.symlink(src_sub, dst_sub)
    print("✓ Ссылки на image/cloth восстановлены")


def _ensure_repo():
    if not os.path.isdir(IDM_VTON_REPO):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yisol/IDM-VTON.git", IDM_VTON_REPO],
            check=True,
        )
    for p in (IDM_VTON_REPO, os.path.join(IDM_VTON_REPO, "gradio_demo")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _download_checkpoints(files):
    from huggingface_hub import hf_hub_download

    for rel_path in files:
        target = os.path.join(IDM_VTON_REPO, rel_path)
        if os.path.exists(target) and os.path.getsize(target) > 1024:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        hf_hub_download(repo_id="yisol/IDM-VTON", repo_type="space", filename=rel_path, local_dir=IDM_VTON_REPO)
        print(f"✓ скачан {rel_path}")


def install_detectron2():
    subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)
    subprocess.run(
        ["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose"],
        check=True,
    )


def generate_densepose(splits=("train", "test")):
    _ensure_repo()
    install_detectron2()
    _download_checkpoints(["ckpt/densepose/model_final_162be9.pkl"])

    from PIL import Image
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    sys.path.insert(0, os.path.join(IDM_VTON_REPO, "gradio_demo"))
    import apply_net

    args = apply_net.create_argument_parser().parse_args((
        "show",
        os.path.join(IDM_VTON_REPO, "configs/densepose_rcnn_R_50_FPN_s1x.yaml"),
        os.path.join(IDM_VTON_REPO, "ckpt/densepose/model_final_162be9.pkl"),
        "dp_segm", "-v", "--opts", "MODEL.DEVICE", "cuda",
    ))

    for split in splits:
        image_dir = os.path.join(OUTPUT_DIR, split, "image")
        out_dir = os.path.join(OUTPUT_DIR, split, "image-densepose")
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(os.listdir(image_dir))
        already = len(os.listdir(out_dir))
        print(f"[{split}] image-densepose: {len(names)} файлов, уже готово {already}")

        for i, name in enumerate(names):
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path):
                continue
            human_img = Image.open(os.path.join(image_dir, name)).convert("RGB").resize((MASK_WIDTH, MASK_HEIGHT))
            img_arg = convert_PIL_to_numpy(_apply_exif_orientation(human_img), format="BGR")
            pose_img = args.func(args, img_arg)
            pose_img = pose_img[:, :, ::-1]
            Image.fromarray(pose_img).save(out_path)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}")

        print(f"✓ [{split}] image-densepose готовы: {out_dir}")


if __name__ == "__main__":
    restore_from_previous_run()
    _repair_image_cloth_symlinks_if_needed()
    generate_densepose()
    print("Готово. DATA_DIR для 3_train_lora.py:", OUTPUT_DIR)
