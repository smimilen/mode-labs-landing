"""
Препроцессинг датасета под IDM-VTON: скачивание VITON-HD, генерация
agnostic-mask и image-densepose, сборка структуры, которую ждёт
3_train_lora.py.

Собрано по коду самого репозитория github.com/yisol/IDM-VTON (клонирован и
прочитан построчно), в первую очередь по gradio_demo/app.py — там ровно та
же последовательность вызовов (Parsing → OpenPose → get_mask_location →
apply_net DensePose), просто на один загруженный кадр за раз. Здесь то же
самое, но батчем по всему датасету.

СТРУКТУРА СКРИПТА — 3 независимых шага, запускайте по очереди:
  download_and_reorganize()  — скачивает VITON-HD (прямая ссылка, без анкеты)
                                и раскладывает по структуре train/test
  generate_agnostic_masks()  — human parsing + openpose → agnostic-mask/
                                (ONNX + лёгкая CNN, идёт на CPU, риск низкий)
  generate_densepose()       — detectron2 + DensePose → image-densepose/
                                (САМАЯ ЛОМКАЯ часть, см. предупреждение внутри)

О ЛИЦЕНЗИИ: VITON-HD распространяется по CC BY-NC 4.0 (только некоммерческое
использование). Для коммерческой версии модели единственный чистый путь —
дообучение на собственных данных платформы.

СТАТУС (2026-08-25): на Kaggle этот скрипт упёрся в 12-часовой лимит
(agnostic-mask досчитались полностью, densepose — частично). Продолжение —
см. 2_resume_densepose.py, который забирает Output этого ноутбука через
Kaggle "Add Input" и продолжает именно с того места, где оборвался лимит.

DressCode в этот скрипт не включён: форма доступа не принимает обычные
gmail-адреса и не выдаётся частным компаниям.
"""

import glob
import json
import os
import subprocess
import sys

# ══════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════
IDM_VTON_REPO = "/kaggle/working/IDM-VTON"          # клон репозитория (нужен для Parsing/OpenPose/apply_net кода)
RAW_ZIP_PATH  = "/kaggle/working/viton_hd.zip"       # куда скачается архив
RAW_EXTRACT_DIR = "/kaggle/working/viton_hd_raw"     # куда распакуется архив
OUTPUT_DIR    = "/kaggle/working/viton-hd-idm"       # итоговый датасет — путь для DATA_DIR в 3_train_lora.py

VITON_HD_GDRIVE_ID = "1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w"  # официальная прямая ссылка с github.com/shadow2496/VITON-HD

CATEGORY      = "upper_body"   # VITON-HD — только верхняя одежда
MASK_WIDTH    = 384             # см. get_mask_location(...) в utils_mask.py — дефолт именно такой
MASK_HEIGHT   = 512             # должно совпадать с WIDTH/HEIGHT в 3_train_lora.py, если их меняете — меняйте и тут


# ══════════════════════════════════════════
# ШАГ 0 — Установка и подготовка репозитория
# ══════════════════════════════════════════
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
    """Тянет конкретные файлы из HF Space yisol/IDM-VTON (там же, где веса модели)."""
    from huggingface_hub import hf_hub_download

    for rel_path in files:
        target = os.path.join(IDM_VTON_REPO, rel_path)
        if os.path.exists(target) and os.path.getsize(target) > 1024:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        hf_hub_download(
            repo_id="yisol/IDM-VTON",
            repo_type="space",
            filename=rel_path,
            local_dir=IDM_VTON_REPO,
        )
        print(f"✓ скачан {rel_path}")


# ══════════════════════════════════════════
# ШАГ 1 — Скачивание VITON-HD и раскладка по train/test
# ══════════════════════════════════════════
def download_and_reorganize():
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    _ensure_repo()

    if not os.path.exists(RAW_ZIP_PATH):
        import gdown

        gdown.download(id=VITON_HD_GDRIVE_ID, output=RAW_ZIP_PATH, quiet=False)

    if not os.path.isdir(RAW_EXTRACT_DIR):
        os.makedirs(RAW_EXTRACT_DIR, exist_ok=True)
        subprocess.run(["unzip", "-q", RAW_ZIP_PATH, "-d", RAW_EXTRACT_DIR], check=True)

    # структура архива внутри не задокументирована авторами — ищем train/ и
    # test/ где бы они ни оказались, вместо того чтобы гадать точный путь
    train_dirs = glob.glob(os.path.join(RAW_EXTRACT_DIR, "**", "train"), recursive=True)
    test_dirs = glob.glob(os.path.join(RAW_EXTRACT_DIR, "**", "test"), recursive=True)
    if not train_dirs or not test_dirs:
        raise RuntimeError(
            f"Не нашёл train/ или test/ внутри {RAW_EXTRACT_DIR} — "
            f"посмотрите руками (`!find {RAW_EXTRACT_DIR} -maxdepth 3`) и поправьте пути ниже."
        )
    train_src, test_src = train_dirs[0], test_dirs[0]
    print(f"Нашёл: train={train_src}, test={test_src}")

    for split, src in (("train", train_src), ("test", test_src)):
        dst = os.path.join(OUTPUT_DIR, split)
        os.makedirs(dst, exist_ok=True)
        for sub in ("image", "cloth"):
            src_sub, dst_sub = os.path.join(src, sub), os.path.join(dst, sub)
            if os.path.isdir(src_sub) and not os.path.islink(dst_sub) and not os.path.isdir(dst_sub):
                os.symlink(src_sub, dst_sub)  # симлинк вместо копии — экономит место и время на Kaggle
        # vitonhd_{split}_tagged.json нужно положить рядом (см. README IDM-VTON) —
        # если тегов нет в архиве, get_dataset ниже подставит дефолтную подпись "shirts"
        tagged_json_candidates = glob.glob(os.path.join(RAW_EXTRACT_DIR, "**", f"vitonhd_{split}_tagged.json"), recursive=True)
        if tagged_json_candidates:
            import shutil

            shutil.copy(tagged_json_candidates[0], os.path.join(dst, f"vitonhd_{split}_tagged.json"))
        else:
            with open(os.path.join(dst, f"vitonhd_{split}_tagged.json"), "w") as f:
                json.dump({}, f)
            print(f"⚠️ vitonhd_{split}_tagged.json не найден в архиве — подписи будут дефолтные ('shirts')")

    # train_pairs.txt / test_pairs.txt — если их нет в архиве, строим по
    # содержимому image/ (в VITON-HD одежда и человек на фото — одна и та же
    # пара по имени файла, режим "paired", как в оригинальном train_xl.py)
    for split, src in (("train", train_src), ("test", test_src)):
        pairs_candidates = glob.glob(os.path.join(RAW_EXTRACT_DIR, "**", f"{split}_pairs.txt"), recursive=True)
        dst_pairs = os.path.join(OUTPUT_DIR, f"{split}_pairs.txt")
        if pairs_candidates:
            import shutil

            shutil.copy(pairs_candidates[0], dst_pairs)
        else:
            image_dir = os.path.join(src, "image")
            names = sorted(os.listdir(image_dir))
            with open(dst_pairs, "w") as f:
                for n in names:
                    f.write(f"{n} {n}\n")
            print(f"⚠️ {split}_pairs.txt не найден в архиве — сгенерирован по содержимому image/")

    print(f"✓ Датасет разложен в {OUTPUT_DIR} — это и есть DATA_DIR для 3_train_lora.py")


# ══════════════════════════════════════════
# ШАГ 2 — agnostic-mask (human parsing + openpose, CPU-friendly)
# ══════════════════════════════════════════
def generate_agnostic_masks(splits=("train", "test")):
    _ensure_repo()
    # human parsing явно считается на CPU (providers=['CPUExecutionProvider'] в
    # preprocess/humanparsing/run_parsing.py) — onnxruntime-gpu тут не нужен и
    # только конфликтует с версией CUDA в образе Kaggle (libcudart.so.13 ImportError)
    subprocess.run(["pip", "uninstall", "-y", "-q", "onnxruntime-gpu"], check=False)
    subprocess.run(["pip", "install", "-q", "onnxruntime", "opencv-python"], check=True)
    _download_checkpoints([
        "ckpt/humanparsing/parsing_atr.onnx",
        "ckpt/humanparsing/parsing_lip.onnx",
        "ckpt/openpose/ckpts/body_pose_model.pth",
    ])

    from PIL import Image
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from utils_mask import get_mask_location

    gpu_id = 0
    parsing_model = Parsing(gpu_id)
    openpose_model = OpenPose(gpu_id)

    for split in splits:
        image_dir = os.path.join(OUTPUT_DIR, split, "image")
        out_dir = os.path.join(OUTPUT_DIR, split, "agnostic-mask")
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(os.listdir(image_dir))
        print(f"[{split}] agnostic-mask: {len(names)} файлов")

        for i, name in enumerate(names):
            out_path = os.path.join(out_dir, name.replace(".jpg", "_mask.png"))
            if os.path.exists(out_path):
                continue
            human_img = Image.open(os.path.join(image_dir, name)).convert("RGB").resize((MASK_WIDTH, MASK_HEIGHT))
            keypoints = openpose_model(human_img)
            model_parse, _ = parsing_model(human_img)
            mask, _ = get_mask_location("hd", CATEGORY, model_parse, keypoints, width=MASK_WIDTH, height=MASK_HEIGHT)
            mask.save(out_path)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}")

        print(f"✓ [{split}] agnostic-mask готовы: {out_dir}")


# ══════════════════════════════════════════
# ШАГ 3 — image-densepose (detectron2 + DensePose — самая ломкая часть)
# ══════════════════════════════════════════
def install_detectron2():
    """
    Репозиторий IDM-VTON несёт vendored-копию detectron2/densepose с уже
    скомпилированным .so (gradio_demo/detectron2, gradio_demo/densepose) —
    но он собран под Python 3.9 (cp39) на конкретной версии CUDA/torch, и на
    текущем Kaggle/Colab-образе (обычно другой Python и другой torch) он,
    скорее всего, НЕ загрузится. Поэтому здесь — сборка detectron2 заново
    под текущее окружение, а не использование vendored .so.
    """
    subprocess.run(
        ["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"],
        check=True,
    )
    subprocess.run(
        [
            "pip", "install", "-q",
            "git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose",
        ],
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
        print(f"[{split}] image-densepose: {len(names)} файлов")

        for i, name in enumerate(names):
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path):
                continue
            human_img = Image.open(os.path.join(image_dir, name)).convert("RGB").resize((MASK_WIDTH, MASK_HEIGHT))
            img_arg = convert_PIL_to_numpy(_apply_exif_orientation(human_img), format="BGR")
            pose_img = args.func(args, img_arg)          # RGB<->BGR как в gradio_demo/app.py
            pose_img = pose_img[:, :, ::-1]
            Image.fromarray(pose_img).save(out_path)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}")

        print(f"✓ [{split}] image-densepose готовы: {out_dir}")


# ══════════════════════════════════════════
# ЗАПУСК — раскомментируйте нужные шаги
# ══════════════════════════════════════════
if __name__ == "__main__":
    download_and_reorganize()
    generate_agnostic_masks()
    generate_densepose()
    print("Готово. DATA_DIR для 3_train_lora.py:", OUTPUT_DIR)
