"""
LoRA-дообучение IDM-VTON — версия, сверенная с реальной архитектурой
github.com/yisol/IDM-VTON (train_xl.py, src/unet_hacked_tryon.py,
src/unet_hacked_garmnet.py, inference.py — код прочитан из клона репозитория).

Реальная архитектура — связка из нескольких моделей:

  1. TryonNet (src/unet_hacked_tryon.py) — основной denoising UNet (SDXL).
     На вход у него 13-канальный латент = конкатенация:
     [шумный латент фото (4) + маска (1) + латент "человека без одежды" (4)
      + латент DensePose-карты позы (4)].
  2. GarmentNet (src/unet_hacked_garmnet.py) — второй UNet, прогоняет латент
     одежды и отдаёт список feature-карт (garment_features). TryonNet
     подмешивает их в свой self-attention на каждом блоке — это и есть
     механизм "надевания" именно этой вещи.
  3. IP-Adapter (CLIP image encoder + Resampler) — второй, "визуальный"
     канал описания одежды: его токены конкатенируются к тексту в
     encoder_hidden_states. Resampler уже вшит в чекпоинт как
     unet.encoder_hid_proj — его нужно вызывать вручную (сам forward
     TryonNet его не вызывает, см. комментарий в load_models()).
  4. Два текстовых энкодера SDXL (CLIP-L и CLIP-G) — два промпта:
     "model is wearing X" (для TryonNet) и "a photo of X" (для GarmentNet).

Слои to_q/to_k/to_v/to_out.0 внутри attention — обычный класс
diffusers.models.attention_processor.Attention, "хак" не трогает его, а
только то, что в него подаётся (см. import в src/attentionhacked_tryon.py).
Поэтому LoRA (peft) вешается на эти слои штатно, как в любом SDXL-скрипте.

БАЗОВЫЙ ЧЕКПОИНТ:
Обучаем LoRA не поверх голой SDXL, а поверх уже готовой "yisol/IDM-VTON" —
там TryonNet/GarmentNet/Resampler уже обучены на VITON-HD+DressCode. LoRA
только доучивает модель под вашу нишу, а не учит примерке с нуля.

⚠️ ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ДАННЫХ (без этого dataset-класс упадёт):
    DATA_DIR/
      train_pairs.txt
      train/
        image/                     — фото человека в одежде (jpg)
        cloth/                     — фото одежды на белом фоне (то же имя файла)
        agnostic-mask/             — БИНАРНАЯ маска "где была одежда"
                                      (<imagename без .jpg>_mask.png)
        image-densepose/           — DensePose-карта человека (то же имя файла)
        vitonhd_train_tagged.json  — текстовые теги одежды (идут вместе с VITON-HD)

    agnostic-mask и image-densepose генерируются 1_prepare_dataset.py +
    2_resume_densepose.py — см. эти файлы.

ЧЕСТНО ПРО VRAM (Kaggle/Colab T4, 15-16GB):
    Одновременно в памяти — две SDXL-модели (~2.6B параметров каждая) + два
    текстовых энкодера + CLIP image encoder. Даже с LoRA, fp16, gradient
    checkpointing и 8-bit Adam это КРАЙ возможностей бесплатного T4.
    Рекомендации по умолчанию ниже (512x384, batch=1) — это не "рекомендуемое
    качество", а "то, что вообще имеет шанс влезть в память". Если всё равно
    OOM — дальше снижайте RESOLUTION или LORA_RANK.

Полный цикл на T4: по опыту авторов полноценного train_xl.py — часы на
эпоху при batch=6/768x1024 на куда более мощных GPU. На T4 с LoRA и
уменьшённым разрешением ждите пропорционально дольше на шаг; ориентир —
проверять качество на промежуточных чекпоинтах, а не гнаться за TRAIN_STEPS.

⚠️ СТАТУС (2026-08-25): последний прогон (Version 2, Add Input сделан)
всё равно упал на первом шаге с той же ошибкой "не нашёл _output_.zip".
Причина — баг в restore_dataset(): Kaggle монтирует output подключённого
ноутбука на глубину в несколько папок (напр.
/kaggle/input/notebooks/<user>/<slug>/_output_.zip), а старый код искал
только на один уровень (/kaggle/input/*/_output_.zip) и ничего не находил,
даже когда Input был подключён верно. Исправлено на рекурсивный поиск
(glob "**", recursive=True).

Следующий прогон после этого фикса дошёл до install_deps(), но упал там:
`ImportError: cannot import name 'cached_download' from 'huggingface_hub'`
— diffusers==0.25.0 использует эту функцию, а в свежем образе Kaggle уже
стоит huggingface_hub без неё (убрали в поздних релизах). Исправлено —
пакеты теперь явно пинят huggingface_hub==0.20.3 последним пунктом списка.

Следующий прогон прошёл install_deps() и restore_dataset(), дошёл до
apply_lora(), упал там: `RuntimeError: CUDA Setup failed despite GPU being
available` — это не из нашего кода, а из bitsandbytes==0.39.0. Он падает
прямо при импорте на текущем CUDA-окружении Kaggle, а peft.get_peft_model()
безусловно пытается его импортировать, если пакет вообще установлен (наш
try/except в train() тут не спасает — ошибка раньше, внутри apply_lora()).
Исправлено — bitsandbytes больше не ставим вообще, используется обычный
AdamW. Само обучение ещё ни разу не выполнялось — это первая попытка после
всех трёх фиксов.
"""

import os
import sys
import subprocess

# ══════════════════════════════════════════
# КОНФИГУРАЦИЯ — поменяйте под себя
# ══════════════════════════════════════════
IDM_VTON_REPO   = "/kaggle/working/IDM-VTON"                  # куда клонируем репозиторий (нужен для src/ и ip_adapter/)
DATA_DIR        = "/kaggle/working/viton-hd-idm"               # сюда распакуется датасет из PREV_DATASET_NOTEBOOK (см. restore_dataset())
OUTPUT_DIR      = "/kaggle/working/lora_output"                # куда сохранится LoRA
BASE_MODEL      = "yisol/IDM-VTON"                             # уже обученный чекпоинт с HuggingFace
HF_TOKEN        = None                                          # впишите свой токен, если BASE_MODEL окажется gated

# ноутбук с готовым датасетом (2_resume_densepose.py) — подключите его через
# Add Input → Notebook Output, ПЕРЕД запуском этого скрипта
PREV_DATASET_NOTEBOOK = "notebookd4fbf8b5b7"

TRAIN_STEPS                 = 1000     # 500-1500 обычно достаточно для LoRA
LEARNING_RATE                = 1e-4
LORA_RANK                    = 16       # 8-32, больше = точнее но тяжелее
LORA_ALPHA                   = 16
BATCH_SIZE                   = 1        # T4 тянет только batch_size=1 на этой архитектуре
GRADIENT_ACCUMULATION_STEPS   = 4        # эмулирует batch=4, не тратя больше VRAM
HEIGHT                       = 512      # понижено с "родных" 1024 ради T4, см. предупреждение выше
WIDTH                        = 384
SNR_GAMMA                    = None     # 5.0 включает SNR-взвешивание лосса, как в train_xl.py
CHECKPOINT_EVERY_STEPS        = 200
ENABLE_XFORMERS               = False   # см. предупреждение в apply_lora() — по умолчанию выключено!


# ══════════════════════════════════════════
# ШАГ 0 — Распаковка готового датасета из output предыдущего ноутбука
# (2_resume_densepose.py)
# ══════════════════════════════════════════
def restore_dataset():
    import glob
    import zipfile

    if os.path.isdir(DATA_DIR) and os.path.exists(os.path.join(DATA_DIR, "train_pairs.txt")):
        print(f"✓ Датасет уже на месте: {DATA_DIR}")
        return

    # Kaggle монтирует output чужого/своего ноутбука на разной глубине в
    # зависимости от контекста (напр. /kaggle/input/notebooks/<user>/<slug>/
    # или /kaggle/input/<slug>/) — одноуровневый glob это пропускал и был
    # источником "не нашёл _output_.zip" даже при подключённом Input.
    # Ищем рекурсивно по любой глубине.
    candidates = glob.glob(f"/kaggle/input/**/{PREV_DATASET_NOTEBOOK}/_output_.zip", recursive=True)
    if not candidates:
        candidates = glob.glob("/kaggle/input/**/_output_.zip", recursive=True)
    if not candidates:
        raise RuntimeError(
            f"Не нашёл _output_.zip с датасетом в /kaggle/input — подключите ноутбук "
            f"'{PREV_DATASET_NOTEBOOK}' через Add Input → Notebook Output перед запуском."
        )

    print(f"Распаковываю датасет из {candidates[0]}...")
    os.makedirs("/kaggle/working", exist_ok=True)
    with zipfile.ZipFile(candidates[0]) as zf:
        zf.extractall("/kaggle/working")
    print(f"✓ Датасет распакован в {DATA_DIR}")

    # быстрая проверка, что ничего не побилось при zip/unzip (особенно
    # image/cloth — они были symlink'ами на исходный VITON-HD)
    for split in ("train", "test"):
        for sub in ("image", "cloth", "agnostic-mask", "image-densepose"):
            d = os.path.join(DATA_DIR, split, sub)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  [{split}] {sub}: {n} файлов" + ("  ⚠️ ПУСТО!" if n == 0 else ""))


# ══════════════════════════════════════════
# ШАГ 1 — Установка зависимостей + клон архитектуры IDM-VTON
# ══════════════════════════════════════════
def install_deps():
    # версии как в environment.yaml оригинального репозитория — код в src/ это
    # по сути слегка изменённые внутренности diffusers конкретной версии,
    # другая версия diffusers может не совпасть по API и сломать импорт
    packages = [
        "diffusers==0.25.0",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "peft==0.8.2",       # LoRA
        # bitsandbytes НЕ ставим. peft.get_peft_model() сам пытается
        # `import bitsandbytes`, если пакет вообще установлен (проверка
        # is_bnb_available() смотрит только на наличие пакета, не на
        # рабочесть) — а bitsandbytes==0.39.0 падает RuntimeError'ом прямо
        # при импорте, потому что не может определить CUDA на текущем
        # образе Kaggle. Наш try/except вокруг bitsandbytes в train() тут
        # не спасает — ошибка происходит раньше, внутри apply_lora().
        # Без bitsandbytes используется обычный AdamW — для LoRA (обучаются
        # только маленькие adapter-веса, не вся модель) разница в VRAM
        # незначительна.
        "Pillow",
        # diffusers==0.25.0 (конец 2023) импортирует cached_download из
        # huggingface_hub — эту функцию убрали в поздних версиях
        # huggingface_hub, которые уже стоят в свежем образе Kaggle
        # (ImportError: cannot import name 'cached_download'). Ставим её
        # последней, чтобы никакой более поздний pip install из списка выше
        # не подтянул обратно свежую версию как транзитивную зависимость.
        "huggingface_hub==0.20.3",
    ]
    for pkg in packages:
        subprocess.run(["pip", "install", "-q", pkg], check=True)

    if not os.path.isdir(IDM_VTON_REPO):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yisol/IDM-VTON.git", IDM_VTON_REPO],
            check=True,
        )
    if IDM_VTON_REPO not in sys.path:
        sys.path.insert(0, IDM_VTON_REPO)

    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)

    print("✓ Зависимости установлены, репозиторий IDM-VTON готов")


# ══════════════════════════════════════════
# ШАГ 2 — Модели: TryonNet + GarmentNet + IP-Adapter + текстовые энкодеры
# Загрузка зеркалит inference.py оригинального репозитория (грузим готовый
# чекпоинт "yisol/IDM-VTON" по субпапкам), а не train_xl.py (тот собирает
# модель с нуля из голой SDXL — нам это не нужно, мы дообучаем готовое).
# ══════════════════════════════════════════
def load_models(device, weight_dtype):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        BASE_MODEL, subfolder="text_encoder_2", torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        BASE_MODEL, subfolder="image_encoder", torch_dtype=weight_dtype
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        BASE_MODEL, subfolder="unet_encoder", torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=weight_dtype)

    # Resampler IP-Adapter уже "вшит" в unet как unet.encoder_hid_proj.
    # forward TryonNet его сам НЕ вызывает — в src/unet_hacked_tryon.py эта
    # строка буквально закомментирована в оригинале (`# image_embeds =
    # self.encoder_hid_proj(image_embeds)...`), TryonNet просто конкатенирует
    # то, что ему передали в added_cond_kwargs["image_embeds"]. Поэтому
    # прогоняем Resampler вручную, ровно как это делает train_xl.py со своим
    # отдельным image_proj_model.
    image_proj_model = unet.encoder_hid_proj

    for m in (vae, text_encoder, text_encoder_2, image_encoder, unet_encoder, image_proj_model):
        m.requires_grad_(False)
        m.to(device)
        m.eval()

    unet.requires_grad_(False)  # LoRA сама включит grad только на adapter-слоях
    unet.to(device)
    unet.enable_gradient_checkpointing()  # до LoRA-обёртки — так безопаснее

    return dict(
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        unet=unet,
        image_proj_model=image_proj_model,
    )


def apply_lora(unet):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # attention-слои TryonNet
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # мастер-веса LoRA держим в fp32 для стабильности обучения (та же схема,
    # что в официальных SDXL LoRA-скриптах diffusers) — сам форвард всё
    # равно идёт в fp16 через autocast Accelerator(mixed_precision="fp16")
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if ENABLE_XFORMERS:
        # ⚠️ РИСК: у TryonNet кастомные attn-процессоры (IP-Adapter +
        # garment-features), а enable_xformers_memory_efficient_attention()
        # в diffusers по умолчанию подменяет ВСЕ attn-процессоры на общий
        # XFormersAttnProcessor — это может тихо сломать вклейку
        # garment_features/IP-adapter токенов. Проверьте генерацию после
        # первого чекпоинта, если включаете этот флаг.
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("xformers недоступен или несовместим, продолжаем без него:", e)

    return unet


def save_lora(unet, path):
    os.makedirs(path, exist_ok=True)
    unet.save_pretrained(path)  # peft: сохраняет только adapter-веса (обычно 20-150 МБ)
    print(f"✓ LoRA сохранена в {path}")


# ══════════════════════════════════════════
# ШАГ 3 — Обучение
# Логика шага — прямое повторение train_xl.py оригинального репозитория
# (тот же порядок: vae.encode латентов, конкатенация 13 каналов, два
# текстовых промпта, IP-adapter токены, garment_features из GarmentNet,
# MSE/SNR-лосс), но: веса заморожены везде кроме LoRA, батч=1 + grad
# accumulation, чекпоинты — это только LoRA-адаптер, а не вся модель.
# ══════════════════════════════════════════
def train():
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from torch.utils.data import DataLoader
    from diffusers.training_utils import compute_snr

    from train_xl import VitonHDDataset  # официальный dataset-класс репозитория, без изменений

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        project_config=ProjectConfiguration(project_dir=OUTPUT_DIR),
    )
    device = accelerator.device
    weight_dtype = torch.float16

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = load_models(device, weight_dtype)
    unet = apply_lora(models["unet"])

    optimizer_cls = torch.optim.AdamW
    try:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    except ImportError:
        print("bitsandbytes недоступен — используем обычный AdamW (займёт больше памяти)")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=LEARNING_RATE)

    dataset = VitonHDDataset(
        dataroot_path=DATA_DIR, phase="train", order="paired", size=(HEIGHT, WIDTH)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    tokenizer, tokenizer_2 = models["tokenizer"], models["tokenizer_2"]
    text_encoder, text_encoder_2 = models["text_encoder"], models["text_encoder_2"]
    vae, image_encoder, unet_encoder = models["vae"], models["image_encoder"], models["unet_encoder"]
    image_proj_model, noise_scheduler = models["image_proj_model"], models["noise_scheduler"]

    def encode_prompt(captions):
        ids = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        ids2 = tokenizer_2(
            captions, max_length=tokenizer_2.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        out = text_encoder(ids, output_hidden_states=True)
        out2 = text_encoder_2(ids2, output_hidden_states=True)
        embeds = torch.concat([out.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
        pooled = out2[0]
        return embeds, pooled

    def compute_time_ids(bsz):
        # ровно как в train_xl.py: original_size=target_size=(HEIGHT, HEIGHT),
        # crop=(0,0) — это поведение оригинального скрипта, не опечатка,
        # сохраняем как есть ради совместимости с тем, на чём училась модель
        row = [HEIGHT, HEIGHT, 0, 0, HEIGHT, HEIGHT]
        return torch.tensor([row] * bsz, device=device)

    print(f"Начинаем LoRA-обучение: {TRAIN_STEPS} шагов на {len(dataset)} парах")
    unet.train()
    step = 0
    while step < TRAIN_STEPS:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    pixel_values = batch["image"].to(device, dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                    masked_latents = vae.encode(
                        batch["im_mask"].to(device, dtype=vae.dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                    pose_map = vae.encode(
                        batch["pose_img"].to(device, dtype=vae.dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                    cloth_latents = vae.encode(
                        batch["cloth_pure"].to(device, dtype=vae.dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                    masks = batch["inpaint_mask"].to(device)
                    mask = F.interpolate(masks, size=(HEIGHT // 8, WIDTH // 8))

                    bsz = model_input.shape[0]
                    noise = torch.randn_like(model_input)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
                    latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

                    encoder_hidden_states, pooled_text_embeds = encode_prompt(batch["caption"])
                    text_embeds_cloth, _ = encode_prompt(batch["caption_cloth"])

                    clip_pixel_values = torch.cat(
                        [batch["cloth"][i] for i in range(bsz)], dim=0
                    ).to(device, dtype=image_encoder.dtype)
                    clip_hidden = image_encoder(clip_pixel_values, output_hidden_states=True).hidden_states[-2]
                    ip_tokens = image_proj_model(clip_hidden)

                    add_time_ids = compute_time_ids(bsz)
                    added_cond_kwargs = {
                        "text_embeds": pooled_text_embeds,
                        "time_ids": add_time_ids,
                        "image_embeds": ip_tokens,
                    }

                    _, reference_features = unet_encoder(
                        cloth_latents, timesteps, text_embeds_cloth, return_dict=False
                    )
                    reference_features = list(reference_features)

                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    garment_features=reference_features,
                ).sample

                if SNR_GAMMA is None:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, SNR_GAMMA * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                if step % 50 == 0:
                    print(f"Шаг {step}/{TRAIN_STEPS} · loss={loss.item():.4f}")
                if step % CHECKPOINT_EVERY_STEPS == 0 and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(unet)
                    save_lora(unwrapped, os.path.join(OUTPUT_DIR, f"checkpoint-{step}"))
                if step >= TRAIN_STEPS:
                    break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        save_lora(unwrapped, OUTPUT_DIR)
        print("Готово. Загрузите файлы из OUTPUT_DIR в HuggingFace repo для использования.")


# ══════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════
if __name__ == "__main__":
    restore_dataset()
    install_deps()
    train()
