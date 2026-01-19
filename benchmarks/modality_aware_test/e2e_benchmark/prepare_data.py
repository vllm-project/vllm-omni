# ruff: noqa: E402
import os
import sys

# handle hf mirror before importing datasets/huggingface_hub
_use_mirror = "--use_mainland_hf_mirror" in sys.argv

if _use_mirror:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(_proxy_key, None)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    print("🌏 Using mainland China HF mirror: https://hf-mirror.com")


import argparse
import json
import random
import re
import shutil
from collections import OrderedDict

import requests
import soundfile as sf
from datasets import Video, concatenate_datasets, load_dataset
from tqdm import tqdm

SAMPLE_SIZE = 500


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(item, save_path):
    image = item["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(save_path)
    caption = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
    return f"Does the image match the following caption? {caption} Answer in a short sentence."


def save_audio(item, save_path):
    audio_data = item["audio"]["array"]
    sample_rate = item["audio"]["sampling_rate"]
    sf.write(save_path, audio_data, sample_rate)
    text = item["text"].capitalize()
    return f"Does the audio match the following text? {text} Answer in a short sentence."


def convert_hf_url_to_mirror(hf_url: str, mirror_endpoint: str) -> str:
    pattern = r"hf://datasets/([^@]+)@([^/]+)/(.+)"
    match = re.match(pattern, hf_url)
    if match:
        repo_id = match.group(1)
        revision = match.group(2)
        filepath = match.group(3)
        return f"{mirror_endpoint}/datasets/{repo_id}/resolve/{revision}/{filepath}"
    else:
        raise ValueError(f"Cannot parse hf:// URL: {hf_url}")


def save_video(item, save_path):
    video_data = item.get("video")

    if video_data is None:
        raise ValueError("No video column found in item")

    if isinstance(video_data, dict):
        if video_data.get("bytes") is not None:
            with open(save_path, "wb") as f:
                f.write(video_data["bytes"])
        elif video_data.get("path") is not None:
            video_path = video_data["path"]
            if video_path.startswith("hf://"):
                mirror_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
                https_url = convert_hf_url_to_mirror(video_path, mirror_endpoint)
                response = requests.get(https_url, timeout=60)
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(response.content)
            else:
                shutil.copy(video_path, save_path)
        else:
            raise ValueError(f"Video dict has neither valid 'bytes' nor 'path': {video_data.keys()}")
    else:
        raise ValueError(f"Unexpected video format (expected dict): {type(video_data)}")

    return ""


def prepare_datasets(data_path_prefix):
    caption_pool = []

    configs = OrderedDict(
        [
            (
                "image",
                {
                    "path": "jxie/coco_captions",
                    "split": "test",
                    "handler": save_image,
                    "ext": ".jpg",
                    "prompt_key": "caption",
                    "dedup_key": "cocoid",
                },
            ),
            (
                "audio",
                {
                    "path": "openslr/librispeech_asr",
                    "name": "clean",
                    "split": "test",
                    "handler": save_audio,
                    "ext": ".wav",
                    "prompt_key": "text",
                },
            ),
            (
                "video",
                {
                    "path": "nateraw/kinetics-mini",
                    "split": ["train", "validation"],
                    "handler": save_video,
                    "ext": ".mp4",
                    "prompt_key": "label",
                },
            ),
            (
                "text",
                {
                    "path": "stanfordnlp/web_questions",
                    "split": "test",
                    "handler": None,
                    "ext": None,
                    "prompt_key": "question",
                },
            ),
        ]
    )

    print(f"🚀 Starting data preparation in {data_path_prefix}...")

    for modality, cfg in configs.items():
        print(f"\nProcessing [{modality}]...")

        base_dir = os.path.join(data_path_prefix, modality)
        files_dir = os.path.join(base_dir, "files")
        if modality != "text":
            ensure_dir(files_dir)
        else:
            ensure_dir(base_dir)

        try:
            splits_to_load = cfg["split"]
            if isinstance(splits_to_load, str):
                splits_to_load = [splits_to_load]

            ds_list = []

            for split_name in splits_to_load:
                load_args = {"path": cfg["path"], "split": split_name}
                if "name" in cfg:
                    load_args["name"] = cfg["name"]

                sub_ds = load_dataset(**load_args, streaming=True)

                if modality == "video":
                    sub_ds = sub_ds.cast_column("video", Video(decode=False))

                ds_list.append(sub_ds)

            if len(ds_list) == 1:
                ds = ds_list[0]
            else:
                print(f"⚡ Concatenating {len(ds_list)} splits for {modality}...")
                ds = concatenate_datasets(ds_list)

            if modality == "video":
                print("⚡ Video column cast to decode=False verified.")

        except Exception as e:
            print(f"⚠️ Failed to load dataset for {modality}: {e}")
            continue

        metadata_list = []
        seen_ids = set()
        success_count = 0

        pbar = tqdm(total=SAMPLE_SIZE, desc=f"Saving {modality}")

        for i, item in enumerate(ds):
            if success_count >= SAMPLE_SIZE:
                break

            if cfg.get("dedup_key"):
                uid = item.get(cfg["dedup_key"])
                if uid is None and modality == "image":
                    uid = item.get("image_id") or item.get("id")

                if uid is not None:
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)

            try:
                filename = f"{modality}_{success_count:05d}{cfg['ext']}" if cfg["ext"] else None
                file_abs_path = os.path.join(files_dir, filename) if filename else None

                prompt = ""

                if modality == "text":
                    prompt = f"{item[cfg['prompt_key']]} Answer in a short sentence."
                    metadata_list.append({"idx": success_count, "prompt": prompt})
                else:
                    handler_result = cfg["handler"](item, file_abs_path)

                    if modality == "video":
                        if caption_pool:
                            caption = random.choice(caption_pool)
                        else:
                            caption = "a person performing an action"

                        prompt = f"Does the video match the following caption? {caption} Answer in a short sentence."

                    elif modality == "image":
                        prompt = handler_result

                    elif modality == "audio":
                        prompt = handler_result

                    metadata_list.append({"idx": success_count, "file": filename, "prompt": prompt})

                    if modality == "image":
                        raw_cap = item.get("caption")
                        if isinstance(raw_cap, list):
                            raw_cap = raw_cap[0]
                        caption_pool.append(raw_cap)

                success_count += 1
                pbar.update(1)

            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

        pbar.close()

        if success_count < SAMPLE_SIZE:
            print(f"⚠️ Dataset only has {success_count} samples (requested {SAMPLE_SIZE})")

        jsonl_path = os.path.join(base_dir, "metadata.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in metadata_list:
                f.write(json.dumps(entry) + "\n")

        print(f"✅ Saved {len(metadata_list)} samples to {jsonl_path}")

    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare multimodal datasets for benchmarking")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to save the prepared datasets")
    parser.add_argument(
        "--use_mainland_hf_mirror", action="store_true", help="Use hf-mirror.com for users in mainland China"
    )
    args = parser.parse_args()
    prepare_datasets(args.data_dir)
