"""
Download official Chatterbox audio prompts for all languages
"""
import os
import requests
from pathlib import Path

# Official audio prompts from Chatterbox demo
LANGUAGE_PROMPTS = {
    "ar": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
    "da": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
    "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "el": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
    "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    "fi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
    "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "he": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
    "hi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
    "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    "ja": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
    "ko": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
    "ms": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
    "nl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
    "no": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
    "pl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
    "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
    "ru": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
    "sv": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
    "sw": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
    "tr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
    "zh": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
}

def download_file(url: str, destination: Path):
    """Download a file from URL to destination"""
    if destination.exists():
        print(f"✓ Already exists: {destination.name}")
        return True

    try:
        print(f"⬇️  Downloading: {destination.name}...", end=" ")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        destination.write_bytes(response.content)
        print(f"✅ Done ({len(response.content) // 1024} KB)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    # Create audio prompts directory
    base_dir = Path(__file__).parent.parent
    prompts_dir = base_dir / "audio_prompts" / "multilingual"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Saving audio prompts to: {prompts_dir}")
    print(f"📥 Downloading {len(LANGUAGE_PROMPTS)} language prompts...\n")

    success_count = 0
    for lang_code, url in LANGUAGE_PROMPTS.items():
        filename = f"{lang_code}_prompt.flac"
        destination = prompts_dir / filename

        if download_file(url, destination):
            success_count += 1

    print(f"\n✅ Downloaded {success_count}/{len(LANGUAGE_PROMPTS)} prompts successfully!")
    print(f"📁 Location: {prompts_dir}")

if __name__ == "__main__":
    main()
