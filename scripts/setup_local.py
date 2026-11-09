"""Prepare local configurations and optionally obtain the public model weights."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    target = ROOT / ".local-configs"
    target.mkdir(exist_ok=True)
    for template in sorted((ROOT / "configs").glob("*.yaml")):
        output = target / template.name
        output.write_text(template.read_text().replace("__PROJECT_ROOT__", str(ROOT)))
    print(f"Configurations prepared in {target}")
    print("Project launchers use these resolved files by default.")
    if args.download_models:
        from huggingface_hub import snapshot_download
        for name in ("Kronos-small", "Kronos-Tokenizer-base"):
            snapshot_download(
                repo_id=f"NeoQuasar/{name}",
                local_dir=ROOT / "pretrained" / name,
                allow_patterns=["*.json", "*.safetensors", "README.md"],
            )

if __name__ == "__main__":
    main()
