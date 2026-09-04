import argparse
import shutil
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def prepare_three_class_dataset(extracted_dir: Path, target_dir: Path):
    include_classes = ["COVID19", "PNEUMONIA", "TURBERCULOSIS"]
    for split in ["train", "val", "test"]:
        for cls in include_classes:
            src = extracted_dir / split / cls
            dst = target_dir / split / cls
            if not src.exists():
                raise FileNotFoundError(f"Missing expected folder: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Extract and prepare 3-class chest X-ray dataset.")
    parser.add_argument("--zip-path", type=str, default="archive (3).zip")
    parser.add_argument("--extract-dir", type=str, default="raw_dataset")
    parser.add_argument("--target-dir", type=str, default="dataset")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    extract_dir = Path(args.extract_dir)
    target_dir = Path(args.target_dir)

    if not zip_path.exists():
        
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    print(f"Extracting {zip_path} -> {extract_dir}")
    extract_zip(zip_path, extract_dir)
    print(f"Preparing 3-class dataset -> {target_dir}")
    prepare_three_class_dataset(extract_dir, target_dir)
    print("Done. Dataset is ready in:", target_dir)


if __name__ == "__main__":
    main()
