import os
from pathlib import Path

def check_casia_dataset(base_dir="./data"):
    casia1_path = Path(base_dir) / "CASIA1"
    casia2_path = Path(base_dir) / "CASIA2"

    def count_images(folder):
        if not folder.exists():
            return 0
        return sum(1 for f in folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"])

    print("="*50)
    print("🔍 Checking CASIA dataset structure...")
    print("="*50)

    # CASIA1
    if casia1_path.exists():
        print(f"✅ Found CASIA1 at: {casia1_path}")
        au_count = count_images(casia1_path / "Au")
        sp_count = count_images(casia1_path / "Sp")
        print(f"   - Authentic (Au): {au_count}")
        print(f"   - Tampered (Sp): {sp_count}")
    else:
        print("❌ CASIA1 folder not found!")

    # CASIA2
    if casia2_path.exists():
        print(f"\n✅ Found CASIA2 at: {casia2_path}")
        au_count = count_images(casia2_path / "Au")
        tp_count = count_images(casia2_path / "Tp")
        print(f"   - Authentic (Au): {au_count}")
        print(f"   - Tampered (Tp): {tp_count}")
    else:
        print("❌ CASIA2 folder not found!")

    print("="*50)

if __name__ == "__main__":
    check_casia_dataset("./data")  # change "./data" if your dataset is stored elsewhere
