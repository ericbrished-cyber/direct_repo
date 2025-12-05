import os
import re
import shutil
from pathlib import Path

# ---- 1. DEV pmcids ----
dev_pmcids = [
    57750,
    1216327,
    2681019,
    3276927,
    3580751,
    3751573,
    4450164,
    5244530,
    5498715,
    5771543,
]

#test is complement to these pmcids

dev_pmcids = set(dev_pmcids)

# ---- 2. Paths ----
# 2. Base dir = folder where dev_test_split.py lives
base_dir = Path(__file__).resolve().parent

pdf_root    = base_dir / "PDF_all"   # FOLDER: data/PDF_all
dev_folder  = base_dir / "PDF_dev"
test_folder = base_dir / "PDF_test"

os.makedirs(dev_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# ---- 3. Split into DEV / TEST based on pmcid in filename ----
for fname in os.listdir(pdf_root):
    if not fname.lower().endswith(".pdf"):
        continue

    m = re.search(r'(\d+)', fname)

    if not m:
        print(f"Skipping (no pmcid found): {fname}")
        continue

    pmcid = int(m.group(1))
    src = os.path.join(pdf_root, fname)

    if pmcid in dev_pmcids:
        dst = os.path.join(dev_folder, fname)
        split = "DEV"
    else:
        dst = os.path.join(test_folder, fname)
        split = "TEST"

    shutil.copy2(src, dst)  # or shutil.move(src, dst)
    print(f"{split}: {fname}")
