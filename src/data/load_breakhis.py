"""
src/data/load_breakhis.py

Walk the BreakHis directory tree and return a DataFrame.

Expected structure under dataset_path:
  histology_slides/breast/
    benign/SOB/<subtype>/<patient_folder>/<mag>X/*.png
    malignant/SOB/<subtype>/<patient_folder>/<mag>X/*.png

Returns a DataFrame with columns:
  path          absolute path to image
  label         0 = benign, 1 = malignant
  subtype       histological subtype string
  patient_dir   patient folder name (used as fallback patient ID)
  magnification 40X | 100X | 200X | 400X
"""

import pandas as pd
from pathlib import Path


BENIGN    = 0
MALIGNANT = 1


def load_breakhis(dataset_path: str) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    slides_root  = dataset_path / "histology_slides" / "breast"

    if not slides_root.exists():
        raise FileNotFoundError(
            f"Expected directory not found: {slides_root}\n"
            "Check that dataset_path points to the BreaKHis_v1 root and "
            "that the histology_slides/breast/ subfolder is present."
        )

    records = []
    for class_name, label in [("benign", BENIGN), ("malignant", MALIGNANT)]:
        class_dir = slides_root / class_name / "SOB"
        if not class_dir.exists():
            continue

        for subtype_dir in sorted(class_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            subtype = subtype_dir.name

            for patient_dir in sorted(subtype_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                for mag_dir in sorted(patient_dir.iterdir()):
                    if not mag_dir.is_dir():
                        continue
                    mag = mag_dir.name  # e.g. 40X

                    for img in sorted(mag_dir.glob("*.png")):
                        records.append({
                            "path":          str(img),
                            "label":         label,
                            "subtype":       subtype,
                            "patient_dir":   patient_dir.name,
                            "magnification": mag,
                        })

    if not records:
        raise RuntimeError(
            f"No PNG images found under {slides_root}.\n"
            "Verify the directory structure matches BreakHis v1."
        )

    df = pd.DataFrame(records)
    print(
        f"Loaded {len(df):,} images  "
        f"({(df.label == 0).sum():,} benign, "
        f"{(df.label == 1).sum():,} malignant)"
    )
    return df
