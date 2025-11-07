#!/usr/bin/env python3
"""
DICOM De-Identification Script
---------------------------------
- Works on a single DICOM file or a directory (recursively).
- Produces anonymized copies without overwriting originals.
- Edit ANONYMIZATION_SCHEME below to control how tags are modified.
"""

import os
import argparse
from pydicom import dcmread
from pydicom.uid import generate_uid, UID
from pydicom.errors import InvalidDicomError
from pydicom.dataset import Dataset

# ================= EDIT THIS SECTION =========================================
# Define which DICOM fields to remove or replace.
# Keys must match DICOM keyword names.
#
# * None = delete the tag entirely
# * String = replace with given value
# * "UID" = generate a deterministic anonymized UID
# ============================================================================

ANONYMIZATION_SCHEME = {
    "PatientName": "ANON",
    "PatientID": "000000",
    "PatientBirthDate": None,
    "PatientSex": None,
    "InstitutionName": "ANON_HOSPITAL",
    "ReferringPhysicianName": None,
    "AccessionNumber": None,
    "StudyID": "STUDY",
    "StudyInstanceUID": "UID",
    "SeriesInstanceUID": "UID",
    "FrameOfReferenceUID": "UID"
}

# ============================================================================


def anonymize_dataset(ds: Dataset):
    """Apply anonymization scheme to a DICOM dataset."""
    for key, action in ANONYMIZATION_SCHEME.items():
        if key not in ds:
            continue

        if action is None:
            del ds[key]
        elif action == "UID":
            ds[key].value = UID(generate_uid())
        else:
            ds[key].value = action

    # Recommended: remove private tags
    ds.remove_private_tags()

    return ds


def process_file(in_path: str, out_path: str):
    """Load, anonymize, and save a single DICOM file."""
    try:
        ds = dcmread(in_path)
    except InvalidDicomError:
        print(f"Skipping non-DICOM file: {in_path}")
        return

    ds = anonymize_dataset(ds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.save_as(out_path, write_like_original=False)
    print(f"[OK] {in_path} → {out_path}")


def process_directory(in_dir: str, out_dir: str):
    """Recursively anonymize all DICOM files in a directory."""
    for root, _, files in os.walk(in_dir):
        for f in files:
            in_path = os.path.join(root, f)

            relative = os.path.relpath(root, in_dir)
            out_path = os.path.join(out_dir, relative, f)

            process_file(in_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="DICOM Anonymization Tool")
    parser.add_argument("input", help="DICOM file or directory to anonymize")
    parser.add_argument("output", help="Output file or directory")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        process_file(args.input, args.output)


if __name__ == "__main__":
    main()
