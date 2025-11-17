#!/usr/bin/env python3
import os
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

def print_dataset(ds, indent=0):
    """Recursively print all DICOM metadata except PixelData."""
    prefix = " " * indent
    for elem in ds:
        if elem.keyword == "PixelData":
            # Skip pixel data (could be hundreds of MB)
            print(f"{prefix}{elem.tag} {elem.keyword}: <PixelData omitted>")
            continue

        if elem.VR == "SQ":  # sequence — list of datasets
            print(f"{prefix}{elem.tag} {elem.keyword} (Sequence)")
            for i, item in enumerate(elem.value):
                print(f"{prefix}  Item {i}:")
                print_dataset(item, indent + 4)
        else:
            print(f"{prefix}{elem.tag} {elem.keyword}: {elem.value}")

def process_directory(dicom_dir):
    for root, dirs, files in os.walk(dicom_dir):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                ds = dcmread(path, stop_before_pixels=True)  # automatically excludes PixelData
            except InvalidDicomError:
                continue  # skip non-DICOM
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            print("\n=======================================")
            print(f"FILE: {path}")
            print("=======================================")
            print_dataset(ds)
            return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print DICOM metadata (excluding Pixel Data)")
    parser.add_argument("directory", help="Path to a DICOM directory")
    args = parser.parse_args()

    process_directory(args.directory)
