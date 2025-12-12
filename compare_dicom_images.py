import pydicom
from pydicom.tag import Tag

def compare_dicom_headers(file_path_a: str, file_path_b: str) -> dict:
    """
    Compares the headers of two DICOM files, excluding PixelData.

    Args:
        file_path_a: Path to the first DICOM file.
        file_path_b: Path to the second DICOM file.

    Returns:
        A dictionary of differences found, or an empty dictionary if they match.
    """
    try:
        # Read the files, stopping before reading pixel data for efficiency
        ds_a = pydicom.dcmread(file_path_a, stop_before_pixels=True)
        ds_b = pydicom.dcmread(file_path_b, stop_before_pixels=True)
    except Exception as e:
        return {"error": f"Failed to read one or both DICOM files: {e}"}

    # Tag for PixelData (7FE0, 0010)
    PIXEL_DATA_TAG = Tag(0x7FE0, 0x0010)
    
    # Store all differences found
    differences = {}

    # --- 1. Compare Elements in Dataset A against Dataset B ---
    for tag_a in ds_a.keys():
        # 1a. Skip Pixel Data
        if tag_a == PIXEL_DATA_TAG:
            continue

        tag_name = ds_a[tag_a].name

        # 1b. Check if the tag exists in B
        if tag_a not in ds_b:
            differences[tag_name] = f"Tag {tag_a} present in A but missing in B."
            continue

        # 1c. Compare values
        val_a = ds_a[tag_a].value
        val_b = ds_b[tag_a].value

        if val_a != val_b:
            # Handle Sequence (SQ) tags recursively if needed, but for simple comparison, 
            # comparing string representations is often easiest/fastest.
            if ds_a[tag_a].VR == 'SQ':
                # For sequences, a simple value comparison might miss deep differences.
                # A full recursive comparison is more complex, so we flag the sequence difference.
                if str(val_a) != str(val_b):
                    differences[tag_name] = f"Sequence difference at {tag_a}: A='{str(val_a)[:50]}...', B='{str(val_b)[:50]}...'"
            else:
                differences[tag_name] = f"Value difference at {tag_a}: A='{val_a}', B='{val_b}'"

    # --- 2. Check for Elements in Dataset B that are missing in Dataset A ---
    # This step ensures that B doesn't contain extra header tags A lacks
    for tag_b in ds_b.keys():
        if tag_b == PIXEL_DATA_TAG:
            continue
            
        if tag_b not in ds_a:
            tag_name = ds_b[tag_b].name
            differences[tag_name] = f"Tag {tag_b} present in B but missing in A."

    return differences

# -----------------
# Example Usage
# -----------------
# Replace 'file1.dcm' and 'file2.dcm' with your actual file paths
# differences = compare_dicom_headers('file1.dcm', 'file2.dcm')

# if not differences:
#     print("✅ Headers are identical (excluding PixelData).")
# else:
#     print("❌ Headers differ. Details:")
#     for tag_name, message in differences.items():
#         print(f"  - {tag_name}: {message}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch CTA-DEFACE pipeline: DICOM dirs -> defaced NIfTI + DICOM (full header reuse)."
    )
    # Argument Definitions
    ap.add_argument(
        "-i", "--dicom-root-in", required=True,
        help="Root input folder containing DICOM case directories (or a single DICOM dir)."
    )
    # ... (other ap.add_argument calls)

    args = ap.parse_args()

    # Process and clean up arguments
    root_in = os.path.abspath(args.dicom_root_in)
    # ...

    # Find cases
    cases = find_case_dirs(root_in)
    # ...

    extra_args = args.cta_extra_args if args.cta_extra_args is not None else []

    # Loop through cases, calling the core logic function
    for case_dir in cases:
        process_case(case_dir, root_in, root_out_dicom, root_out_nifti, work_root, extra_args=extra_args)

    print("\nAll cases processed.")


if __name__ == "__main__":
    main()
