#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DICOM De-Identification: Pro Edition
------------------------------------
Features:
- Config profiles (GDPR-STRICT, RESEARCH-PSEUDONYMIZED)
- Deterministic pseudonymization with user-editable SALT
- New anonymized UIDs (NEWUID) or pseudonymous UIDs (PSEUDOUID)
- Pixel black-out for burned-in overlays (best-effort, 2D/3D)
- Parallel processing (multiprocessing.Pool)
- Short file names with run-wide timestamp
- CSV audit log (input, output, tag, old, new, PID)

Usage examples:
  python dicom_anonymize_pro.py -i /path/to/DICOM -p GDPR-STRICT
  python dicom_anonymize_pro.py -i study.dcm -p RESEARCH-PSEUDONYMIZED --salt "my-secret"

Notes:
- This script never overwrites sources; it writes to a new output folder: ANON_EXPORT_YYYYMMDD_HHMMSS
"""

import os
import csv
import argparse
import datetime
import hashlib
import multiprocessing as mp
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import UID, generate_uid
from pydicom.errors import InvalidDicomError
import cv2
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================== USER-EDITABLE CONFIG ============================

# Edit the SALT to make your pseudonyms deterministic but private to your site.
# Changing the SALT changes the pseudonym results (IDs and UIDs).
DEFAULT_SALT = "CHANGE-ME-SALT-RHYTHM-UOC-2025"

# Numeric root for pseudonymous UIDs (must be numeric dotted string).
# Use an enterprise/organization root if you have one; this is an example.
PSEUDO_UID_ROOT = "1.2.826.0.1.3680043.10.54321"

# ========================= END USER-EDITABLE CONFIG ==========================


# --------------------------- PSEUDONYMIZATION CORE ---------------------------

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_pseudo_patient_id(real_id: Optional[str], salt: str) -> str:
    """
    Deterministic, short pseudonymous PatientID.
    Output: 'PX' + 6 hex chars, e.g., 'PX3F91AD'
    """
    base = real_id if (real_id and str(real_id).strip()) else "UNKNOWN"
    h = sha1_hex(salt + "::PID::" + str(base))
    return "PX" + h[:6].upper()


def pseudo_numeric_suffix(s: str, digits: int = 15) -> str:
    """
    Create a numeric suffix from a hex hash -> decimal with fixed number of digits.
    """
    val = int(s, 16) % (10 ** digits)
    return f"{val:0{digits}d}"


def make_pseudo_uid(pid: str, salt: str, kind: str) -> UID:
    """
    Deterministic numeric UID from pid, salt and kind ('study', 'series', 'for', etc.)
    """
    h = sha1_hex(salt + f"::UID::{kind}::" + pid)
    suf = pseudo_numeric_suffix(h, digits=12)  # compact but safe
    return UID(f"{PSEUDO_UID_ROOT}.{suf}")


# ------------------------------- PIXEL BLACKOUT ------------------------------

def _blackout_on_2d(arr2d: np.ndarray) -> np.ndarray:
    """
    Best-effort blackout of likely burned-in overlays using edge detection + box.
    Keeps dtype, sets masked region to 0.
    """
    # Normalize to uint8 for detection, keep original dtype for output
    a_min, a_max = arr2d.min(), arr2d.max()
    a_norm = ((arr2d - a_min) / (a_max - a_min + 1e-9) * 255.0).astype(np.uint8)

    edges = cv2.Canny(a_norm, 100, 200)
    if np.count_nonzero(edges) == 0:
        return arr2d

    # Expand/dilate edges and find bounding box
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    ys, xs = np.where(dil > 0)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    out = arr2d.copy()
    out[y0:y1+1, x0:x1+1] = 0  # blackout
    return out


def blackout_pixels_if_needed(ds: Dataset, profile: Dict[str, Any]) -> Dataset:
    """
    Blackout pixel data in a DICOM dataset when requested by the anonymization profile.
    This function inspects the provided DICOM dataset and an anonymization profile to
    optionally apply a pixel blackout operation. If the profile contains a truthy
    "PixelBlackout" value and the dataset contains pixel data, the function attempts
    to read the pixel array and apply a 2D blackout operation (via _blackout_on_2d)
    to either a single 2D image or each frame of a 3D image stack. The modified
    pixel data is written back to ds.PixelData using the original array dtype.
    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The DICOM dataset to inspect and possibly modify. This object is mutated
        in-place when blackout is applied.
    profile : Dict[str, Any]
        Anonymization profile dictionary. The key "PixelBlackout" controls whether
        pixel blackout should be performed (truthy => perform blackout).
    Returns
    -------
    pydicom.dataset.Dataset
        The original dataset, potentially modified. If no blackout is performed
        (because the profile disables it, PixelData is absent, reading the pixel
        array fails, or the array shape is unsupported), the original dataset is
        returned unchanged.
    Behavior and edge cases
    -----------------------
    - If profile.get("PixelBlackout", False) is falsy, the function returns ds
      unchanged.
    - If "PixelData" is not present in ds, the function returns ds unchanged.
    - The function attempts to access ds.pixel_array and will silently return ds
      unchanged if an exception occurs while reading pixels.
    - Supported pixel array shapes:
        - 2D arrays (H, W): _blackout_on_2d is applied directly.
        - 3D arrays (N, H, W): _blackout_on_2d is applied per-frame and frames are
          stacked back into a 3D array.
      Other dimensionalities are considered unsupported and result in ds being
      returned unchanged.
    - When writing back modified pixels, the function preserves the original numpy
      dtype by casting the blackout result to px.dtype and storing the bytes in
      ds.PixelData.
    - The function does not raise for the common failure modes described above;
      it prefers to leave the dataset unchanged on error.
    Dependencies
    ------------
    - Expects a helper function _blackout_on_2d(array: np.ndarray) -> np.ndarray that
      returns a same-shaped array with pixel blackout applied.
    - Uses ds.pixel_array which typically requires the pydicom package and an
      appropriate pixel handler (e.g., numpy) to be available.
    Example
    -------
    profile = {"PixelBlackout": True}
    ds = pydicom.dcmread("image.dcm")
    ds = blackout_pixels_if_needed(ds, profile)
    """
    if not profile.get("PixelBlackout", False):
        return ds
    if "PixelData" not in ds:
        return ds

    try:
        px = ds.pixel_array
    except Exception:
        return ds

    # Handle 2D or 3D stacks (H, W) or (N, H, W)
    if px.ndim == 2:
        out = _blackout_on_2d(px)
    elif px.ndim == 3:
        frames = []
        for i in range(px.shape[0]):
            frames.append(_blackout_on_2d(px[i]))
        out = np.stack(frames, axis=0)
    else:
        # Unsupported shape; skip
        return ds

    # Write back with the original dtype
    ds.PixelData = out.astype(px.dtype).tobytes()
    return ds


# ------------------------------ ANONYMIZATION -------------------------------

AUDIT_EXCLUDE = {"KeepPrivateTags", "PixelBlackout", "RetainStudyDate"}

def anonymize_dataset(ds: Dataset, profile: Dict[str, Any], salt: str) -> Tuple[Dataset, Dict[str, Tuple[Any, Any]], str]:
    """
    Anonymizes a DICOM dataset according to a given profile.
    This function applies anonymization rules to a DICOM dataset based on the provided profile,
    generating pseudonymous IDs and managing DICOM attributes according to specified actions.
    Args:
        ds (Dataset): The DICOM dataset to be anonymized
        profile (Dict[str, Any]): Anonymization profile containing field-action mappings.
            Possible actions include:
            - None: Remove the field
            - "NEWUID": Generate a new random UID
            - "PSEUDOUID": Generate a deterministic pseudo-UID
            - "PSEUDO": Use the pseudo PatientID
            - Any other value: Replace with the specified value
        salt (str): Salt string used for generating deterministic pseudo-identifiers
    Returns:
        Tuple[Dataset, Dict[str, Tuple[Any, Any]], str]: A tuple containing:
            - The anonymized DICOM dataset
            - An audit dictionary mapping field names to (old_value, new_value) pairs
            - The new pseudo PatientID
    Notes:
        - PatientID is always pseudonymized deterministically regardless of profile settings
        - Study dates can be optionally retained using the RetainStudyDate profile flag
        - Private tags can be kept using the KeepPrivateTags profile flag
        - Pixel data can be selectively blacked out based on profile settings
    """
    audit: Dict[str, Tuple[Any, Any]] = {}

    # Deterministic pseudonymous PatientID (regardless of profile PatientID default)
    real_pid = str(getattr(ds, "PatientID", "UNKNOWN"))
    new_pid = make_pseudo_patient_id(real_pid, salt)
    old_pid = getattr(ds, "PatientID", None)
    ds.PatientID = new_pid
    audit["PatientID"] = (old_pid, new_pid)

    # Apply profile fields
    for key, action in profile.items():
        if key in AUDIT_EXCLUDE:
            continue

        old_val = ds.get(key) if key in ds else None

        if action is None:
            if key in ds:
                del ds[key]
            new_val = "<removed>"
        elif action == "NEWUID":
            new_val = UID(generate_uid())
            ds.__setattr__(key, new_val)
        elif action == "PSEUDOUID":
            # Kind hint from key
            kind_map = {
                "StudyInstanceUID": "study",
                "SeriesInstanceUID": "series",
                "FrameOfReferenceUID": "for"
            }
            kind = kind_map.get(key, "uid")
            new_val = make_pseudo_uid(new_pid, salt, kind)
            ds.__setattr__(key, new_val)
        elif isinstance(action, str) and action.upper() == "PSEUDO":
            new_val = new_pid
            ds.__setattr__(key, new_val)
        else:
            new_val = action
            ds.__setattr__(key, new_val)

        audit[key] = (old_val, new_val)

    # Optional: retain StudyDate (if present) in Research profile
    if profile.get("RetainStudyDate", False):
        # no-op; already retained if present
        pass
    else:
        # if explicitly present in dataset, remove for strict mode
        if "StudyDate" in ds:
            audit["StudyDate"] = (ds.StudyDate, "<removed>")
            del ds.StudyDate

    # Remove private tags if requested
    if not profile.get("KeepPrivateTags", False):
        ds.remove_private_tags()

    # Pixel black-out (best-effort)
    ds = blackout_pixels_if_needed(ds, profile)

    return ds, audit, new_pid


# ------------------------------ FILE NAMING --------------------------------

def make_short_filename(ds: Dataset, pid: str, run_ts: str) -> str:
    """
    Short, timestamped name shared by the whole run.
    Pattern: {PID}_{YYYYMMDD_HHMMSS}_S{series}_I{instance}.dcm
    """
    series = int(getattr(ds, "SeriesNumber", 1) or 1)
    inst = int(getattr(ds, "InstanceNumber", 1) or 1)
    return f"{pid}_{run_ts}_S{series:03}_I{inst:04}.dcm"


# ------------------------------- WORKER -------------------------------------

def process_one(args: Tuple[str, str, Dict[str, Any], str, str]) -> Optional[Tuple[str, str, Dict[str, Tuple[Any, Any]], str]]:
    """
    Worker function to process a single DICOM file.
    
    This function reads a DICOM file, applies anonymization based on the provided profile,
    and writes the anonymized data to a specified output path. It returns a tuple containing
    the input path, output path, audit dictionary, and patient ID.

    Args:
        args (Tuple[str, str, Dict[str, Any], str, str]): A tuple containing:
            - in_path (str): The path to the input DICOM file.
            - out_root (str): The root directory for output files.
            - profile (Dict[str, Any]): Anonymization profile containing field-action mappings.
            - salt (str): Salt string used for generating deterministic identifiers.
            - run_ts (str): Timestamp for naming output files.

    Returns:
        Optional[Tuple[str, str, Dict[str, Tuple[Any, Any]], str]]:
            A tuple containing:
            - in_path (str): The original input file path.
            - out_path (str): The path to the anonymized output file.
            - audit (Dict[str, Tuple[Any, Any]]): A dictionary mapping field names to (old_value, new_value) pairs.
            - pid (str): The pseudonymous patient ID.
            Returns None if any error occurs during processing.
    """
    in_path, out_root, profile, salt, run_ts = args
    try:
        ds = dcmread(in_path)  # Read the DICOM dataset from the input path
    except InvalidDicomError:
        print(InvalidDicomError)
        return None  # Return None if the file is not a valid DICOM file
    except Exception:
        print("Other exception")
        return None  # Return None for any other exceptions during reading

    try:
        ds, audit, pid = anonymize_dataset(ds, profile, salt)  # Anonymize the dataset
    except Exception:
        print("Exception in anonymize_dataset")
        return None  # Return None if an error occurs during anonymization

    # Create output directory structure based on the pseudonymous patient ID
    out_dir = os.path.join(out_root, "DICOM", pid)
    os.makedirs(out_dir, exist_ok=True)  # Ensure the output directory exists

    new_name = make_short_filename(ds, pid, run_ts)  # Generate a short filename for the output
    out_path = os.path.join(out_dir, new_name)  # Construct the full output path

    try:
        ds.save_as(out_path, write_like_original=False)  # Save the anonymized dataset to the output path
    except Exception:
        print("Exception in save_as")
        return None  # Return None if an error occurs during saving

    return (in_path, out_path, audit, pid)  # Return the results of processing


# ------------------------------- PIPELINE -----------------------------------

def collect_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = []
        for r, _, fs in os.walk(input_path):
            for f in fs:
                files.append(os.path.join(r, f))
        return files
    else:
        return [input_path]


def run_pipeline(input_path: str, output_path: str, profile: Dict[str, Any], salt: str) -> str:
    """
    Run the anonymization pipeline for a directory of files and produce an audit log.
    Parameters
    ----------
    input_path : str
        Path to the input directory (or file) containing data to be anonymized. This
        path will be scanned by collect_files() to produce the set of files to process.
    profile : Dict[str, Any]
        Anonymization profile/configuration describing which fields/tags to transform
        and how. The exact expected keys/values depend on the implementation of
        process_one; typically this contains rules, mappings, or strategies used to
        transform identifying values.
    salt : str
        Salt string used for deterministic anonymization (e.g., hashing). Must be
        stable across runs if reproducible mapping is required.
    Returns
    -------
    str
        The path to the output root directory created for this run (format:
        "ANON_EXPORT_YYYYMMDD_HHMMSS"). The directory will contain anonymized files
        and a CSV audit log named "anonymization_log.csv".
    Behavior and side effects
    -------------------------
    - Creates an output directory named "ANON_EXPORT_<timestamp>" in the current
      working directory. If the directory already exists, it will be reused.
    - Uses collect_files(input_path) to enumerate files to process.
    - Builds a list of job tuples (file, out_root, profile, salt, run_ts) and processes
      them in parallel using multiprocessing.Pool(). The number of worker processes
      is the Pool default (usually the number of CPU cores).
    - Expects process_one(job) to return either a falsy value (to indicate failure/skip)
      or a tuple (in_path, out_path, audit, pid) where:
        - in_path: original input file path (str)
        - out_path: path to the anonymized output file (str)
        - audit: mapping of tag -> (old_value, new_value) describing substitutions
        - pid: process identifier or metadata (used for audit)
    - Writes a CSV audit log with columns:
        run_ts, input, output, pid, tag, old, new
      Each audit entry for each processed file is written as one CSV row. The file is
      opened with UTF-8 encoding and newline="" to ensure correct CSV formatting across platforms.
    - Prints a short summary to stdout indicating completion, output folder, and audit log path.
    Error handling and caveats
    --------------------------
    - Exceptions raised by os.makedirs, collect_files, process_one, or file I/O will
      propagate to the caller; consider wrapping this function to handle/retry errors
      or to perform cleanup.
    - Because multiprocessing.Pool() is used, objects passed to process_one must be
      picklable. Ensure profile and any closures or functions referenced by process_one
      are compatible with multiprocessing.
    - The CSV audit log is written after all parallel tasks complete; long-running
      pipelines may require more incremental logging to avoid losing progress on crash.
    - If deterministic anonymization is required across runs, ensure the provided salt
      and profile remain constant and that process_one implements deterministic behavior.
    Example
    -------
    >>> out_dir = run_pipeline("/path/to/input", profile=my_profile_dict, salt="somesalt")
    """
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = f"ANON_EXPORT_{run_ts}"
    out_root = os.path.join(output_path, out_root)
    os.makedirs(out_root, exist_ok=True)

    files = collect_files(input_path)
    jobs = [(f, out_root, profile, salt, run_ts) for f in files]

    # CSV audit log
    log_path = os.path.join(out_root, "anonymization_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_ts", "input", "output", "pid", "tag", "old", "new"])
        for j in jobs:
            in_path, out_path, audit, pid = process_one(j)
            for k, (old, new) in audit.items():
                writer.writerow([run_ts, in_path, out_path, pid, k, old, new])

    print("\n De-identification complete.")
    print(f"   Output folder : {out_root}")
    print(f"   Audit log     : {log_path}\n")
    return out_root


# --------------------------------- CLI / GUI --------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DICOM De-Identification (Pro)")
    p.add_argument("-i", "--input", default=os.path.abspath(os.getcwd()), required=False, help="DICOM file or directory")
    p.add_argument("-o", "--output", default=os.path.abspath(os.getcwd()), required=False,
                   help="DICOM out directory (default: current working directory)")
    p.add_argument("-p", "--profile-fname", default="GDPR-strict.json",
                   help="Anonymization profile filename")
    p.add_argument("--salt", default=DEFAULT_SALT, help="Site-specific secret salt for deterministic pseudonyms")
    p.add_argument("--gui", default=True, action="store_true", help="Launch a simple GUI to run the pipeline")
    return p.parse_args()


def _safe_load_profile(pfname: str) -> Optional[Dict[str, Any]]:
    try:
        with open(pfname, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if not isinstance(profile, dict):
            return None
        return profile
    except Exception:
        return None


def gui_main(initial_input: Optional[str], initial_output: Optional[str], initial_profile: Optional[str], initial_salt: str):
    try:
        root = tk.Tk()
        root.title("DICOM De-Identification (Pro) - GUI")
    except Exception:
        print("tkinter is not available; cannot launch GUI.")
        return

    # Input selection
    tk.Label(root, text="Input directory:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    input_var = tk.StringVar(value=initial_input or "")
    tk.Entry(root, textvariable=input_var, width=60).grid(row=0, column=1, padx=6, pady=4)
    def browse_input():
        # Let user pick either a file or a directory, starting from cwd
        d = filedialog.askdirectory(title="Select input folder", initialdir=os.getcwd())
        if d:
            input_var.set(d)
    tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2, padx=6)
    
    tk.Label(root, text="Output directory:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    output_var = tk.StringVar(value=initial_output or "")
    tk.Entry(root, textvariable=output_var, width=60).grid(row=1, column=1, padx=6, pady=4)
    def browse_output():
        d = filedialog.askdirectory(title="Select output directory", initialdir=os.getcwd())
        if d:
            output_var.set(d)
    tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2, padx=6)

    # Profile selection
    tk.Label(root, text="Profile JSON file:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    profile_var = tk.StringVar(value=initial_profile or "")
    tk.Entry(root, textvariable=profile_var, width=60).grid(row=2, column=1, padx=6, pady=4)
    def browse_profile():
        p = filedialog.askopenfilename(title="Select profile JSON", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if p:
            profile_var.set(p)
    tk.Button(root, text="Browse", command=browse_profile).grid(row=2, column=2, padx=6)

    # Salt
    tk.Label(root, text="Salt:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
    salt_var = tk.StringVar(value=initial_salt or DEFAULT_SALT)
    tk.Entry(root, textvariable=salt_var, width=60).grid(row=3, column=1, padx=6, pady=4)

    status_var = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status_var).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=6)

    # Start button handler
    def start():
        inp = input_var.get().strip()
        outdir = output_var.get().strip()
        pf = profile_var.get().strip()
        salt = salt_var.get().strip() or DEFAULT_SALT
        
        if not inp:
            messagebox.showerror("Error", "Please select an input directory.")
            return
        if not outdir:
            messagebox.showerror("Error", "Please select an output directory.")
            return
        if not pf:
            messagebox.showerror("Error", "Please select a profile JSON file.")
            return

        profile = _safe_load_profile(pf)
        if profile is None:
            messagebox.showerror("Error", f"Could not load profile from '{pf}'.")
            return

        # disable UI while running
        for w in root.winfo_children():
            w.configure(state="disabled")
        status_var.set("Running...")

        def runner():
            try:
                out = run_pipeline(inp, outdir, profile, salt)
                status_var.set(f"Completed. Output: {out}")
                messagebox.showinfo("Finished", f"De-identification complete.\nOutput folder: {out}")
            except Exception as e:
                status_var.set(f"Error: {e}")
                messagebox.showerror("Error", f"An error occurred:\n{e}")
            finally:
                for w in root.winfo_children():
                    w.configure(state="normal")

        t = threading.Thread(target=runner, daemon=True)
        t.start()

    tk.Button(root, text="Start", command=start, width=12).grid(row=4, column=2, pady=8)

    root.mainloop()


def main():
    args = parse_args()
    if args.gui:
        # Allow optional CLI-provided defaults for the GUI
        inp = args.input
        out = args.output
        pfname = args.profile_fname
        gui_main(inp, out, pfname, args.salt)
        return

    # CLI mode: require input
    if not args.input:
        print("Error: --input is required in CLI mode. Use --gui to launch the graphical interface.")
        return

    profile = _safe_load_profile(args.profile_fname)
    if profile is None:
        print(f"Profile file '{args.profile_fname}' did not contain a valid JSON object or could not be read.")
        return

    run_pipeline(args.input, args.output, profile, args.salt)


if __name__ == "__main__":
    main()
