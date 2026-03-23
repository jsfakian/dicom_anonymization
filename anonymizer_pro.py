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
- Parallel processing (multiprocessing.Pool)  [NOTE: current run loop is sequential; easy to enable Pool later]
- Short file names with run-wide timestamp
- CSV audit log (input, output, tag, old, new, PID)

SECURITY FIX (this version):
- Excludes PixelData and other binary/large payload fields from audit logging
  so image bytes can never end up in anonymization_log.csv.

Usage examples:
  python anonymizer_pro.py -i /path/to/DICOM -p GDPR-strict.json
  python anonymizer_pro.py -i study.dcm -p research-pseudonymized.json --salt "my-secret"

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
PSEUDO_UID_ROOT = "1.3.6.1.4.1.37476.9000.207.999"

# ========================= END USER-EDITABLE CONFIG ==========================


# ============================== AUDIT CONTROLS ===============================

# Profile keys that are not real DICOM attributes
AUDIT_EXCLUDE = {"KeepPrivateTags", "PixelBlackout", "RetainStudyDate"}

# Binary / large payload fields that must NEVER be written to the audit CSV.
# (Keywords are pydicom attribute keywords.)
EXCLUDE_FROM_AUDIT = {
    # Pixel / frame data
    "PixelData",
    "FloatPixelData",
    "DoubleFloatPixelData",
    "PixelDataProviderURL",

    # Overlays / curves / waveforms / audio / docs (can be huge and sensitive)
    "OverlayData",
    "CurveData",
    "WaveformData",
    "AudioSampleData",
    "EncapsulatedDocument",
    "SpectroscopyData",

    # Icons/thumbnails can also contain pixels (often small, but still pixels)
    "IconImageSequence",
}

# If a value is very large (e.g., huge text / blob), avoid logging it.
MAX_AUDIT_VALUE_LEN = 512

# ============================ END AUDIT CONTROLS ==============================


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
    a_min, a_max = arr2d.min(), arr2d.max()
    a_norm = ((arr2d - a_min) / (a_max - a_min + 1e-9) * 255.0).astype(np.uint8)

    edges = cv2.Canny(a_norm, 100, 200)
    if np.count_nonzero(edges) == 0:
        return arr2d

    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    ys, xs = np.where(dil > 0)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    out = arr2d.copy()
    out[y0:y1+1, x0:x1+1] = 0
    return out


def blackout_pixels_if_needed(ds: Dataset, profile: Dict[str, Any]) -> Dataset:
    """
    Blackout pixel data when requested by the anonymization profile.
    """
    if not profile.get("PixelBlackout", False):
        return ds
    if "PixelData" not in ds:
        return ds

    try:
        px = ds.pixel_array
    except Exception:
        return ds

    if px.ndim == 2:
        out = _blackout_on_2d(px)
    elif px.ndim == 3:
        frames = []
        for i in range(px.shape[0]):
            frames.append(_blackout_on_2d(px[i]))
        out = np.stack(frames, axis=0)
    else:
        return ds

    ds.PixelData = out.astype(px.dtype).tobytes()
    return ds


# ------------------------------ AUDIT HELPERS --------------------------------

def _audit_safe_value(keyword: str, value: Any) -> Any:
    """
    Return a value safe for CSV logging.
    - Never log binary/pixel payloads
    - Truncate very large strings
    - Avoid dumping huge lists/bytes
    """
    # Block known binary fields by keyword
    if keyword in EXCLUDE_FROM_AUDIT:
        return "<binary omitted>"

    # Block raw bytes/bytearray by type as a second layer of protection
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary omitted>"

    # Some pydicom values can be very large arrays/lists; keep short summary
    if isinstance(value, (list, tuple)) and len(value) > 20:
        return f"<list len={len(value)} omitted>"

    # Truncate long strings
    if isinstance(value, str) and len(value) > MAX_AUDIT_VALUE_LEN:
        return value[:MAX_AUDIT_VALUE_LEN] + "...<truncated>"

    # Fallback: stringify safely if it’s a complex object
    try:
        s = str(value)
        if len(s) > MAX_AUDIT_VALUE_LEN:
            return s[:MAX_AUDIT_VALUE_LEN] + "...<truncated>"
        return value
    except Exception:
        return "<unprintable omitted>"


# ------------------------------ ANONYMIZATION --------------------------------

def anonymize_dataset(ds: Dataset, profile: Dict[str, Any], salt: str) -> Tuple[Dataset, Dict[str, Tuple[Any, Any]], str]:
    """
    Anonymizes a DICOM dataset according to a given profile.
    Returns: (ds, audit, new_pid)
    """
    audit: Dict[str, Tuple[Any, Any]] = {}

    # Deterministic pseudonymous PatientID (regardless of profile PatientID default)
    real_pid = str(getattr(ds, "PatientID", "UNKNOWN"))
    new_pid = make_pseudo_patient_id(real_pid, salt)

    old_pid = getattr(ds, "PatientID", None)
    ds.PatientID = new_pid
    audit["PatientID"] = (_audit_safe_value("PatientID", old_pid), _audit_safe_value("PatientID", new_pid))

    # Apply profile fields
    for key, action in profile.items():
        if key in AUDIT_EXCLUDE:
            continue

        # In this script, JSON keys are expected to be pydicom keywords (e.g., StudyInstanceUID).
        keyword = key

        # Capture old value safely (avoid PixelData, etc.)
        old_val = ds.get(keyword) if keyword in ds else None
        old_val_safe = _audit_safe_value(keyword, old_val)

        if action is None:
            if keyword in ds:
                del ds[keyword]
            new_val = "<removed>"
            new_val_safe = new_val
        elif action == "NEWUID":
            new_val = UID(generate_uid())
            ds.__setattr__(keyword, new_val)
            new_val_safe = _audit_safe_value(keyword, new_val)
        elif action == "PSEUDOUID":
            kind_map = {
                "StudyInstanceUID": "study",
                "SeriesInstanceUID": "series",
                "FrameOfReferenceUID": "for"
            }
            kind = kind_map.get(keyword, "uid")
            new_val = make_pseudo_uid(new_pid, salt, kind)
            ds.__setattr__(keyword, new_val)
            new_val_safe = _audit_safe_value(keyword, new_val)
        elif isinstance(action, str) and action.upper() == "PSEUDO":
            new_val = new_pid
            ds.__setattr__(keyword, new_val)
            new_val_safe = _audit_safe_value(keyword, new_val)
        else:
            new_val = action
            ds.__setattr__(keyword, new_val)
            new_val_safe = _audit_safe_value(keyword, new_val)

        # IMPORTANT: Never leak binary payloads to audit
        audit[keyword] = (old_val_safe, new_val_safe)

    # Optional: retain StudyDate (if present) in Research profile
    if not profile.get("RetainStudyDate", False):
        if "StudyDate" in ds:
            audit["StudyDate"] = (_audit_safe_value("StudyDate", ds.StudyDate), "<removed>")
            del ds.StudyDate

    # Remove private tags if requested
    if not profile.get("KeepPrivateTags", False):
        ds.remove_private_tags()

    # Pixel black-out (best-effort)
    ds = blackout_pixels_if_needed(ds, profile)

    return ds, audit, new_pid


# ------------------------------ FILE NAMING ----------------------------------

def make_short_filename(ds: Dataset, pid: str, run_ts: str) -> str:
    """
    Short, timestamped name shared by the whole run.
    Pattern: {PID}_{YYYYMMDD_HHMMSS}_S{series}_I{instance}.dcm
    """
    series = int(getattr(ds, "SeriesNumber", 1) or 1)
    inst = int(getattr(ds, "InstanceNumber", 1) or 1)
    return f"{pid}_{run_ts}_S{series:03}_I{inst:04}.dcm"


# ------------------------------- WORKER --------------------------------------

def process_one(args: Tuple[str, str, Dict[str, Any], str, str]) -> Optional[Tuple[str, str, Dict[str, Tuple[Any, Any]], str]]:
    in_path, out_root, profile, salt, run_ts = args
    try:
        ds = dcmread(in_path)
    except InvalidDicomError:
        return None
    except Exception:
        return None

    try:
        ds, audit, pid = anonymize_dataset(ds, profile, salt)
    except Exception:
        return None

    out_dir = os.path.join(out_root, "DICOM", pid)
    os.makedirs(out_dir, exist_ok=True)

    new_name = make_short_filename(ds, pid, run_ts)
    out_path = os.path.join(out_dir, new_name)

    try:
        ds.save_as(out_path, write_like_original=False)
    except Exception:
        return None

    return (in_path, out_path, audit, pid)


# ------------------------------- PIPELINE ------------------------------------

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
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = f"ANON_EXPORT_{run_ts}"
    out_root = os.path.join(output_path, out_root)
    os.makedirs(out_root, exist_ok=True)

    files = collect_files(input_path)
    jobs = [(f, out_root, profile, salt, run_ts) for f in files]

    log_path = os.path.join(out_root, "anonymization_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_ts", "input", "output", "pid", "tag", "old", "new"])

        # NOTE: This loop is sequential in the original script; keeping it stable.
        # You can swap to mp.Pool easily later if desired.
        for j in jobs:
            res = process_one(j)
            if not res:
                continue
            in_path, out_path, audit, pid = res
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

    tk.Label(root, text="Input directory:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    input_var = tk.StringVar(value=initial_input or "")
    tk.Entry(root, textvariable=input_var, width=60).grid(row=0, column=1, padx=6, pady=4)

    def browse_input():
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

    tk.Label(root, text="Profile JSON file:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    profile_var = tk.StringVar(value=initial_profile or "")
    tk.Entry(root, textvariable=profile_var, width=60).grid(row=2, column=1, padx=6, pady=4)

    def browse_profile():
        pth = filedialog.askopenfilename(title="Select profile JSON", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if pth:
            profile_var.set(pth)
    tk.Button(root, text="Browse", command=browse_profile).grid(row=2, column=2, padx=6)

    tk.Label(root, text="Salt:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
    salt_var = tk.StringVar(value=initial_salt or DEFAULT_SALT)
    tk.Entry(root, textvariable=salt_var, width=60).grid(row=3, column=1, padx=6, pady=4)

    status_var = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status_var).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=6)

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
        inp = args.input
        out = args.output
        pfname = args.profile_fname
        gui_main(inp, out, pfname, args.salt)
        return

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
