#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from pydicom.datadict import DicomDictionary

import cv2
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

NAME_TO_KEYWORD: Dict[str, str] = {}
for _tag, entry in DicomDictionary.items():
    # entry = (VR, VM, name, is_retired, keyword)
    name = entry[2]
    keyword = entry[4]
    if name and keyword:
        NAME_TO_KEYWORD[name] = keyword

# =========================== USER-EDITABLE CONFIG ============================

DEFAULT_SALT = "CHANGE-ME-SALT-RHYTHM-UOC-2025"
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
    Apply pixel blackout if profile["PixelBlackout"] is True.
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
        out = np.stack([_blackout_on_2d(px[i]) for i in range(px.shape[0])], axis=0)
    else:
        return ds

    ds.PixelData = out.astype(px.dtype).tobytes()
    return ds


# ------------------------------ ANONYMIZATION -------------------------------

# Keys in the JSON that are NOT DICOM attributes but global flags
PROFILE_FLAGS = {"KeepPrivateTags", "PixelBlackout", "RetainStudyDate", "Private attributes"}


def _clean_attr_name(name: str) -> str:
    """
    Remove zero-width spaces and collapse multiple spaces to a single space.
    Helps with things like 'GPS Latitude\u200b Ref' or 'GPS Latitude  Ref'.
    """
    if not isinstance(name, str):
        name = str(name)
    # remove zero-width spaces and similar
    name = name.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    # collapse whitespace
    parts = name.split()
    return " ".join(parts)

def keyword_for_name(attr_name: str) -> str | None:
    """
    Map DICOM Attribute Name (e.g. "Patient's Name")
    to pydicom keyword (e.g. "PatientName").
    Returns None if not found.
    """
    return NAME_TO_KEYWORD.get(attr_name)


def anonymize_dataset(ds: Dataset, profile: Dict[str, Any], salt: str) -> Tuple[Dataset, Dict[str, Tuple[Any, Any]], str]:
    """
    Anonymize a DICOM dataset according to an explicit JSON profile.

    The profile must list EVERY attribute we care about with one of:
      - null        -> delete tag (if present)
      - "KEEP"      -> leave tag unchanged
      - "PSEUDO"    -> set to deterministic pseudo PatientID
      - "NEWUID"    -> new random UID
      - "PSEUDOUID" -> deterministic UID from pseudo PatientID
      - any string  -> literal replacement value

    Keys in the JSON are DICOM Attribute Names (e.g. "Patient's Name"),
    which are mapped to pydicom keywords via keyword_for_name().
    """
    audit: Dict[str, Tuple[Any, Any]] = {}

    # Pseudonymous PatientID used for PSEUDO / PSEUDOUID
    real_pid = str(getattr(ds, "PatientID", "UNKNOWN"))
    pseudo_pid = make_pseudo_patient_id(real_pid, salt)

    # Apply each rule in the JSON profile
    for attr_name_raw, action in profile.items():
        # Skip global flags, they are handled separately
        if attr_name_raw in PROFILE_FLAGS:
            continue

        attr_name = _clean_attr_name(attr_name_raw)
        kw = keyword_for_name(attr_name)  # map "Patient's Name" -> "PatientName"
        if not kw:
            # Unknown or non-standard attribute name; skip safely
            continue

        old_val = None
        try:
            old_val = ds.get(kw)
        except:
            pass
            #print (f"Warning: could not read attribute '{attr_name}' ({kw});")
        new_val = old_val  # default if KEEP or unknown action

        # Interpret action
        if action is None:
            # null -> delete
            if kw in ds:
                del ds[kw]
            new_val = "<removed>"
        elif isinstance(action, str):
            code = action.strip().upper()
            if code == "KEEP":
                # Do not modify, just record
                new_val = old_val
            elif code == "PSEUDO":
                new_val = pseudo_pid
                ds.__setattr__(kw, new_val)
            elif code == "NEWUID":
                new_val = UID(generate_uid())
                ds.__setattr__(kw, new_val)
            elif code == "PSEUDOUID":
                kind_map = {
                    "StudyInstanceUID": "study",
                    "SeriesInstanceUID": "series",
                    "FrameOfReferenceUID": "for"
                }
                kind = kind_map.get(kw, "uid")
                new_val = make_pseudo_uid(pseudo_pid, salt, kind)
                ds.__setattr__(kw, new_val)
            else:
                # Literal replacement string
                new_val = action
                ds.__setattr__(kw, new_val)
        else:
            # Unknown type -> no change
            new_val = old_val

        audit[attr_name] = (old_val, new_val)

    # Handle private tags according to profile
    if not profile.get("KeepPrivateTags", False):
        ds.remove_private_tags()

    # Pixel blackout
    ds = blackout_pixels_if_needed(ds, profile)

    # Return pseudo_pid so it can be used in filenames if desired
    return ds, audit, pseudo_pid


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
    """
    in_path, out_root, profile, salt, run_ts = args
    try:
        ds = dcmread(in_path)
    except InvalidDicomError:
        print(f"Skipping non-DICOM: {in_path}")
        return None
    except Exception as e:
        print(f"Error reading {in_path}: {e}")
        return None

    try:
        ds, audit, pid = anonymize_dataset(ds, profile, salt)
    except Exception as e:
        import traceback
        print(f"Error anonymizing {in_path}: {e}")
        traceback.print_exc()
        return None

    out_dir = os.path.join(out_root, "DICOM", pid)
    os.makedirs(out_dir, exist_ok=True)

    new_name = make_short_filename(ds, pid, run_ts)
    out_path = os.path.join(out_dir, new_name)

    try:
        ds.save_as(out_path, write_like_original=False)
    except Exception as e:
        print(f"Error saving {out_path}: {e}")
        return None

    return (in_path, out_path, audit, pid)


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
    Uses the explicit JSON profile (no default delete).
    """
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(output_path, f"ANON_EXPORT_{run_ts}")
    os.makedirs(out_root, exist_ok=True)

    files = collect_files(input_path)
    jobs = [(f, out_root, profile, salt, run_ts) for f in files]

    log_path = os.path.join(out_root, "anonymization_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_ts", "input", "output", "pid", "tag", "old", "new"])

        # NOTE: currently serial; you can replace with mp.Pool if you want parallel
        for j in jobs:
            result = process_one(j)
            if not result:
                continue
            in_path, out_path, audit, pid = result
            for k, (old, new) in audit.items():
                writer.writerow([run_ts, in_path, out_path, pid, k, old, new])

    print("\nDe-identification complete.")
    print(f"   Output folder : {out_root}")
    print(f"   Audit log     : {log_path}\n")
    return out_root


# --------------------------------- CLI / GUI --------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DICOM De-Identification (Pro)")
    p.add_argument("-i", "--input", default=os.path.abspath(os.getcwd()), required=False,
                   help="DICOM file or directory")
    p.add_argument("-o", "--output", default=os.path.abspath(os.getcwd()), required=False,
                   help="Output directory (default: current working directory)")
    p.add_argument("-p", "--profile-fname", default="GDPR-strict_explicit.json",
                   help="Anonymization profile filename (explicit JSON)")
    p.add_argument("--salt", default=DEFAULT_SALT,
                   help="Site-specific secret salt for deterministic pseudonyms")
    p.add_argument("--gui", default=True, action="store_true",
                   help="Launch a simple GUI to run the pipeline")
    return p.parse_args()


def _safe_load_profile(pfname: str) -> Optional[Dict[str, Any]]:
    try:
        with open(pfname, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if not isinstance(profile, dict):
            return None
        return profile
    except Exception as e:
        print(f"Error loading profile {pfname}: {e}")
        return None


def gui_main(initial_input: Optional[str], initial_output: Optional[str],
             initial_profile: Optional[str], initial_salt: str):
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
        p = filedialog.askopenfilename(title="Select profile JSON",
                                       filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if p:
            profile_var.set(p)

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
            try:
                w.configure(state="disabled")
            except Exception:
                pass
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
                    try:
                        w.configure(state="normal")
                    except Exception:
                        pass

        t = threading.Thread(target=runner, daemon=True)
        t.start()

    tk.Button(root, text="Start", command=start, width=12).grid(row=4, column=2, pady=8)
    root.mainloop()


def main():
    args = parse_args()
    if args.gui:
        gui_main(args.input, args.output, args.profile_fname, args.salt)
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
