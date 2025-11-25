#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import hashlib
from typing import Dict, Any, Optional, List, Tuple

from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import UID
from pydicom.datadict import DicomDictionary
from pydicom.sequence import Sequence

# ====================== SHARED HELPERS (mirror anonymizer) ===================

NAME_TO_KEYWORD: Dict[str, str] = {}
for _tag, entry in DicomDictionary.items():
    # entry = (VR, VM, name, is_retired, keyword)
    name = entry[2]
    keyword = entry[4]
    if name and keyword:
        NAME_TO_KEYWORD[name] = keyword

PROFILE_FLAGS = {"KeepPrivateTags", "PixelBlackout", "RetainStudyDate", "Private attributes"}

DEFAULT_SALT = "CHANGE-ME-SALT-RHYTHM-UOC-2025"
PSEUDO_UID_ROOT = "1.2.826.0.1.3680043.10.54321"


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
    val = int(s, 16) % (10 ** digits)
    return f"{val:0{digits}d}"


def make_pseudo_uid(pid: str, salt: str, kind: str) -> UID:
    """
    Deterministic numeric UID from pid, salt and kind ('study', 'series', 'for', etc.)
    """
    h = sha1_hex(salt + f"::UID::{kind}::" + pid)
    suf = pseudo_numeric_suffix(h, digits=12)
    return UID(f"{PSEUDO_UID_ROOT}.{suf}")


def _clean_attr_name(name: str) -> str:
    """
    Remove zero-width + NBSP and collapse multiple spaces.
    """
    if not isinstance(name, str):
        name = str(name)
    for ch in ["\u200b", "\u200c", "\u200d", "\xa0"]:
        name = name.replace(ch, "")
    parts = name.split()
    return " ".join(parts)


def has_private_tags(ds: Dataset) -> bool:
    for de in ds.iterall():
        if de.tag.is_private:
            return True
    return False


# ============================= VALIDATION CORE ===============================

class ValidationError:
    def __init__(self, tag_name: str, msg: str, old: Any, new: Any, expected: Any):
        self.tag_name = tag_name
        self.msg = msg
        self.old = old
        self.new = new
        self.expected = expected

    def __str__(self) -> str:
        return (f"[{self.tag_name}] {self.msg}\n"
                f"  original : {self.old!r}\n"
                f"  anonymized: {self.new!r}\n"
                f"  expected : {self.expected!r}")


def build_normalized_profile(profile: Dict[str, Any]) -> Dict[str, Tuple[str, Any]]:
    """
    Return mapping: cleaned_attr_name -> (original_key, action).
    This lets us:
      - match JSON rules regardless of whitespace/hidden chars
      - still know the action (even if it's JSON null -> None).
    """
    norm: Dict[str, Tuple[str, Any]] = {}
    for k, v in profile.items():
        if k in PROFILE_FLAGS:
            continue
        cleaned = _clean_attr_name(k)
        if cleaned in norm:
            # Later entries overwrite earlier ones; that's fine.
            pass
        norm[cleaned] = (k, v)
    return norm


def validate_against_profile(
    orig: Dataset,
    anon: Dataset,
    profile: Dict[str, Any],
    salt: str
) -> List[ValidationError]:
    errors: List[ValidationError] = []

    # Build normalized profile mapping for name lookup
    norm_profile = build_normalized_profile(profile)

    # Compute pseudo PID as anonymizer does
    real_pid_val = getattr(orig, "PatientID", None)
    real_pid = str(real_pid_val) if real_pid_val is not None else "UNKNOWN"
    pseudo_pid = make_pseudo_patient_id(real_pid, salt)

    # 1) For every attribute that actually existed in the original:
    #    check what happened to it in the anonymized dataset.
    for de in orig.iterall():
        # We'll skip private tags here; we handle them via KeepPrivateTags later.
        if de.tag.is_private:
            continue

        attr_name_raw = de.name            # pydicom's Attribute Name
        attr_name = _clean_attr_name(attr_name_raw)
        rule_entry = norm_profile.get(attr_name, None)
        old_val = de.value
        new_de = anon.get(de.tag)
        new_val = new_de.value if new_de is not None else None

        if rule_entry is None:
        #    # No explicit rule in profile for this attribute.
        #    # You can choose how strict you want to be here.
        #    # For now: if value changed, flag it; if same, accept.
        #    if isinstance(old_val, Sequence) and isinstance(new_val, Sequence):
        #        # shallow comparison: length only
        #        if len(old_val) != len(new_val):
        #            errors.append(
        #                ValidationError(
        #                    attr_name,
        #                    "No profile rule for attribute, but sequence length changed",
        #                    old_val,
        #                    new_val,
        #                    old_val,
        #                )
        #            )
        #    else:
        #        if old_val != new_val:
        #            print(f"Validating {attr_name} tag {de.tag} old value: {old_val}, new value: {new_val}.")
        #            errors.append(
        #                ValidationError(
        #                    attr_name,
        #                    "No profile rule for attribute, but value changed",
        #                    old_val,
        #                    new_val,
        #                    old_val,
        #                )
        #            )
            continue

        original_json_key, action = rule_entry

        # Interpret profile rule for this actual attribute
        if action is None:
            # null -> delete if present
            if new_de is not None:
                errors.append(
                    ValidationError(
                        attr_name,
                        f"Profile ({original_json_key}) says 'null' (delete), but tag is still present",
                        old_val,
                        new_val,
                        "<removed>",
                    )
                )
            continue

        if isinstance(action, str):
            code = action.strip().upper()

            if code == "KEEP":
                # Must stay the same (or both missing)
                if isinstance(old_val, Sequence) and isinstance(new_val, Sequence):
                    if len(old_val) != len(new_val):
                        errors.append(
                            ValidationError(
                                attr_name,
                                "Expected 'KEEP', but sequence length changed",
                                old_val,
                                new_val,
                                old_val,
                            )
                        )
                else:
                    if old_val != new_val:
                        errors.append(
                            ValidationError(
                                attr_name,
                                "Expected 'KEEP', but value changed",
                                old_val,
                                new_val,
                                old_val,
                            )
                        )
                continue

            if code == "PSEUDO":
                expected_val = pseudo_pid
                if new_val != expected_val:
                    errors.append(
                        ValidationError(
                            attr_name,
                            "Expected 'PSEUDO' (pseudo PatientID), but anonymized value does not match",
                            old_val,
                            new_val,
                            expected_val,
                        )
                    )
                continue

            if code == "NEWUID":
                # We can't know exact UID, but we know:
                #  - new must exist
                #  - must be valid UID
                #  - must differ from original if original existed
                if new_val is None:
                    errors.append(
                        ValidationError(
                            attr_name,
                            "Expected 'NEWUID', but anonymized tag is missing",
                            old_val,
                            new_val,
                            "new random UID",
                        )
                    )
                    continue
                try:
                    uid = UID(str(new_val))
                    if not uid.is_valid:
                        errors.append(
                            ValidationError(
                                attr_name,
                                "Expected 'NEWUID', but produced value is not a valid UID",
                                old_val,
                                new_val,
                                "valid UID",
                            )
                        )
                except Exception:
                    errors.append(
                        ValidationError(
                            attr_name,
                            "Expected 'NEWUID', but produced value cannot be parsed as UID",
                            old_val,
                            new_val,
                            "valid UID",
                        )
                    )
                if old_val is not None and str(new_val) == str(old_val):
                    errors.append(
                        ValidationError(
                            attr_name,
                            "Expected 'NEWUID', but UID did not change",
                            old_val,
                            new_val,
                            "different UID",
                        )
                    )
                continue

            if code == "PSEUDOUID":
                # Map by keyword to determine kind
                kw = de.keyword or ""
                kind_map = {
                    "StudyInstanceUID": "study",
                    "SeriesInstanceUID": "series",
                    "FrameOfReferenceUID": "for",
                }
                kind = kind_map.get(kw, "uid")
                expected_val = make_pseudo_uid(pseudo_pid, salt, kind)
                if new_val is None or str(new_val) != str(expected_val):
                    errors.append(
                        ValidationError(
                            attr_name,
                            "Expected 'PSEUDOUID', but anonymized UID does not match deterministic value",
                            old_val,
                            new_val,
                            expected_val,
                        )
                    )
                continue

            # Literal replacement string
            expected_val = action
            if new_val != expected_val:
                errors.append(
                    ValidationError(
                        attr_name,
                        "Expected literal replacement from profile, but value differs",
                        old_val,
                        new_val,
                        expected_val,
                    )
                )
            continue

        # If we reach here, the action type is something unexpected
        errors.append(
            ValidationError(
                attr_name,
                f"Unsupported action type {type(action).__name__} for profile key '{original_json_key}'",
                old_val,
                new_val,
                None,
            )
        )

    # 2) Private tags handling (global flag)
    keep_private = bool(profile.get("KeepPrivateTags", False))
    if not keep_private:
        if has_private_tags(anon):
            errors.append(
                ValidationError(
                    "PrivateTags",
                    "KeepPrivateTags is False/missing, but anonymized dataset still has private tags",
                    old="(original private tags not enumerated)",
                    new="(private tags still present)",
                    expected="No private tags",
                )
            )

    # 3) Optionally: you could check for tags that appear in anon but not in orig
    # If they also have a profile rule, you might want to validate those too.
    # For now we ignore that to keep the logic focused.

    return errors


# ============================== CLI / MAIN ==================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate DICOM anonymization against JSON profile "
                    "by inspecting all attributes present in the original dataset."
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Original (non-anonymized) DICOM file",
    )
    p.add_argument(
        "-a",
        "--anon",
        required=True,
        help="Anonymized DICOM file to validate",
    )
    p.add_argument(
        "-p",
        "--profile",
        required=True,
        help="JSON profile file used for anonymization",
    )
    p.add_argument(
        "--salt",
        default=DEFAULT_SALT,
        help="Salt used for pseudonymization (must match anonymizer).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Original DICOM not found: {args.input}")
        return

    if not os.path.isfile(args.anon):
        print(f"ERROR: Anonymized DICOM not found: {args.anon}")
        return

    if not os.path.isfile(args.profile):
        print(f"ERROR: Profile JSON not found: {args.profile}")
        return

    try:
        orig = dcmread(args.input)
    except Exception as e:
        print(f"ERROR: Failed to read original DICOM: {e}")
        return

    try:
        anon = dcmread(args.anon)
    except Exception as e:
        print(f"ERROR: Failed to read anonymized DICOM: {e}")
        return

    try:
        with open(args.profile, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read profile JSON: {e}")
        return

    if not isinstance(profile, dict):
        print("ERROR: Profile JSON is not an object at top level.")
        return

    errors = validate_against_profile(orig, anon, profile, args.salt)

    if not errors:
        print("✅ Anonymized DICOM matches the profile w.r.t. all attributes present in the original.")
    else:
        print("❌ Anonymized DICOM does NOT fully match the profile. Details:\n")
        for e in errors:
            print(str(e))
            print("-" * 72)
        # For CI you might want:
        # import sys
        # sys.exit(1)


if __name__ == "__main__":
    main()
# ============================== END OF FILE ================================
