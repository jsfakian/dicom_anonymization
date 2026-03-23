# Technical Description of the DICOM Anonymization Tool

## Overview

A Python CLI tool for anonymizing DICOM medical imaging files. It removes or transforms patient-identifiable metadata while preserving the technical data needed for imaging research and quality assurance. Original files are never modified — anonymized output is written to a separate timestamped directory.

### How It Works

DICOM files contain two types of data:

- **Pixel data** — the medical images themselves (preserved)
- **Metadata (DICOM attributes)** — patient, exam, scanner, and acquisition details (selectively removed or pseudonymized)

The tool reads each file, applies a JSON-based anonymization profile to the metadata, and writes the result to the output directory.

### Flexible JSON-based anonymization profiles

Enables explicit control of every DICOM tag (KEEP / null / PSEUDO / PSEUDOUID), allowing user-custom profiles. Allows selective retention of any research or dosimetry-relevant metadata

A strict, GDPR-oriented profile is provided as a ready-to-use template, ensuring that all direct and indirect identifiers are appropriately removed or pseudonymized while maintaining compliance with data protection requirements. Users are encouraged to utilize this sample profile as the baseline for their anonymization tasks. The supplied profile is designed to be sufficiently conservative for safe data sharing while still preserving essential structural information, such as UID relationships, required for dataset integrity, and can be easily adapted by modifying selected fields in the JSON configuration.

### What Gets Removed

Direct patient identifiers are deleted entirely, including:

- Patient names, IDs, birth dates, and addresses
- Contact information and physician identifiers
- Accession numbers

### Pseudonymization

Some identifiers cannot simply be deleted because they maintain the internal structure of the dataset — images must stay grouped by study, series must stay ordered, and cross-image references must remain valid.

For these, the tool generates deterministic pseudonymous values using a hashing algorithm combined with a user-supplied secret **salt**. This means:

- Dataset relationships remain intact
- Pseudonyms are consistent across multiple runs with the same salt
- Original values cannot be reconstructed

This applies to both free-text identifiers and DICOM UIDs (Study Instance UID, Series Instance UID, SOP Instance UID, Frame of Reference UID).

### Technical Metadata Preserved

Attributes required for imaging research are kept, including:

- Scanner acquisition parameters and image geometry
- Slice thickness and spacing
- Tube voltage and current
- Radiation dose indicators (CTDIvol, DLP)

### Pipeline Architecture

| Step | Component | Description |
| ---- | --------- | ----------- |
| 1 | Input discovery | Finds DICOM files to process |
| 2 | DICOM parsing | Reads dataset and metadata |
| 3 | Policy engine | Applies rules from the JSON profile |
| 4 | UID pseudonymization | Generates deterministic pseudonymous UIDs |
| 5 | Sequence traversal | Recursively handles nested DICOM structures |
| 6 | Pixel processing *(optional)* | Masks burned-in text annotations |
| 7 | Output writer | Saves anonymized files |
| 8 | Audit logger | Records all transformations |

### JSON Profile Format

Profiles map DICOM attribute names to anonymization actions:

```json
{
  "Patient's Name": null,
  "Patient ID": null,
  "Study Instance UID": "PSEUDOUID",
  "Series Instance UID": "PSEUDOUID",
  "Slice Thickness": "KEEP"
}
```

| Action | Effect |
| ------ | ------ |
| `KEEP` | Attribute value is preserved unchanged |
| `null` | Attribute is removed from the file |
| `"PSEUDO"` | Replaced with a deterministic pseudonym (based on salt) |
| `"PSEUDOUID"` | Replaced with a deterministic pseudonymous UID (based on salt) |
| `"NEWUID"` | Replaced with a newly generated random UID |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **CLI Interface** | Run from terminal with configurable options |
| **Deterministic Pseudonymization** | Uses a user-defined SALT to generate repeatable, consistent pseudonyms |
| **Profile-Based Configuration** | Pre-built profiles for GDPR-strict and research-grade anonymization |
| **UID Management** | Generates new UIDs or pseudonymous UIDs for StudyInstanceUID, SeriesInstanceUID, etc. |
| **Pixel Blackout** | Detects and masks burned-in text overlays on DICOM images using edge detection |
| **Batch Processing** | Works on single DICOM files or entire folder hierarchies |
| **Audit Trail** | CSV log tracks all transformations (old values → new values, per tag) |
| **Safe Output** | Never modifies source data; creates timestamped `ANON_EXPORT_*` folder |

---

## Installation

See [INSTALL.md](INSTALL.md) for detailed setup instructions on Windows, macOS, and Linux.

**Quick Setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate           # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project Files

Here's what each file does, so you know which tool to use:

| File | Purpose |
|------|---------|
| **`anonymizer_pro.py`** | ⭐ **MAIN TOOL** — Use this to anonymize DICOM files |
| `GDPR-strict.json` | Anonymization profile for maximum de-identification (legal/public use) |
| `research-pseudonymized.json` | Anonymization profile for research use (keeps clinical dates) |
| `requirements.txt` | List of Python packages needed (installed during setup) |
| `README.md` | This file — overview and documentation |
| `INSTALL.md` | Step-by-step installation guide |
| `dump_dicom_metadata.py` | **Utility** — View all tags in a DICOM file (debugging) |
| `compare_dicom_images.py` | **Utility** — Compare original vs anonymized DICOM files |
| `validator.py` | **Utility** — Validate anonymized files |
| `sample1/` & `sample2/` | **Sample Data** — Test DICOM files for trying out the tool |
| `testSample/` | **Sample Data** — Small test DICOM file |
| `screenshots/` | Visual guides for installation (non-technical users) |

**Recommendation:** Focus on `anonymizer_pro.py`. The other Python files are utilities for troubleshooting and are optional.

---

## Usage

### Command-Line Interface

Anonymize a DICOM file or folder:

```bash
python anonymizer_pro.py \
  -i /path/to/input_dicom_or_folder \
  -p GDPR-strict.json \
  --salt "your-secret-salt-phrase"
```

**Arguments:**

- `-i, --input` — Path to DICOM file or folder (required)
- `-p, --profile` — JSON profile file (default: `GDPR-strict.json`)
- `--salt` — Secret phrase for deterministic pseudonymization (default: random)
- `-o, --output` — Output directory (default: `ANON_EXPORT_YYYYMMDD_HHMMSS`)

**Examples:**

```bash
# Anonymize a single DICOM file
python anonymizer_pro.py -i patient.dcm -p GDPR-strict.json --salt "my-secret"

# Anonymize an entire study folder with research profile
python anonymizer_pro.py -i /data/studies/001 -p research-pseudonymized.json --salt "my-secret"

# Use default GDPR-strict profile
python anonymizer_pro.py -i data.dcm --salt "my-secret"
```

---

## Anonymization Profiles

Profiles control which DICOM tags are removed, replaced, or pseudonymized.

### `GDPR-strict.json` (Maximum De-Identification)

Removes or replaces all personally identifiable information. **Use for public sharing or non-research environments.**

```json
{
  "PatientName": null,
  "PatientBirthDate": null,
  "PatientSex": null,
  "InstitutionName": "DE-IDENTIFIED",
  "StudyID": "ANON",
  "StudyInstanceUID": "PSEUDOUID",
  "SeriesInstanceUID": "PSEUDOUID",
  "FrameOfReferenceUID": "PSEUDOUID",
  "PixelBlackout": true,
  "KeepPrivateTags": false,
  "RetainStudyDate": false
}
```

**Profile Tag Rules:**

- `null` — Remove tag entirely
- `"STRING"` — Replace with fixed string
- `"PSEUDO"` — Generate deterministic pseudonym based on SALT
- `"PSEUDOUID"` — Generate deterministic UID based on SALT
- `true/false` — Special actions (e.g., `PixelBlackout: true` masks burned-in text)

---

## Output

Anonymized files are saved in a timestamped folder:

```
ANON_EXPORT_20260303_143012/
├── anonymization_log.csv          # Detailed transformation log
└── DICOM/
    └── PX3F91AD/                  # Patient pseudonym
        ├── PX3F91AD_20260303_143012_S001_I0001.dcm
        ├── PX3F91AD_20260303_143012_S001_I0002.dcm
        └── ...
```

The **anonymization_log.csv** contains:

- Original and new DICOM tag values
- Patient pseudonym
- Study/Series UIDs
- Transformation timestamp

---

## Security & Best Practices

1. **Use a Secret SALT** — Choose a strong, unique phrase. Store it securely (password manager, HSM, etc.)
2. **Test First** — Always test with a small dataset before processing large studies
3. **Reproducibility** — The same SALT produces the same pseudonyms across runs
4. **Audit Log Security** — The CSV log is unencrypted; protect it like PHI
5. **Version Control** — Keep profiles in version control; rotate profiles with SALT changes

---

## Troubleshooting

- **"Invalid DICOM file"** — File may be corrupted or encrypted. Try extracting with `dcmdump` or another DICOM tool.
- **"Profile not found"** — Ensure JSON file is in the working directory or provide full path.
- **"Permission denied"** — Ensure read access to input folder and write access to output directory.

---

## Contributing

- Report bugs or request features via GitHub Issues
- Contributions welcome — please test thoroughly and document behavior changes
- Keep privacy and reproducibility in mind when proposing changes

---

## License

MIT License — see LICENSE file for details
