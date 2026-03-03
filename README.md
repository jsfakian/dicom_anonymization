# DICOM De-Identification Tool

A **DICOM anonymization tool** for research and clinical use. Supports deterministic pseudonymization, customizable anonymization profiles, pixel blackout, audit logging, and multi-platform support.

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

Profiles control which DICOM tags are removed, replaced, or pseudonymized. Choose a profile that matches your use case:

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

### `research-pseudonymized.json` (Research-Grade)
Keeps clinically relevant data while pseudonymizing identifiers. **Safe for research with proper governance.**

```json
{
  "PatientName": "PSEUDO",
  "PatientSex": "UNKNOWN",
  "PatientBirthDate": null,
  "RetainStudyDate": true,
  "PixelBlackout": false,
  "KeepPrivateTags": true,
  "StudyInstanceUID": "PSEUDOUID",
  "SeriesInstanceUID": "PSEUDOUID"
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
