# DICOM De-Identification

This project provides a **research-grade DICOM anonymization tool** with both **GUI** and **CLI** interfaces.  
It supports deterministic pseudonymization, UID remapping, pixel blackout of burned-in overlays, audit logging, and customizable anonymization profiles.

## Features

| Feature | Description |
|--------|-------------|
| GUI & CLI | Run with a graphical interface or from terminal scripts |
| Deterministic pseudonymization | Uses a user-defined SALT to generate repeatable pseudonyms |
| GDPR / Research-friendly profiles | Configurable JSON-based anonymization profiles |
| Study/Series/Frame UID rewriting | Generates new or pseudonymous UIDs |
| Pixel blackout | Detects and masks burned-in overlays using edge detection |
| Multi-file and folder support | Works on single DICOM files or entire study folders |
| Audit Log | Creates a CSV log showing old vs. new values for traceability |
| Safe output | Never modifies source data; creates a new ANON_EXPORT_* folder |

---

## Installation

Requires Python 3.8+.

Install dependencies (if a requirements file exists):
pip install -r requirements.txt

Or install the package in editable mode:
pip install -e .

```
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.\.venv\Scripts\activate         # Windows
```

## Make it executable

```
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install --upgrade pip wheel
pip install pydicom numpy opencv-python pyinstaller
```

```
pyinstaller --noconfirm --onefile --windowed \
  --name DICOM-DeID anonymizer_pro.py
```

## GUI Usage

To start the graphical application:

```bash
python dicom_anonymize_pro_gui.py
```

| Field                | Description                                              |
| -------------------- | -------------------------------------------------------- |
| **Input**            | A DICOM file *or* a folder containing a series           |
| **Profile JSON**     | An anonymization configuration file                      |
| **SALT**             | Secret string to ensure deterministic pseudonyms         |
| **Output Directory** | The directory where the anonymized results will be saved |

## Command-line usage

Basic example:
python anonymizer_pro.py --input data.csv --output data.anonymized.csv \
    --columns name,email --strategy hash --salt my-secret-salt

Typical options (may vary slightly depending on implementation):

- --input PATH         Input dir
- --output PATH        Output dir
- --profile-fname FILE JSON config that defines per-column rules
- --salt VALUE         Salt or key for deterministic transformations

```
python anonymize_pro.py \
  -i /path/to/dicom_or_folder \
  -o /path/to/output \
  -p GDPR-strict.json \
  --salt "my-secret-salt" 
```

## Example Profiles

### GDPR-strict.json

```
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

### research_pseudo.json

```
{
  "PatientName": "PSEUDO",
  "PatientSex": "UNKNOWN",
  "RetainStudyDate": true,
  "PixelBlackout": false,
  "KeepPrivateTags": true,
  "StudyInstanceUID": "PSEUDOUID",
  "SeriesInstanceUID": "PSEUDOUID"
}
```

## Output Structure

```
ANON_EXPORT_20250110_153012/
 ├── anonymization_log.csv
 └── DICOM/
     └── PX3F91AD/
         ├── PX3F91AD_20250110_153012_S001_I0001.dcm
         ├── PX3F91AD_20250110_153012_S001_I0002.dcm
         └── ...
```

## Safety & reproducibility

- Use a secret salt/seed to make transformations deterministic across runs
- Test with a small dataset
- Keep mapping tables (if generated) secure; they may re-identify users

## Logging & errors

- Tool logs progress and a summary of transformations applied
- Invalid rows are either skipped, logged, or fail fast depending on config

## Contributing

- Open issues for bugs or feature requests
- Add tests for new anonymization strategies
- Keep privacy and reproducibility considerations in mind when changing behavior

## License

MIT License — see LICENSE file for details
