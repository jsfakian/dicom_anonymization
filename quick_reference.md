# DICOM Anonymization Tool - Quick Reference Card

## Installation (One-Time Setup)

### Option 1: Automated (macOS/Linux)
```bash
bash setup_environment.sh
source .venv/bin/activate
```

### Option 2: Manual (All Platforms)
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Installation
```bash
python anonymizer_pro.py --help
```

---

## Standard Commands

### Single File with Default Settings
```bash
python anonymizer_pro.py \
  -i /path/to/file.dcm \
  --salt "YourSecretKey2026"
```

### Batch Process Directory (GDPR-strict)
```bash
python anonymizer_pro.py \
  -i /path/to/studies/ \
  -p GDPR-strict.json \
  --salt "YourSecretKey2026"
```

### Batch with Research Profile
```bash
python anonymizer_pro.py \
  -i /path/to/studies/ \
  -p research-pseudonymized.json \
  --salt "YourSecretKey2026" \
  -o custom_output_folder/
```

### Use SALT from Secure File
```bash
python anonymizer_pro.py \
  -i input.dcm \
  --salt "$(cat /secure/.salt_file)"
```

---

## Argument Reference

| Argument | Type | Required | Default | Example |
|----------|------|----------|---------|---------|
| `-i, --input` | path | YES | - | `-i /data/studies/` |
| `--salt` | string | YES | - | `--salt "SecureKey123"` |
| `-p, --profile-fname` | file | NO | `GDPR-strict.json` | `-p research-pseudonymized.json` |
| `-o, --output` | path | NO | Current directory | `-o /output/path/` |
| `-h, --help` | flag | NO | - | `--help` (shows all options) |

---

## Built-in Profiles

### GDPR-strict.json (Maximum Privacy)
- **PatientName:** Removed/Replaced with "ANON"
- **PatientID:** Pseudonym (PX + 6 hex chars)
- **Dates:** All removed (StudyDate, PatientBirthDate)
- **UIDs:** Random (every run produces different UIDs)
- **Pixel Blackout:** Enabled (masks burned-in text)
- **Use Case:** Public data sharing, strict compliance

**Example:**
```bash
python anonymizer_pro.py -i study.dcm -p GDPR-strict.json --salt "KEY"
```

### research-pseudonymized.json (Balanced Approach)
- **PatientName:** Pseudonym
- **PatientID:** Pseudonym (consistent across runs)
- **Dates:** Kept (StudyDate, StudyTime preserved)
- **UIDs:** Deterministic (same run = same UIDs)
- **Pixel Blackout:** Disabled
- **Use Case:** Institutional research, data linkage needed

**Example:**
```bash
python anonymizer_pro.py -i study.dcm -p research-pseudonymized.json --salt "KEY"
```

---

## Output Structure

```
ANON_EXPORT_YYYYMMDD_HHMMSS/
├── anonymization_log.csv           # Audit trail of all changes
└── DICOM/
    └── PXa3f5c8/                   # Patient pseudonym folder
        ├── PXa3f5c8_20260303_120704_S001_I0001.dcm
        ├── PXa3f5c8_20260303_120704_S001_I0002.dcm
        └── ... (one file per image)
```

**Filename Format:** `{PSEUDONYM}_{TIMESTAMP}_S{SERIES}_I{IMAGE}.dcm`

**CSV Columns:** run_timestamp, input_file, output_file, original_patient_id, pseudonym_patient_id, dicom_tag, original_value, new_value, status

---

## Utility Scripts

### View DICOM Metadata (Before Anonymization)
```bash
python dump_dicom_metadata.py sample1/Brain\ 5\ mm/img001.dcm

# Show only specific tags
python dump_dicom_metadata.py sample1/Brain\ 5\ mm/img001.dcm | grep -E "PatientName|PatientID|StudyDate"
```

### Validate Anonymized Files
```bash
python validator.py \
  -i demo_output/DICOM/PX*/ \
  -p GDPR-strict.json
```

### Compare Original vs. Anonymized
```bash
python compare_dicom_images.py \
  -original sample1/Brain\ 5\ mm/img001.dcm \
  -anonymized demo_output/DICOM/PXa3f5c8/PXa3f5c8_..._I0001.dcm
```

---

## Common Workflows

### Workflow 1: Quick Single Study
```bash
# Process one patient's brain study
python anonymizer_pro.py \
  -i sample1/Brain\ 5\ mm/ \
  --salt "MySecretKey"

# Check output
ls -la ANON_EXPORT_*/DICOM/*/

# View audit log
head -20 ANON_EXPORT_*/anonymization_log.csv
```

### Workflow 2: Batch Process Multiple Studies
```bash
# Store SALT securely
echo "SecureRandomKey123456789" > /tmp/.salt_key
chmod 600 /tmp/.salt_key

# Process all studies
python anonymizer_pro.py \
  -i /data/studies/ \
  -p GDPR-strict.json \
  --salt "$(cat /tmp/.salt_key)" \
  -o /data/anonymized/

# Verify results
wc -l /data/anonymized/ANON_*/anonymization_log.csv
find /data/anonymized/ANON_*/DICOM/ -name "*.dcm" | wc -l
```

### Workflow 3: Custom Profile for Research
```bash
# Copy and modify profile
cp GDPR-strict.json my-research-profile.json
# Edit my-research-profile.json to suit your needs

# Use custom profile
python anonymizer_pro.py \
  -i input_studies/ \
  -p my-research-profile.json \
  --salt "ResearchKey"

# Validate against custom profile
python validator.py -i output/ -p my-research-profile.json
```

---

## Deterministic Pseudonymization

### How It Works
```
PatientID: "12345"
SALT: "SecureKey2026"
                    ↓ SHA1 Hash
Pseudonym: "PXa3f5c8"
```

### Key Properties
- **Same input + Same SALT = Same pseudonym (deterministic)**
- **Different input + Same SALT = Different pseudonym**
- **Can't reverse without SALT (secure one-way)**

### Example
```bash
# Run 1
python anonymizer_pro.py -i patient_001.dcm --salt "KEY"
# Output: DICOM/PXa3f5c8/...

# Run 2 (same patient, same SALT)
python anonymizer_pro.py -i patient_001_reimport.dcm --salt "KEY"
# Output: DICOM/PXa3f5c8/...  ← Same pseudonym!

# Run 3 (different SALT)
python anonymizer_pro.py -i patient_001.dcm --salt "DIFFERENT_KEY"
# Output: DICOM/PXf7d2e1/...  ← Different pseudonym
```

---

## Profile Customization

### Create Custom Profile
```bash
cp GDPR-strict.json custom-profile.json
```

### Profile Structure
```json
{
  "profile_name": "my-profile",
  "profile_version": "1.0",
  "description": "Custom de-identification rules",
  "actions": {
    "PatientName": "ANON",           // ✓ Fixed replacement
    "PatientID": "PSEUDO",           // ✓ Pseudonym hash
    "PatientBirthDate": null,        // ✓ Remove completely
    "StudyDate": "ANON",             // ✓ Replace with fixed value
    "StudyInstanceUID": "NEWUID",    // ✓ Random UID
    "SeriesInstanceUID": "PSEUDOUID",// ✓ Deterministic UID
    "PatientAge": "ANON",            // ✓ Custom string
    "PatientSex": null,              // ✓ Remove
    "PixelBlackout": true,           // ✓ Enable pixel masking
    "KeepPrivateTags": false,        // ✓ Remove private tags
    "RetainStudyDate": false         // ✓ Don't keep dates
  }
}
```

### Tag Action Guide
| Action | Meaning | Use Case |
|--------|---------|----------|
| `null` | Remove tag | `"PatientBirthDate": null` |
| `"STRING"` | Replace with text | `"PatientName": "ANON"` |
| `"PSEUDO"` | Patient pseudonym | `"PatientID": "PSEUDO"` |
| `"PSEUDOUID"` | Deterministic UID | `"SeriesInstanceUID": "PSEUDOUID"` |
| `"NEWUID"` | Random UID | `"StudyInstanceUID": "NEWUID"` |

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `-bash: python: command not found` | Python not installed or wrong shell | Install Python 3.8+; use `python3` |
| `ModuleNotFoundError: No module named 'pydicom'` | Virtual env not activated | Run `source .venv/bin/activate` |
| `Input path does not exist` | Wrong file path | Verify path with `ls /path/to/file` |
| `Invalid DICOM file` | File corrupted or not DICOM format | Check with `file filename.dcm` |
| `Pseudonyms different between runs` | SALT changed | Keep SALT identical for consistency |
| `Output folder already exists` | Previous run output still present | Use different `--output` or remove old |
| `Out of memory` | Processing very large images | Process one file at a time |
| `Slow processing (>5 sec per file)` | Pixel blackout enabled on large images | Disable `PixelBlackout` or use SSD |

---

## Security Checklist

- [ ] SALT is 20+ characters, random, stored securely
- [ ] SALT is NOT in version control (git)
- [ ] SALT is NOT shared with recipients of anonymized data
- [ ] Original files NOT shared with anonymized data
- [ ] Audit logs stored separately with restricted permissions
- [ ] Output directory on secure/controlled system
- [ ] Delete/archive anonymization_log.csv after verification

---

## Sample Data for Testing

| Folder | Size | Files | Use |
|--------|------|-------|-----|
| `sample1/Brain 5 mm/` | ~10MB | 36 DICOM | Standard test |
| `sample2/Brain 5 mm/` | ~10MB | 35 DICOM | Duplicate test |
| `testSample/` | ~100KB | 1 DICOM | Quick test |

**Quick Test:**
```bash
python anonymizer_pro.py -i testSample/img001.dcm --salt "TEST"
# Should complete in <1 second
```

---

## Performance Expectations

| Dataset | Files | Approx. Time | Rate |
|---------|-------|--------------|------|
| Single file | 1 | 0.5s | - |
| One series | 10 | 3s | 3.3 files/sec |
| One study | 100 | 25s | 4 files/sec |
| Large batch | 1000 | 250s | 4 files/sec |

**Speed Tips:**
1. Use SSD (faster than HDD by ~30%)
2. Disable pixel blackout if not needed (saves ~20%)
3. Use research-pseudonymized profile (faster UID generation)

---

## Getting Help

### Check Logs
```bash
# View error messages
tail -100 ANON_EXPORT_*/anonymization_log.csv

# Look for ERROR status
grep -i error ANON_EXPORT_*/anonymization_log.csv
```

### Run Help Command
```bash
python anonymizer_pro.py --help
```

### Test Installation
```bash
python -c "import pydicom; import numpy; import cv2; print('✓ All dependencies OK')"
```

### Validate Anonymization
```bash
python validator.py -i output/DICOM/ -p GDPR-strict.json
```

---

## Quick Links

- **Main Script:** `anonymizer_pro.py`
- **Profiles:** `GDPR-strict.json`, `research-pseudonymized.json`
- **Setup Script:** `setup_environment.sh`
- **Full Documentation:** `README.md`, `INSTALL.md`
- **Technical Details:** `technical_appendix.md`
- **Demo Guide:** `demo_walkthrough_guide.md`

---

**Last Updated:** March 3, 2026 | **Version:** 1.0
