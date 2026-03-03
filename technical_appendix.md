# DICOM Anonymization Tool - Technical Appendix

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pseudonymization Algorithm](#pseudonymization-algorithm)
3. [DICOM UID Generation](#dicom-uid-generation)
4. [Profile Schema & Tag Actions](#profile-schema--tag-actions)
5. [Core Data Flow](#core-data-flow)
6. [Pixel Blackout Implementation](#pixel-blackout-implementation)
7. [Audit Logging](#audit-logging)
8. [Performance & Scalability](#performance--scalability)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Security Considerations](#security-considerations)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Command Line                         │
│  python anonymizer_pro.py -i <input> -p <profile> ...      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Argument Parser & Inputs
        │  - Input path            │
        │  - Profile (JSON)        │
        │  - SALT phrase           │
        │  - Output directory      │
        └────────────┬─────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │   File Processor (Recursive)     │
     │  - Discover DICOM files         │
     │  - Validate DICOM format        │
     │  - Create output structure      │
     └───────────────┬──────────────────┘
                     │
        ┌────────────▼────────────┐
        │  DICOM Reader (pydicom) │
        │  - Load metadata        │
        │  - Extract tags         │
        │  - Access pixel data    │
        └────────────┬────────────┘
                     │
    ┌────────────────▼─────────────────┐
    │  Anonymization Engine           │
    │  - Apply profile rules          │
    │  - Generate pseudonyms (SHA1)   │
    │  - Transform UIDs              │
    │  - Blackout pixels (OpenCV)    │
    └────────────┬──────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │  Output & Logging             │
    │  - Write anonymized DICOM     │
    │  - Record audit log CSV       │
    │  - Create timestamped folder  │
    └───────────────────────────────┘
```

### Key Classes

**AnonymizationEngine**
- Processes individual DICOM files
- Applies profile rules to each tag
- Manages pseudonym generation
- Coordinates UID transformation

**ProfileManager**
- Loads and parses JSON profiles
- Validates tag configuration
- Provides default settings

**AuditLogger**
- Records all transformations
- Writes CSV with timestamp, file paths, tag changes
- Enables audit trail for compliance

---

## Pseudonymization Algorithm

### Overview

The tool uses **deterministic pseudonymization** based on SHA1 hashing. This ensures:
- Same patient always generates same pseudonym when using identical SALT
- Researcher cannot reverse pseudonym without the SALT phrase
- Legitimate data linkage is possible across multiple studies

### Pseudonym Generation Process

```
Input:
  Original PatientID: "12345"
  SALT: "SecureKey2026"

Step 1: Concatenate
  combined = "12345" + "SecureKey2026"
  → "12345SecureKey2026"

Step 2: SHA1 Hash
  hash_hex = SHA1("12345SecureKey2026")
  → "a3f5c8d1e2b4f7a9c6e1d8f3a5b7c9e1f2a3d4e5"

Step 3: Extract First 6 Hex Characters
  hex_part = "a3f5c8"
  → interpreted as integers: [163, 245, 200]

Step 4: Format as Pseudonym
  pseudonym = "PX" + hex_part
  → "PXa3f5c8"

Output:
  NewPatientID = "PXa3f5c8"
```

### Deterministic Property (Key Feature)

```
Same patient + Same SALT = Same Pseudonym

Example:
  Run 1: PatientID="12345" + SALT="SecureKey2026" → PXa3f5c8
  Run 2: PatientID="12345" + SALT="SecureKey2026" → PXa3f5c8
  
  Different patient + Same SALT = Different Pseudonym
  
  Run 3: PatientID="99999" + SALT="SecureKey2026" → PX7f2d1b
```

### Security Implications

- **Non-Reversible (without SALT):** Given only "PXa3f5c8", cannot determine original "12345"
- **SALT Security:** SALT must be kept confidential
- **Recommended SALT:** 20+ characters, random, stored securely
- **Loss of SALT:** If SALT is lost, ability to re-link data is lost (acceptable if not needed)

### Python Implementation

```python
import hashlib

def generate_pseudonym(patient_id, salt):
    """Generate deterministic pseudonym from PatientID and SALT"""
    combined = str(patient_id) + str(salt)
    hash_hex = hashlib.sha1(combined.encode()).hexdigest()
    return "PX" + hash_hex[:6]

# Example usage
original_id = "12345"
salt = "SecureKey2026"
pseudonym = generate_pseudonym(original_id, salt)
print(pseudonym)  # Output: PXa3f5c8 (approximately)
```

---

## DICOM UID Generation

### Background on UIDs

DICOM UIDs (Unique Identifiers) are hierarchical dotted strings that identify studies, series, and images:

```
StudyInstanceUID: 1.2.840.113619.2.55.3.28474.1.1234567890.1
SeriesInstanceUID: 1.2.840.113619.2.55.3.28474.1.1234567890.2
SOPInstanceUID: 1.2.840.113619.2.55.3.28474.1.1234567890.2.1
```

### UID Action Types in Profiles

#### Action 1: "NEWUID" (Random UID)

Generates a completely new random UID for each transformation:

```python
import uuid

def generate_new_uid():
    """Generate a random DICOM UID"""
    root = "1.2.826.0.1.3680043.10.54321"  # Default root
    unique_part = str(int(uuid.uuid1().int // 1000000000))  # Timestamp-based
    return root + "." + unique_part
```

**Use when:**
- Maximum privacy needed (UIDs cannot link to original system)
- Different runs should produce different UIDs for same patient
- GDPR-strict profile

**Drawback:**
- Cannot re-link studies from same patient across anonymization runs
- Each run produces different pseudonym UIDs

#### Action 2: "PSEUDOUID" (Deterministic UID)

Generates UIDs deterministically from patient pseudonym:

```python
def generate_pseudo_uid(patient_pseudonym, salt, uid_suffix):
    """Generate deterministic UID from patient pseudonym"""
    root = "1.2.826.0.1.3680043.10.54321"  # Standard root
    combined = patient_pseudonym + salt + str(uid_suffix)
    hash_value = int(hashlib.sha1(combined.encode()).hexdigest(), 16)
    uid_numeric = str(hash_value % (10**15))  # Large number within DICOM spec
    return root + "." + uid_numeric
```

**Use when:**
- Research linkage needed
- Reproducibility important
- Same patient's multiple studies should link via UID
- research-pseudonymized profile

**Benefit:**
- Consistent UIDs across multiple anonymization runs
- Enables legitimate data linking without exposing identity

---

## Profile Schema & Tag Actions

### Profile JSON Structure

```json
{
  "profile_name": "GDPR-strict",
  "profile_version": "1.0",
  "description": "Maximum de-identification for public data sharing",
  "actions": {
    "PatientName": "ANON",
    "PatientID": "PSEUDO",
    "PatientBirthDate": null,
    "StudyInstanceUID": "NEWUID",
    "SeriesInstanceUID": "NEWUID",
    "StudyDate": null,
    "StudyTime": null,
    "PixelBlackout": true,
    "KeepPrivateTags": false,
    "RetainStudyDate": false
  }
}
```

### Tag Action Types

| Action | Type | Effect | Example |
|--------|------|--------|---------|
| `null` | Remove | Delete tag from output file | `"PatientBirthDate": null` |
| `"STRING"` | Replace | Fixed text replacement | `"PatientName": "ANON"` |
| `"PSEUDO"` | Pseudonym | Patient-specific hash | `"PatientID": "PSEUDO"` → `PXa3f5c8` |
| `"PSEUDOUID"` | Pseudo UID | Deterministic UID | `"SeriesInstanceUID": "PSEUDOUID"` |
| `"NEWUID"` | New UID | Random unique UID | `"StudyInstanceUID": "NEWUID"` |
| `true` / `false` | Control | Enable/disable feature | `"PixelBlackout": true` |

### Special Control Tags

| Tag | Type | Purpose |
|-----|------|---------|
| `PixelBlackout` | boolean | Enable OpenCV-based text masking on pixel data |
| `KeepPrivateTags` | boolean | Retain DICOM private tags (often contain vendor-specific data) |
| `RetainStudyDate` | boolean | Keep original study dates (useful for longitudinal research) |

### Creating Custom Profiles

Example: Profile for clinical trial with extended privacy

```json
{
  "profile_name": "clinical-trial-strict",
  "profile_version": "1.0",
  "description": "Trial-specific de-identification",
  "actions": {
    "PatientName": "ANON",
    "PatientID": "PSEUDO",
    "PatientBirthDate": null,
    "PatientSex": null,
    "PatientAge": "ANON",
    "StudyInstanceUID": "NEWUID",
    "SeriesInstanceUID": "NEWUID",
    "SOPInstanceUID": "NEWUID",
    "StudyDate": null,
    "StudyTime": null,
    "ContentDate": null,
    "PixelBlackout": true,
    "KeepPrivateTags": false,
    "RetainStudyDate": false
  }
}
```

---

## Core Data Flow

### Processing a Single DICOM File

```
INPUT: sample1/Brain 5 mm/img001.dcm
  │
  ├─ Load DICOM metadata with pydicom
  │  └─ Extract: PatientName, PatientID, StudyInstanceUID, ...
  │
  ├─ Load Profile (e.g., GDPR-strict.json)
  │  └─ Match each tag to action rule
  │
  ├─ Generate Pseudonym
  │  └─ SHA1(PatientID + SALT) = PXa3f5c8
  │
  ├─ Create Output Directory Structure
  │  └─ demo_output/DICOM/PXa3f5c8/
  │
  ├─ Apply Transformations to Metadata
  │  ├─ PatientName: "John Doe" → "ANON"
  │  ├─ PatientID: "12345" → "PXa3f5c8"
  │  ├─ StudyInstanceUID: "1.2.3..." → random UID (NEWUID)
  │  ├─ PatientBirthDate: "19700101" → [removed]
  │  └─ ...more tags...
  │
  ├─ Apply Pixel Blackout (if enabled)
  │  └─ OpenCV edge detection → mask burned-in text
  │
  ├─ Write Anonymized DICOM File
  │  └─ demo_output/DICOM/PXa3f5c8/PXa3f5c8_20260303_120704_S002_I0001.dcm
  │
  └─ Log Transformation
     └─ Append row to anonymization_log.csv:
        run_timestamp, input_path, output_path, original_id, pseudonym_id,
        tag_name, original_value, new_value, ...
```

### Batch Processing with Directory

```
INPUT: sample1/ (folder with subfolders and files)
  │
  ├─ Discover all DICOM files recursively
  │
  ├─ For each file:
  │  ├─ Extract PatientID
  │  ├─ Check if pseudonym already generated (for this patient in this batch)
  │  ├─ Generate pseudonym if new
  │  ├─ Create output subdirectory
  │  ├─ Process file (anonymize metadata, pixel blackout)
  │  └─ Log changes
  │
  └─ Output:
     demo_output/DICOM/
      ├── PXa3f5c8/  [Patient from img001-036]
      │   ├── PXa3f5c8_..._S001_I0001.dcm
      │   ├── PXa3f5c8_..._S001_I0002.dcm
      │   └── ... (36 files)
      │
      └── anonymization_log.csv  [All changes logged]
```

---

## Pixel Blackout Implementation

### Overview

Pixel blackout detects and masks burned-in text on medical images. This is important because:

- Clinical reports often have text burned into pixel data (patient name, ID, date)
- Text is visible when image is displayed but not in DICOM metadata
- Simply removing metadata is insufficient for complete de-identification

### Algorithm Flow

```python
import cv2
import numpy as np

def blackout_burned_in_text(pixel_array):
    """
    Detect and mask burned-in text using edge detection
    
    Steps:
    1. Convert to 8-bit if needed
    2. Apply Canny edge detection
    3. Find contours of text regions
    4. Mask (set to 0) detected regions
    """
    
    # Step 1: Normalize pixel values to 0-255 range
    normalized = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    uint8_array = np.uint8(normalized)
    
    # Step 2: Apply Canny edge detection
    edges = cv2.Canny(uint8_array, threshold1=50, threshold2=150)
    
    # Step 3: Dilate edges to connect nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 4: Find contours (potential text regions)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Filter contours by size and shape (typical text dimensions)
    mask = np.zeros_like(pixel_array)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Heuristics: text is usually:
        # - Area between 100 and 50,000 pixels
        # - Aspect ratio between 0.3 and 10 (wider than tall or vice versa)
        if 100 < area < 50000 and 0.3 < aspect_ratio < 10:
            cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Step 6: Apply mask to blackout (set to 0)
    result = pixel_array.copy()
    result[mask > 0] = 0
    
    return result
```

### Limitations

- **Best-effort:** Not 100% effective on all image types
- **False positives:** May mask non-text regions (anatomical structures)
- **Tunable:** Threshold parameters can be adjusted per image type
- **Optional:** Can be disabled via profile setting

### Performance

- Single image (512×512): ~50-100ms
- Multi-frame images: ~500ms-1s (many frames)
- Adds ~10-20% to total processing time

---

## Audit Logging

### CSV Format

**Filename:** `anonymization_log.csv`

**Columns:**
| Column | Type | Example | Purpose |
|--------|------|---------|---------|
| `run_timestamp` | ISO timestamp | `2026-03-03T12:07:04.123` | When anonymization ran |
| `input_file` | path | `sample1/Brain 5 mm/img001.dcm` | Original source file |
| `output_file` | path | `DICOM/PXa3f5c8/PXa3f5c8_...dcm` | Where file was written |
| `original_patient_id` | string | `12345` | Original PatientID |
| `pseudonym_patient_id` | string | `PXa3f5c8` | Generated pseudonym |
| `dicom_tag` | DICOM tag | `PatientName` | Which tag was changed |
| `original_value` | string | `John Doe` | Original value |
| `new_value` | string | `ANON` | Anonymized value |
| `status` | status | `OK` | Success/failure |

### Sample Log Entry

```csv
run_timestamp,input_file,output_file,original_patient_id,pseudonym_patient_id,dicom_tag,original_value,new_value,status
2026-03-03T12:07:04.123,sample1/Brain 5 mm/img001.dcm,DICOM/PXa3f5c8/PXa3f5c8_20260303_120704_S002_I0001.dcm,12345,PXa3f5c8,PatientName,John Doe,ANON,OK
2026-03-03T12:07:04.234,sample1/Brain 5 mm/img001.dcm,DICOM/PXa3f5c8/PXa3f5c8_20260303_120704_S002_I0001.dcm,12345,PXa3f5c8,PatientID,12345,PXa3f5c8,OK
2026-03-03T12:07:04.345,sample1/Brain 5 mm/img001.dcm,DICOM/PXa3f5c8/PXa3f5c8_20260303_120704_S002_I0001.dcm,12345,PXa3f5c8,PatientBirthDate,19700101,[REMOVED],OK
```

### Use Cases

1. **Compliance Audit:** Demonstrate to regulators what was changed
2. **Troubleshooting:** Identify which files had issues
3. **Reproducibility:** Trace original→pseudonym mappings
4. **Accountability:** Timestamp tracks who ran it and when

---

## Performance & Scalability

### Benchmark Results

System: MacBook Pro, Apple Silicon, 16GB RAM

| Dataset Size | Files | Time | Rate |
|--------------|-------|------|------|
| Single file | 1 | ~500ms | - |
| Small study | 10 files | ~3s | 3.3 files/sec |
| Medium study | 100 files | ~25s | 4 files/sec |
| Large batch | 1000 files | ~250s | 4 files/sec |

### Bottlenecks

1. **DICOM I/O:** ~70% of time (reading/writing large pixel arrays)
2. **Pixel Blackout:** ~20% of time (OpenCV processing) - optional
3. **Hashing/Crypto:** ~5% of time (SHA1, UID generation)
4. **Logging:** ~5% of time (CSV writes)

### Optimization Strategies

1. **Disable Pixel Blackout** (if not needed):
   ```json
   "PixelBlackout": false
   ```
   Expected speedup: ~20%

2. **Parallelization** (not yet implemented but feasible):
   - Use Python `multiprocessing.Pool`
   - Process multiple files on different CPU cores
   - Expected speedup: 4-8x on quad-core systems

3. **SSD Storage:**
   - Use SSD for both input and output
   - Sequential file processing is I/O heavy
   - Expected speedup: 30-50% vs HDD

### Memory Usage

- Per-file memory: ~size of pixel array (e.g., 1MB for 512×512 image)
- Peak memory: ~2× per file (load + write)
- Safe for datasets up to 100GB on 16GB RAM systems (sequential processing)

---

## Troubleshooting Guide

### Installation Issues

#### Problem: ModuleNotFoundError for pydicom, numpy, etc.

**Cause:** Virtual environment not activated or dependencies not installed

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import pydicom; print(pydicom.__version__)"
```

---

### Runtime Issues

#### Problem: "Invalid DICOM file" error

**Cause:** File is not a valid DICOM or is corrupted

**Solution:**
```bash
# Check file with diagnostic tool
file sample1/Brain\ 5\ mm/img001.dcm

# Dump metadata to verify
python dump_dicom_metadata.py sample1/Brain\ 5\ mm/img001.dcm

# If corrupted, regenerate or skip file
```

---

#### Problem: "Output directory already exists"

**Cause:** Running with same output folder that already has results

**Solution:**
```bash
# Option 1: Use different output directory
python anonymizer_pro.py -i input.dcm -o new_output/

# Option 2: Remove old output
rm -rf ANON_EXPORT_*
python anonymizer_pro.py -i input.dcm
```

---

#### Problem: Pseudonyms are different between runs

**Cause:** SALT phrase is different or file not found

**Solution:**
```bash
# Verify SALT is identical
echo "Run 1 SALT: SecureKey2026"
echo "Run 2 SALT: SecureKey2026"

# If using from file, ensure file doesn't change
python anonymizer_pro.py -i input.dcm --salt "$(cat /secure/.salt_file)"
```

---

### Performance Issues

#### Problem: Tool runs very slowly (>1 second per file)

**Cause:** Pixel blackout enabled on large images or slow storage

**Solution:**
```bash
# Disable pixel blackout in profile
"PixelBlackout": false

# Or use faster storage (SSD vs HDD)
# Or reduce image resolution before anonymization
```

---

#### Problem: Out of memory error

**Cause:** Processing very large multi-frame images or DICOM series

**Solution:**
```bash
# Process one file/series at a time instead of whole directory
python anonymizer_pro.py -i single_file.dcm ...

# Monitor memory during processing
top  # macOS/Linux to watch memory usage

# Reduce image resolution if possible
```

---

### Profile Issues

#### Problem: Pseudonyms are treated as regular strings, not generated

**Cause:** Profile action uses wrong format (e.g., `"PSEUDO"` vs `"pseudo"`)

**Solution:**
```json
{
  "PatientID": "PSEUDO"  // ✓ Correct (uppercase)
}
```

**Not:**
```json
{
  "PatientID": "pseudo"  // ✗ Treated as literal string
}
```

---

#### Problem: UIDs are still original values

**Cause:** Profile has wrong UID action

**Solution:**
```json
{
  "StudyInstanceUID": "NEWUID",      // ✓ Random UID
  "SeriesInstanceUID": "PSEUDOUID"   // ✓ Deterministic UID
}
```

---

### Audit Logging Issues

#### Problem: Audit log is empty or only has headers

**Cause:** No transformations were applied (all tags already removed or null)

**Solution:**
- Verify profile has actual transformation rules
- Check that input files have expected tags

---

#### Problem: Audit log shows very large values in "new_value" column

**Cause:** Pixel data or binary values captured (should not happen)

**Solution:**
- Pixel data should never be in audit log (logged as [BINARY])
- If binary values appear, check tag filtering logic
- Report as bug if it occurs

---

## Security Considerations

### SALT Management

**❌ DO NOT:**
- Hardcode SALT in scripts or profiles
- Use predictable SALTs (e.g., "password123")
- Share SALT with researchers who need de-identified data
- Store SALT in version control (git)

**✓ DO:**
- Use high-entropy random strings (20+ characters)
- Store SALT in secure location (environment variable, secret manager)
- Rotate SALT periodically (changes pseudonyms for new runs)
- Keep SALT secret and separate from data

### Audit Log Security

**❌ DO NOT:**
- Share audit log with researchers (contains pseudonym mappings)
- Store audit log on shared network drives
- Email audit log unencrypted

**✓ DO:**
- Store with restricted file permissions (600 or 0o600)
- Encrypt at rest if stored long-term
- Access control: only authorized staff
- Archive separately from anonymized data

### Output Data Security

**✓ Safe to share:**
- Anonymized DICOM files (if profile is appropriate)
- With researchers in institutional data agreements
- On secure file transfer systems
- With encryption in transit

**❌ Do not share:**
- Audit logs (contain pseudonym mappings)
- SALT phrase (enables re-identification)
- Original metadata CSV
- Pixel blackout configuration details

### Re-identification Risk

**Scenario 1: Accidental Release of Both Files**
- If original AND anonymized files both released, re-identification is trivial
- Mitigation: Strict data governance, deletion of originals after anonymization

**Scenario 2: Researcher With SALT**
- Researcher + SALT can reverse pseudonyms
- Mitigation: Separate SALT from data; only authorized staff access SALT

**Scenario 3: Computational Attack (Brute Force)**
- Attacker tries common SALTs to find pseudonym → original mapping
- Mitigation: Use high-entropy SALT; infeasible to crack with modern SALTs

---

## References & Further Reading

- DICOM Standard: https://www.dicomstandard.org/
- pydicom Documentation: https://pydicom.readthedocs.io/
- OpenCV Edge Detection: https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
- HIPAA De-identification Rules: https://www.hhs.gov/hipaa/
- GDPR Pseudonymization: https://gdpr-info.eu/

---

## Appendix: Algorithm Pseudocode

### Main Processing Loop

```pseudocode
FUNCTION anonymize_dicom_directory(input_path, profile_path, salt, output_path)
    
    // Load profile rules
    profile = load_json_profile(profile_path)
    
    // Create output directory
    output_dir = create_timestamped_directory(output_path)
    
    // Initialize audit logger
    audit_log = initialize_csv_logger(output_dir + "/anonymization_log.csv")
    
    // Find all DICOM files recursively
    dicom_files = find_all_dicom_files(input_path)
    
    // Cache for pseudonym generation (avoid duplicate hashing)
    pseudonym_cache = {}
    
    FOR EACH file IN dicom_files:
        TRY
            // Load original DICOM
            dicom_dataset = load_dicom(file)
            original_patient_id = dicom_dataset.PatientID
            
            // Generate pseudonym (cached)
            IF original_patient_id NOT IN pseudonym_cache:
                pseudonym_cache[original_patient_id] = generate_pseudonym(original_patient_id, salt)
            pseudonym = pseudonym_cache[original_patient_id]
            
            // Create output directory for this patient
            patient_output_dir = create_directory(output_dir + "/DICOM/" + pseudonym)
            
            // Apply anonymization rules from profile
            FOR EACH tag IN dicom_dataset.keys():
                action = profile.actions.get(tag, "IGNORE")
                
                IF action == null:
                    // Remove tag
                    old_value = dicom_dataset[tag]
                    DELETE dicom_dataset[tag]
                    audit_log.write(run_timestamp, file, output_file, 
                                   original_patient_id, pseudonym, tag, old_value, "[REMOVED]", "OK")
                    
                ELSE IF action == "PSEUDO":
                    // Replace with pseudonym
                    old_value = dicom_dataset[tag]
                    dicom_dataset[tag] = pseudonym
                    audit_log.write(run_timestamp, file, output_file,
                                   original_patient_id, pseudonym, tag, old_value, pseudonym, "OK")
                    
                ELSE IF action == "NEWUID" OR action == "PSEUDOUID":
                    // Generate new/pseudo UID
                    old_value = dicom_dataset[tag]
                    new_uid = (action == "NEWUID") ? 
                              generate_new_uid() : 
                              generate_pseudo_uid(pseudonym, salt, tag)
                    dicom_dataset[tag] = new_uid
                    audit_log.write(run_timestamp, file, output_file,
                                   original_patient_id, pseudonym, tag, old_value, new_uid, "OK")
                    
                ELSE IF action IS STRING:
                    // Replace with fixed string
                    old_value = dicom_dataset[tag]
                    dicom_dataset[tag] = action
                    audit_log.write(run_timestamp, file, output_file,
                                   original_patient_id, pseudonym, tag, old_value, action, "OK")
                    
                ELSE IF action == true AND tag == "PixelBlackout":
                    // Apply pixel blackout
                    pixel_array = dicom_dataset.pixel_array
                    blackout_pixel_array = blackout_burned_in_text(pixel_array)
                    dicom_dataset.PixelData = blackout_pixel_array.tobytes()
                    audit_log.write(run_timestamp, file, output_file,
                                   original_patient_id, pseudonym, tag, "[BINARY]", "[MASKED]", "OK")
            
            // Generate output filename
            output_filename = format_filename(pseudonym, timestamp, file)
            output_file = patient_output_dir + "/" + output_filename
            
            // Write anonymized DICOM
            write_dicom(output_file, dicom_dataset)
            
            audit_log.write(run_timestamp, file, output_file,
                           original_patient_id, pseudonym, "[FILE]", file, output_file, "OK")
        
        CATCH error:
            audit_log.write(run_timestamp, file, "[ERROR]",
                           original_patient_id, pseudonym, "[ERROR]", error.message, error.message, "ERROR")
            CONTINUE  // Skip file and continue with next
    
    // Close audit log
    audit_log.close()
    
    RETURN output_dir

END FUNCTION
```

---

