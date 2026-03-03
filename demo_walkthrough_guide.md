# DICOM Anonymization Tool - Live Demo Walkthrough

## Overview

This guide provides step-by-step instructions for demonstrating the DICOM anonymization tool to partners. The demo is designed to take approximately 10 minutes and showcases the tool's core functionality with real sample data.

---

## Pre-Demo Checklist

- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] Working directory: `/Users/jsfakian/Documents/src/dicom_anonymization`
- [ ] Sample data available: `sample1/Brain 5 mm/` (36 DICOM files)
- [ ] Presentation slides ready
- [ ] Terminal window sized for visibility (larger font recommended)

---

## Demo Flow

### Setup (1 minute)

**What to show before running:**

1. Open file explorer and navigate to `sample1/Brain 5 mm/`
   - Show: "This is raw patient data with 36 brain imaging slices"
   - Click on first file to show it exists

2. Open one DICOM file with a hex editor or metadata viewer:

   ```bash
   python dump_dicom_metadata.py sample1/Brain\ 5\ mm/img001.dcm | head -20
   ```

   **Narrate:**
   - "You can see the PatientName, PatientID, StudyInstanceUID in clear text"
   - "This is PII that must be protected before sharing"

---

### Run Anonymization (3 minutes)

**Command to execute:**

```bash
python anonymizer_pro.py \
  -i sample1/Brain\ 5\ mm/ \
  -p GDPR-strict.json \
  --salt "SecureKey2026" \
  -o demo_output
```

**While the tool runs, narrate:**

- "The tool is now processing 36 DICOM files"
- "For each file, it will:"
  - "Remove/replace patient identifiable information"
  - "Generate deterministic pseudonyms"
  - "Log all changes to an audit file"
  - "Copy the de-identified image to output"
- "Notice: The tool never modifies the original files in sample1/"

**Expected runtime:** ~5-10 seconds (adjust narration based on actual speed)

---

### Inspect Output Structure (2 minutes)

After the tool completes, show the output structure:

```bash
ls -la demo_output/
ls -la demo_output/DICOM/
ls -la demo_output/DICOM/PX*/ | head -10
```

**Narrate:**

- "All output is in a timestamped folder"
- "The PatientID has been replaced with a pseudonym (PX + 6 hex chars)"
- "Notice the filename includes: Patient ID + Timestamp + Series + Image number"
- "This maintains the hierarchical structure for DICOM analysis"

---

### Examine Audit Log (2 minutes)

Display the anonymization log:

```bash
head -20 demo_output/anonymization_log.csv
```

Or open in a spreadsheet to show columns clearly:

```bash
cat demo_output/anonymization_log.csv | column -t -s ',' | less
```

**Narrate:**

- "Every transformation is recorded in the audit log"
- "Columns show: timestamp, input file, output file, patient ID mapping, tag name, old value, new value"
- "This provides compliance evidence for regulators or auditors"
- "Hospitals can verify exactly what was changed"

---

### Compare Before/After Metadata (2 minutes)

Show side-by-side comparison of original vs anonymized:

```bash
echo "=== ORIGINAL FILE ==="
python dump_dicom_metadata.py sample1/Brain\ 5\ mm/img001.dcm | grep -E "PatientName|PatientID|PatientBirthDate|StudyDate" | head -10

echo ""
echo "=== ANONYMIZED FILE ==="
python dump_dicom_metadata.py demo_output/DICOM/PX*/PX*_S002_I0001.dcm | grep -E "PatientName|PatientID|PatientBirthDate|StudyDate" | head -10
```

**Narrate:**

- "Original: Clear patient identification"
- "Anonymized: Patient name replaced with 'ANON', ID changed to pseudonym"
- "Birth dates removed, study dates removed (per GDPR-strict profile)"
- "Same data is usable for research, but without patient privacy risk"

---

## Optional Advanced Demonstrations

### Deterministic Pseudonymization

Show how the same SALT produces consistent results:

```bash
# Create a second export with same patient data and same SALT
python anonymizer_pro.py \
  -i sample1/Brain\ 5\ mm/img001.dcm \
  -p GDPR-strict.json \
  --salt "SecureKey2026" \
  -o demo_output_2

# Compare pseudonym - should be identical
ls demo_output/DICOM/ | head -1
ls demo_output_2/DICOM/ | head -1
```

**Narrate:**

- "Notice both folders have the same pseudonym: PX + same 6 characters"
- "This is key to our deterministic approach"
- "Same patient always gets the same pseudonym when using the same SALT"
- "Researcher can link multiple studies without exposing patient identity"

---

### Validate with Validator Script

(Optional if time permits)

```bash
python validator.py -i demo_output/DICOM/PX*/ -p GDPR-strict.json
```

**Narrate:**

- "The validator checks that our anonymized files match the profile requirements"
- "It verifies that sensitive tags were removed and PII was replaced appropriately"
- "This is useful for compliance audits"

---

## Cleanup

After the demo, clean up the demo output:

```bash
rm -rf demo_output demo_output_2
```

---

## Pro Tips for Presenters

### If the tool runs slowly

- Acknowledge that processing 36 images takes a few seconds
- Use the time to discuss the security implications of de-identification
- Mention that tool can process 1000s of images in batch mode

### If someone asks about pixel blackout

- Explain: "The GDPR-strict profile includes pixel blackout"
- Show in profile: `"PixelBlackout": true`
- Say: "This detects and masks any burned-in clinical text on images"

### If someone asks about different profiles

- Show the two JSON profiles in the repository
- Explain the trade-off: GDPR-strict (maximum privacy) vs research-pseudonymized (more data retained)

### If someone asks about re-identification risk

- Explain: "The SALT is kept secret. Without it, researcher cannot reverse the pseudonym"
- Say: "The same patient always gets the same pseudonym, but without the SALT phrase, the mapping cannot be broken"

---

## Expected Q&A

**Q: What if we lose the SALT?**
A: The pseudonym generation is deterministic but not reversible without the SALT. Losing the SALT means losing the ability to re-link pseudonymized data to the original patient, but the data itself remains fully de-identified.

**Q: How long does processing take for large datasets?**
A: Processing is linear. 100 images ≈ 30 seconds, 1000 images ≈ 5 minutes. Parallelization can be added to speed up batch processing.

**Q: Can we customize the anonymization rules?**
A: Yes! The profiles are simple JSON files. Users can copy and modify them to suit specific compliance requirements.

**Q: Is the audit log encrypted or secured?**
A: The audit log is a plain CSV file. In production, it should be stored in a secure location with appropriate access controls.

**Q: Can the pixel blackout be turned off?**
A: Yes, use the research-pseudonymized.json profile or modify the profile JSON. Set `"PixelBlackout": false`.

---

## Troubleshooting

### Tool hangs or runs very slowly

- **Cause:** Sample data may be corrupted or system IO-bound
- **Solution:** Kill with `Ctrl+C`, check sample data integrity with `dump_dicom_metadata.py`

### Audit log not created

- **Cause:** Output directory permissions or disk space
- **Solution:** Check write permissions on output directory, ensure at least 1GB free disk space

### Python module not found errors

- **Cause:** Virtual environment not activated
- **Solution:** Run `source .venv/bin/activate` before running the demo

---

## Presentation Talking Points

Use these during the demo to reinforce key messages:

- **Security:** "De-identified data is safe to share with research partners and collaborators"
- **Compliance:** "The audit log proves to regulators that we followed the anonymization protocol"
- **Reproducibility:** "Same SALT means we can re-anonymize new batches and maintain data linkage"
- **Flexibility:** "Different profiles for different use cases - strictest for public sharing, lenient for internal research"
- **Safety:** "Source files remain untouched. All changes are in a new timestamped folder"

---

## Post-Demo

After the demo, offer:

1. **Access to documentation:** Provide the Quick Reference card and Technical Appendix
2. **Repository access:** Share GitHub link for partners to access code and profiles
3. **Customization support:** Offer to create custom profiles for specific compliance needs
4. **Batch testing:** Offer to process a small sample of their own data for evaluation
