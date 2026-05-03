---
license: mit
task_categories:
  - image-segmentation
  - image-classification
tags:
  - medical
  - mri
  - brain
  - dicom
  - neuroimaging
  - radiology
pretty_name: Speall MRI Brain Dataset
size_categories:
  - 10K<n<100K
language:
  - en
---

# Speall MRI Brain Dataset

A clinical brain MRI collection of **1,105 patient studies** containing **355,133 DICOM files** (~66 GB raw, **34,574 series**) acquired between December 2021 and October 2024. The corpus spans four scanner vendors (GE, Philips, Siemens, Toshiba) and both 3.0T and 1.5T field strengths, providing a realistic multi-vendor sample of routine neuroradiology practice. Every series is enriched with full DICOM tag extraction (240+ columns), volumetric statistics, automated quality grading (A–F), multiplane montages, intensity histograms, contrast-enhanced views, and a composite ML training score.

---

## Dataset Statistics

| Field | Value |
|---|---|
| Studies | 1,105 |
| Series | 34,574 |
| DICOM Files | 355,133 |
| Raw Volume | ~66 GB |
| Distinct Study Dates | 450 |
| Acquisition Period | December 2021 – October 2024 |
| DICOM Tags per File | 240+ |
| Patient Sex | M 51.6% / F 48.3% |

---

## Scanner Mix

| Vendor | Model | Field | Files | Share |
|---|---|---|---:|---:|
| GE | SIGNA Pioneer | 3.0T | 262,192 | 73.8% |
| Philips | Achieva | 1.5T | 38,713 | 10.9% |
| GE | SIGNA Explorer | 1.5T | 20,161 | 5.7% |
| GE | Signa HDxt | 1.5T | 9,154 | 2.6% |
| GE | SIGNA Creator | 1.5T | 8,378 | 2.4% |
| Siemens | MAGNETOM Essenza | 1.5T | 5,504 | 1.6% |
| GE | Optima MR360 | 1.5T | 4,804 | 1.4% |
| Philips | Achieva dStream | 1.5T | 2,588 | 0.7% |
| Philips | Spectra | 1.5T | 2,010 | 0.6% |
| Philips | Ingenia CX | 3.0T | 983 | 0.3% |
| Toshiba | FILMER 6.0 | 1.5T | 403 | 0.1% |
| Toshiba | MRT200SP3 | 1.5T | 216 | 0.1% |

**Aggregate:** GE 85.9%, Philips 11.9%, Siemens 2.1%, Toshiba 0.1% — 3.0T 79.7%, 1.5T 19.6%.

---

## Top Sequences

| # | Series Description | Series Count |
|---:|---|---:|
| 1 | Ax DSC Perfusion | 47,593 |
| 2 | FILT_PHA: 3D Ax SWAN | 21,514 |
| 3 | 3D Ax SWAN | 21,460 |
| 4 | 3D Ax T1 SPGR FS | 11,296 |
| 5 | 3D SAG T1 SPGR FS | 10,756 |
| 6 | DEFAULT PS SERIES | 9,917 |
| 7 | Ax DWI | 9,217 |
| 8 | Ax T2 FLAIR | 6,912 |
| 9 | Ax T2 PROPELLER | 5,603 |
| 10 | 3D Sag T1 BRAVO | 4,484 |
| 11 | 3D Ax T2 Cube HyperCube. | 4,460 |
| 12 | 3D Ax TOF NECK | 4,236 |

---

## Patient Demographics

- **Sex distribution (by file count):** Male 182,906 (51.6%), Female 171,515 (48.3%), Other/Unknown 712
- **Age range:** Includes subjects from infant (0.42 years) through elderly (87 years); sample ages observed: 0.42, 15, 16, 22, 22, 36, 37, 42, 43, 43, 53, 54, 58, 64, 64, 66, 70, 72, 85, 87
- **Geography:** India (single hospital network; institution names redacted)

---

## Acquisition Period

- **Start:** 2021-12-16
- **End:** 2024-10-31
- **Span:** ~2.9 years (450 distinct study dates)

---

## Per-Series Deliverables

Each series `{name}` ships four files plus one tar shard:

| File | Description |
|---|---|
| `{name}_detail.json` | Full metadata, quality grades, volume statistics, ML training score |
| `{name}_multiplane.png` | Three-plane montage (axial, coronal, sagittal) |
| `{name}_histogram.png` | Intensity distribution |
| `{name}_enhanced.png` | Contrast-enhanced view |
| `{name}.slices.tar` | Raw slice archive for the series |

Per-study deliverables include `cross_series_comparison.png` (multi-metric quality chart).

### ML Training Score

A 0–100 composite score is included in every `_detail.json`:

| Component | Weight |
|---|---:|
| CNR | 25% |
| Noise floor | 20% |
| Inter-slice consistency | 15% |
| Bias field severity | 15% |
| Edge sharpness | 15% |
| Histogram separation | 10% |

Grade bands: A (80–100, premium), B (65–79, standard), C (50–64, usable), D (35–49, limited), F (<35, exclude).

---

## De-identification

All DICOM files have been processed through a PHI-redaction pipeline. Patient names, identifiers, and institution names have been removed from DICOM headers prior to publication. Raw images are not included; only derived artifacts (JSON metadata, PNG montages, slice tars) are distributed.

---

## Citation

```bibtex
@dataset{speall_mri_2026,
  title={Speall MRI Brain Dataset: 1,105 Multi-Vendor Clinical Brain MRI Studies},
  author={Shubh},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/shubhxho/speall-mri},
  note={Acquired Dec 2021 -- Oct 2024 across GE, Philips, Siemens, Toshiba scanners}
}
```
