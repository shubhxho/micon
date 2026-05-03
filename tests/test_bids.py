"""Tests for src/bids/ -- BIDS converter package.

Covers:
  1. SEQUENCE_TO_BIDS has an entry for every sequence_type seen in the sample study.
  2. bids_filename produces BIDS-regex-compliant filenames.
  3. infer_acquisition_label correctly maps orientation prefixes.
  4. convert_study on Speall_MRI_Samples produces the expected BIDS directory layout.
     (NIfTI conversion is mocked -- placeholders are written when DICOMs are absent.)
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from src.bids.converter import convert_study
from src.bids.mappings import (
    SEQUENCE_TO_BIDS,
    bids_filename,
    infer_acquisition_label,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All sequence_type values actually emitted by the sample study series JSONs.
SAMPLE_SEQUENCE_TYPES = [
    "DWI",
    "FLAIR",
    "SWI (SWAN)",
    "T2-weighted",
    "T1-weighted",
    "TOF MRA",
    "ADC Map",
    "CUBE (3D FSE)",
    "Color MIP",
    "Projection MIP",
    "Processed/MIP",
]

# BIDS entity label: only [A-Za-z0-9]+
_BIDS_LABEL_RE = re.compile(r"^[A-Za-z0-9]+$")

# Full BIDS filename regex (permissive -- at minimum sub + ses + suffix + ext)
_BIDS_FILENAME_RE = re.compile(
    r"^sub-[A-Za-z0-9]+"  # sub entity
    r"_ses-[A-Za-z0-9]+"  # ses entity
    r"(?:_acq-[A-Za-z0-9]+)?"  # optional acq
    r"(?:_run-[A-Za-z0-9]+)?"  # optional run
    r"_[A-Za-z0-9]+"  # suffix
    r"\.[A-Za-z0-9.]+$"  # extension(s)
)


# ---------------------------------------------------------------------------
# 1. SEQUENCE_TO_BIDS coverage
# ---------------------------------------------------------------------------


class TestSequenceToBids:
    def test_all_sample_sequence_types_mapped(self) -> None:
        """Every sequence_type the classifier emits must be in SEQUENCE_TO_BIDS."""
        missing = [st for st in SAMPLE_SEQUENCE_TYPES if st not in SEQUENCE_TO_BIDS]
        assert missing == [], f"Missing from SEQUENCE_TO_BIDS: {missing}"

    def test_values_are_two_tuples(self) -> None:
        for seq_type, value in SEQUENCE_TO_BIDS.items():
            assert isinstance(value, tuple), f"{seq_type}: expected tuple, got {type(value)}"
            assert len(value) == 2, f"{seq_type}: expected 2-tuple, got length {len(value)}"

    def test_bids_modality_dirs_are_valid(self) -> None:
        valid_dirs = {"anat", "dwi", "swi", "derivatives"}
        for seq_type, (mod_dir, _suffix) in SEQUENCE_TO_BIDS.items():
            assert mod_dir in valid_dirs, f"{seq_type}: unexpected modality dir {mod_dir!r}"

    def test_dwi_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["DWI"] == ("dwi", "dwi")

    def test_flair_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["FLAIR"] == ("anat", "FLAIR")

    def test_t1w_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["T1-weighted"] == ("anat", "T1w")

    def test_t2w_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["T2-weighted"] == ("anat", "T2w")

    def test_swi_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["SWI (SWAN)"] == ("swi", "swi")

    def test_tof_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["TOF MRA"] == ("anat", "angio")

    def test_adc_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["ADC Map"] == ("dwi", "ADC")

    def test_cube_mapping(self) -> None:
        assert SEQUENCE_TO_BIDS["CUBE (3D FSE)"] == ("anat", "T2w")

    def test_derivative_types_map_to_derivatives(self) -> None:
        for seq_type in ("Color MIP", "Projection MIP", "Processed/MIP"):
            mod_dir, _ = SEQUENCE_TO_BIDS[seq_type]
            assert mod_dir == "derivatives", f"{seq_type} should map to derivatives dir"


# ---------------------------------------------------------------------------
# 2. bids_filename
# ---------------------------------------------------------------------------


class TestBidsFilename:
    def test_basic_no_optional_entities(self) -> None:
        name = bids_filename("001", "01", "T1w")
        assert name == "sub-001_ses-01_T1w.nii.gz"

    def test_with_acq_and_run(self) -> None:
        name = bids_filename("001", "01", "T1w", acq="axial", run=1)
        assert name == "sub-001_ses-01_acq-axial_run-1_T1w.nii.gz"

    def test_custom_extension(self) -> None:
        name = bids_filename("001", "01", "dwi", ext=".bval")
        assert name.endswith(".bval")

    def test_matches_bids_regex(self) -> None:
        names = [
            bids_filename("001", "01", "T1w"),
            bids_filename("001", "01", "T2w", acq="axial"),
            bids_filename("001", "01", "dwi", acq="axial", run=1),
            bids_filename("MEMAR2329", "01", "FLAIR"),
        ]
        for name in names:
            assert _BIDS_FILENAME_RE.match(name), f"{name!r} does not match BIDS filename regex"

    def test_subject_id_sanitized(self) -> None:
        name = bids_filename("MEMAR 2329", "01", "T1w")
        # Spaces must not appear in sub label
        assert "sub-MEMAR2329" in name

    def test_no_none_entities_in_output(self) -> None:
        name = bids_filename("001", "01", "T2w", acq=None, run=None)
        assert "acq" not in name
        assert "run" not in name

    def test_entity_order(self) -> None:
        name = bids_filename("001", "01", "T1w", acq="sagittal", run=2)
        parts = name.replace(".nii.gz", "").split("_")
        assert parts[0].startswith("sub-")
        assert parts[1].startswith("ses-")
        assert parts[2].startswith("acq-")
        assert parts[3].startswith("run-")
        assert parts[4] == "T1w"


# ---------------------------------------------------------------------------
# 3. infer_acquisition_label
# ---------------------------------------------------------------------------


class TestInferAcquisitionLabel:
    @pytest.mark.parametrize(
        "desc,expected",
        [
            ("Ax DWI", "axial"),
            ("Ax T2 FLAIR", "axial"),
            ("AxT1 MEMP", "axial"),
            ("3D Ax SWAN", "axial"),
            ("COR DWI", "coronal"),
            ("SAG T2", "sagittal"),
            (" SAG T2", "sagittal"),
            ("BRAIN ANGIO", None),
            ("ADC (10^-6 mm²/s)", None),
            ("Processed Images", None),
        ],
    )
    def test_orientation_detection(self, desc: str, expected: str | None) -> None:
        assert infer_acquisition_label(desc) == expected

    def test_returns_none_for_unknown(self) -> None:
        assert infer_acquisition_label("Unknown Protocol XYZ") is None

    def test_result_is_bids_label_safe(self) -> None:
        for desc in ["Ax DWI", "SAG T2", "COR DWI"]:
            label = infer_acquisition_label(desc)
            if label is not None:
                assert _BIDS_LABEL_RE.match(label), f"{label!r} is not BIDS-label-safe"


# ---------------------------------------------------------------------------
# 4. convert_study layout test
# ---------------------------------------------------------------------------


class TestConvertStudyLayout:
    """Test that convert_study produces the expected BIDS directory layout.

    DICOMs are absent from Speall_MRI_Samples/series/ so the converter
    writes placeholder .nii.gz files, which is the expected code path.
    We test layout, not pixel content.
    """

    @pytest.fixture()
    def bids_out(self, tmp_path: Path) -> Path:
        return tmp_path / "bids"

    @pytest.fixture()
    def sample_study_dir(self) -> Path:
        return Path(__file__).parent.parent / "Speall_MRI_Samples"

    def test_converts_without_error(self, sample_study_dir: Path, bids_out: Path) -> None:
        result = convert_study(
            study_dir=sample_study_dir,
            bids_root=bids_out,
            subject_id="001",
            session="01",
        )
        assert isinstance(result, dict)
        assert result["series_converted"] > 0

    def test_dataset_description_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001")
        desc_file = bids_out / "dataset_description.json"
        assert desc_file.exists()
        desc = json.loads(desc_file.read_text())
        assert desc["BIDSVersion"] == "1.10.0"
        assert desc["DatasetType"] == "raw"

    def test_readme_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001")
        readme = bids_out / "README"
        assert readme.exists()
        assert "BIDS" in readme.read_text()

    def test_participants_tsv_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001")
        tsv = bids_out / "participants.tsv"
        assert tsv.exists()
        lines = tsv.read_text().splitlines()
        assert lines[0] == "participant_id\tage_bracket\tsex\tsite"
        assert any("sub-001" in line for line in lines[1:])

    def test_participants_json_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001")
        pjson = bids_out / "participants.json"
        assert pjson.exists()
        data = json.loads(pjson.read_text())
        assert "participant_id" in data

    def test_subject_ses_directories_exist(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        sub_dir = bids_out / "sub-001"
        assert sub_dir.exists()

    def test_nifti_files_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        nifti_files = list(bids_out.rglob("*.nii.gz"))
        assert len(nifti_files) > 0, "No .nii.gz files found in BIDS output"

    def test_nifti_files_are_readable_gzip(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        for nifti_path in bids_out.rglob("*.nii.gz"):
            with gzip.open(nifti_path, "rb") as fh:
                header_bytes = fh.read(4)
            # NIfTI-1: first 4 bytes should decode to 348 (sizeof_hdr)
            import struct

            sizeof_hdr = struct.unpack("<i", header_bytes)[0]
            assert sizeof_hdr == 348, f"{nifti_path}: invalid NIfTI-1 sizeof_hdr={sizeof_hdr}"

    def test_json_sidecars_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        json_files = list(bids_out.rglob("sub-*.json"))
        assert len(json_files) > 0, "No sidecar JSON files found"

    def test_json_sidecar_contains_required_fields(
        self, sample_study_dir: Path, bids_out: Path
    ) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        required = {
            "SeriesDescription",
            "_SourceFiles",
            "Manufacturer",
            "ManufacturerModelName",
        }
        for jf in bids_out.rglob("sub-*.json"):
            data = json.loads(jf.read_text())
            missing = required - data.keys()
            assert not missing, f"{jf}: missing required sidecar fields: {missing}"

    def test_json_sidecar_magnetic_field_strength(
        self, sample_study_dir: Path, bids_out: Path
    ) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        for jf in bids_out.rglob("sub-*.json"):
            data = json.loads(jf.read_text())
            if "MagneticFieldStrength" in data:
                assert isinstance(data["MagneticFieldStrength"], (int, float)), (
                    f"{jf}: MagneticFieldStrength must be numeric"
                )

    def test_dwi_bval_bvec_written(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        bval_files = list(bids_out.rglob("*.bval"))
        bvec_files = list(bids_out.rglob("*.bvec"))
        assert len(bval_files) > 0, "No .bval files found"
        assert len(bvec_files) > 0, "No .bvec files found"

    def test_anat_directory_contains_t2w(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        t2w_files = list(bids_out.rglob("*T2w.nii.gz"))
        assert len(t2w_files) > 0, "No T2w NIfTI files found in anat/"

    def test_derivatives_written_for_mip_series(
        self, sample_study_dir: Path, bids_out: Path
    ) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        deriv_dir = bids_out / "derivatives"
        assert deriv_dir.exists(), "derivatives/ directory not created"
        deriv_desc = bids_out / "derivatives" / "speall-mips" / "dataset_description.json"
        assert deriv_desc.exists(), "derivatives/speall-mips/dataset_description.json missing"

    def test_bids_filenames_valid(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "001", session="01")
        for nifti in bids_out.rglob("*.nii.gz"):
            assert _BIDS_FILENAME_RE.match(nifti.name), (
                f"{nifti.name!r} does not match BIDS filename pattern"
            )

    def test_subject_id_with_spaces_sanitized(self, sample_study_dir: Path, bids_out: Path) -> None:
        convert_study(sample_study_dir, bids_out, "MEMAR 2329", session="01")
        sub_dir = bids_out / "sub-MEMAR2329"
        assert sub_dir.exists(), "Subject directory with sanitized ID not found"
