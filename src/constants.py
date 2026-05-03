"""DICOM constants — SOP classes, transfer syntaxes, required tags."""

SOP_CLASS_NAMES = {
    "1.2.840.10008.5.1.4.1.1.11.1": "Grayscale Softcopy PS",
    "1.2.840.10008.5.1.4.1.1.11.2": "Color Softcopy PS",
    "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.2": "CT Image Storage",
    "1.2.840.10008.5.1.4.1.1.7": "Secondary Capture",
    "1.2.840.10008.5.1.4.1.1.4.1": "Enhanced MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.66": "Raw Data Storage",
    "1.2.840.10008.5.1.4.1.1.66.4": "Segmentation Storage",
    "1.2.840.10008.5.1.1.1": "Basic Film Session",
}

TRANSFER_SYNTAX_NAMES = {
    "1.2.840.10008.1.2": "Implicit VR LE",
    "1.2.840.10008.1.2.1": "Explicit VR LE",
    "1.2.840.10008.1.2.2": "Explicit VR BE",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
    "1.2.840.10008.1.2.5": "RLE Lossless",
}

NON_IMAGE_SOP = {
    "1.2.840.10008.5.1.4.1.1.11.1",
    "1.2.840.10008.5.1.4.1.1.11.2",
    "1.2.840.10008.5.1.1.1",
}

REQUIRED_MR_TAGS = [
    "PatientID",
    "PatientName",
    "PatientSex",
    "PatientBirthDate",
    "StudyDate",
    "StudyTime",
    "StudyDescription",
    "StudyInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "SeriesInstanceUID",
    "Modality",
    "Manufacturer",
    "MagneticFieldStrength",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
    "BitsAllocated",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PhotometricInterpretation",
]
