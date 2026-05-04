"""FAIR-compliance artifacts for the Speall MRI corpus.

Sub-modules
-----------
checksums         -- SHA-256 file-integrity manifest (build + verify)
provenance        -- W3C PROV-JSON provenance graph
datacite          -- DataCite Schema 4.5 metadata
datapackage       -- Frictionless Data Package envelope (datapackage.json)
validate_croissant -- Croissant 1.0 structural validator
"""

from . import checksums, datacite, datapackage, provenance, validate_croissant

__all__ = ["checksums", "datacite", "datapackage", "provenance", "validate_croissant"]
