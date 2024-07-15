from typing import NamedTuple


class _FEATURE_KEYS_NT(NamedTuple):
    # MS1 and MS2 shared features
    MASS_TO_CHARGE: str = "m/z"
    CHARGE: str = "charge"
    MASS: str = "mass"
    INTENSITY: str = "intensity"
    RETENTION_TIME: str = "retention_time"
    RETENTION_LENGTH: str = "retention_length"
    ION_MOBILITY_INDEX: str = "ion_mobility_index"
    ION_MOBILITY_LENGTH: str = "ion_mobility_length"
    NUMBER_OF_ISOTOPIC_PEAKS: str = "number_of_isotopic_peaks"
    # additional MS2 features
    SEQUENCE_ID: str = "sequence_id"
    PRECURSOR_ID: str = "precursor_id"


KEY = _FEATURE_KEYS_NT()
