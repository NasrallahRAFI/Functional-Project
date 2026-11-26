"""The raw training/validation adapter must reject locked test data."""
import pandas as pd
import pytest

from dronedet.cnn_v3.dataset import RawWaveformDataset
from dronedet.splits import TestAccessError as LockedSplitError


def test_raw_waveform_dataset_rejects_test_split_before_file_access():
    with pytest.raises(LockedSplitError, match="locked"):
        RawWaveformDataset(split="test", raw_root="missing", manifest=pd.DataFrame(),
                           config=None, mean=None, std=None)
