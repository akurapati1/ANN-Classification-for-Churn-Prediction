import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from churn_features import build_features, load_preprocessors  # noqa: E402

CUSTOMER = dict(geography="France", gender="Female", age=42, tenure=3, balance=60000.0, credit_score=619,
                estimated_salary=101348.88, num_of_products=1, has_cr_card=1, is_active_member=1)


@pytest.fixture(scope="module")
def pre():
    return load_preprocessors()


def test_feature_shape(pre):
    assert build_features(CUSTOMER, *pre).shape == (1, 12)


def test_geography_changes_only_geo_columns(pre):
    a = build_features(CUSTOMER, *pre)
    b = build_features({**CUSTOMER, "geography": "Germany"}, *pre)
    assert np.allclose(a[0, :9], b[0, :9])
    assert not np.allclose(a[0, 9:], b[0, 9:])
