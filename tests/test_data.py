import pandera.errors
import pytest

from churnops.config import TARGET
from churnops.data import clean, split


def test_clean_fixes_blank_total_charges(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_clean_turns_target_into_0_and_1(raw_df):
    assert set(clean(raw_df)[TARGET].unique()) <= {0, 1}


def test_schema_rejects_negative_tenure(raw_df):
    raw_df.loc[0, "tenure"] = -5
    with pytest.raises(pandera.errors.SchemaError):
        clean(raw_df)


def test_schema_rejects_unknown_contract_type(raw_df):
    raw_df.loc[0, "Contract"] = "Three year"
    with pytest.raises(pandera.errors.SchemaError):
        clean(raw_df)


def test_split_keeps_every_customer_once(raw_df):
    df = clean(raw_df)
    train, val, test = split(df, 0.2, 0.2, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
    ids = set(train.customerID) | set(val.customerID) | set(test.customerID)
    assert len(ids) == len(df)   # no customer in two splits: no leakage
