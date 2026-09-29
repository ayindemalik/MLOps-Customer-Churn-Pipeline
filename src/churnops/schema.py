"""The data contract: what a clean row must look like.

If the source system changes (a new contract type, a negative tenure, a
missing column), validation fails loudly here instead of silently
producing a worse model later.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column

YES_NO = ["Yes", "No"]
SERVICE = ["Yes", "No", "No internet service"]

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": Column(str, unique=True),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(YES_NO)),
        "Dependents": Column(str, Check.isin(YES_NO)),
        "tenure": Column(int, Check.in_range(0, 120)),
        "PhoneService": Column(str, Check.isin(YES_NO)),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(SERVICE)),
        "OnlineBackup": Column(str, Check.isin(SERVICE)),
        "DeviceProtection": Column(str, Check.isin(SERVICE)),
        "TechSupport": Column(str, Check.isin(SERVICE)),
        "StreamingTV": Column(str, Check.isin(SERVICE)),
        "StreamingMovies": Column(str, Check.isin(SERVICE)),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(YES_NO)),
        "PaymentMethod": Column(str, Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
        "MonthlyCharges": Column(float, Check.in_range(0, 1000)),
        "TotalCharges": Column(float, Check.ge(0)),
        "Churn": Column(int, Check.isin([0, 1])),
    },
    strict=True,   # an unexpected extra column is also an error
    coerce=True,
)
