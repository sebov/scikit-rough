import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def golf_dataset():
    result = pd.DataFrame(
        np.array(
            [
                ["sunny", "hot", "high", "weak", "no"],
                ["sunny", "hot", "high", "strong", "no"],
                ["overcast", "hot", "high", "weak", "yes"],
                ["rain", "mild", "high", "weak", "yes"],
                ["rain", "cool", "normal", "weak", "yes"],
                ["rain", "cool", "normal", "strong", "no"],
                ["overcast", "cool", "normal", "strong", "yes"],
                ["sunny", "mild", "high", "weak", "no"],
                ["sunny", "cool", "normal", "weak", "yes"],
                ["rain", "mild", "normal", "weak", "yes"],
                ["sunny", "mild", "normal", "strong", "yes"],
                ["overcast", "mild", "high", "strong", "yes"],
                ["overcast", "hot", "normal", "weak", "yes"],
                ["rain", "mild", "high", "strong", "no"],
            ],
            dtype=object,
        ),
        columns=["Outlook", "Temperature", "Humidity", "Wind", "Play"],
    )
    return result
