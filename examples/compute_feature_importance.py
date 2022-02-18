# %%
import more_itertools
import numpy as np
import pandas as pd

from skrough.chaos_measures.chaos_measures import entropy, gini_impurity
from skrough.chaos_score import get_chaos_score
from skrough.dataprep import prepare_df
from skrough.feature_importance import get_feature_importance

# %%
if __name__ == "__main__":
    df = pd.DataFrame(
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
    target_column = "Play"
    x, x_count_distinct, y, y_count_distinct = prepare_df(df, "Play")
    column_names = np.array([col for col in df.columns if col != target_column])

    print(df)
    print()

    for attrs in [[0], [0, 1], [0, 1, 3]]:
        for chaos_function in [gini_impurity, entropy]:
            print(
                f"chaos score for attrs {attrs}({column_names[attrs]}) "
                f"using `{chaos_function.__name__}` chaos function = "
                f"""{get_chaos_score(x, x_count_distinct,
                                   y, y_count_distinct,
                                   attrs, chaos_fun=chaos_function)}"""
            )

    inputs_collection = [
        [[0, 2], [0, 3], [0], [2, 3], [1, 2, 3]],
        [[0], [0, 1], [1, 2]],
        list(more_itertools.powerset(range(4))),
    ]
    for input in inputs_collection:
        for chaos_function in [gini_impurity, entropy]:
            print(
                f"\nfeature importance computed using `{chaos_function.__name__}` "
                f" chaos function for attribute sets: {input}"
            )
            print(
                get_feature_importance(
                    x,
                    x_count_distinct,
                    y,
                    y_count_distinct,
                    column_names,
                    input,
                    chaos_fun=chaos_function,
                )
            )
