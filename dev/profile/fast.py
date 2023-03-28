import cProfile
import pstats

import numpy as np
import pandas as pd

import skrough as rgh


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


df = golf_dataset()
x, x_counts, y, y_count = rgh.dataprep.prepare_factorized_data(df, target_attr="Play")


def run():
    for i in range(10000):
        rgh.checks.check_if_functional_dependency(x, y)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("./dev/profile/fast.pstats")
