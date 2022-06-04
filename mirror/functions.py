import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def apply_abs_eval(df: pd.DataFrame, ground_column: str, mirror_column: str):
    scored_column = df[ground_column] == df[mirror_column]
    scored_column = np.where(scored_column == True, 1, 0)

    return scored_column

def apply_sequence_matcher_eval(df: pd.DataFrame, ground_column: str, mirror_column: str):

    scored_column = df.apply(
        lambda row: SequenceMatcher(None, str(row[ground_column]), str(row[mirror_column])).ratio(),
        axis=1
    )

    return scored_column