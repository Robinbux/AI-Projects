import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple
from pathlib import Path
from src_clean.dataloader import DataLoader

class EDA:

    @staticmethod
    def create_bar_plot():
        pass

if __name__ == "__main__":
    train_df, test_df = DataLoader.load_train_test_dfs()
