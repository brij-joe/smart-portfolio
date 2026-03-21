from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    stock_name: str = "HDFCBANK.NS"
    index_list: list = None

    train_start_date: str = "2018-01-01"
    train_end_date: str = "2025-12-31"

    look_back: int = 60
    features: int = 7

    pct_train: float = 0.7

    units: int = 200
    batch_size: int = 1
    epochs: int = 20

    cache_data: bool = True
    cache_model: bool = True

    stock_data_path: str = f"{str(Path(__file__).resolve().parent.parent.parent)}/.cache/{stock_name}_{train_start_date}_{train_end_date}.pkl"
    index_data_path: str = f"{str(Path(__file__).resolve().parent.parent.parent)}/.cache/Index_{train_start_date}_{train_end_date}.pkl"
    model_path: str = f"{str(Path(__file__).resolve().parent.parent.parent)}/.cache/model_{train_start_date}_{train_end_date}.h5"

    def __post_init__(self):
        if self.index_list is None:
            self.index_list = ['GC=F', 'CL=F', '^BSESN', '^GSPC', '^TNX', '^FTSE']