from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    stock_name: str | List[str] | None
    index_name: str | List[str] | None
    model_name: str | None
    train_start_date: str
    train_end_date: str
    root_path: str

    timesteps: int = 60
    num_features: int = 7
    pct_train: float = 0.8

    # Convolution Layer params
    conv_filters = 32
    conv_kernel_size = 3
    conv_activation = 'relu'

    # LSTM params
    units: int = 128
    batch_size: int = 32
    epochs: int = 20
    dropout: float = 0.2
    learning_rate: float = 1e-3

    # caching
    cache_data: bool = True
    cache_model: bool = True

    stock_data_path: Path = field(init=False)
    index_data_path: Path = field(init=False)
    model_path: Path = field(init=False)

    def __post_init__(self):
        root = Path(self.root_path)
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir = root / "model"
        model_dir.mkdir(parents = True, exist_ok = True)

        self.stock_data_path = data_dir
        self.index_data_path = data_dir
        self.model_path = model_dir / f"{self.model_name}.keras"