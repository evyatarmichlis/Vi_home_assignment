import pandas as pd
from pathlib import Path
import os


class DataLoader:
    """
    Robust Data Loader for loading both Train and Test datasets.
    Centralizes configuration and handles file existence checks gracefully.
    """

    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir).resolve()

        # Central Configuration: Logical Name -> {Filename, Date Columns}
        # This is the Single Source of Truth.
        self.files_config = {
            'web_visits': {'filename': 'web_visits.csv', 'date_cols': ['timestamp']},
            'app_usage': {'filename': 'app_usage.csv', 'date_cols': ['timestamp']},
            'claims': {'filename': 'claims.csv', 'date_cols': ['diagnosis_date']},
            'churn_labels': {'filename': 'churn_labels.csv', 'date_cols': []}
        }

    def _load_single_file(self, dataset_name: str, is_test: bool = False) -> pd.DataFrame:
        """
        Internal helper to load a specific file based on mode (Train/Test).
        Handles path construction, existence checks, and date parsing.
        """
        if dataset_name not in self.files_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

        config = self.files_config[dataset_name]
        base_filename = config['filename']
        date_cols = config['date_cols']

        # Determine target filename with subdirectory structure
        if is_test:
            target_filename = Path('test') / f"test_{base_filename}"
        else:
            target_filename = Path('train') / base_filename

        file_path = self.data_dir / target_filename

        if not file_path.exists():
            print(f"    '{target_filename}' not found. Checking root directory fallback...")

            if is_test:
                # Fallback for test: try root/test_file.csv or root/file.csv
                fallback_path = self.data_dir / f"test_{base_filename}"
                if not fallback_path.exists():
                    fallback_path = self.data_dir / base_filename
            else:
                # Fallback for train: try root/file.csv
                fallback_path = self.data_dir / base_filename

            if fallback_path.exists():
                file_path = fallback_path
                print(f"    Found fallback file: {file_path.name}")
            else:
                # If neither exists, we can't load it
                raise FileNotFoundError(f"Could not find {target_filename} or valid fallback in {self.data_dir}")

        print(f"   Loading {dataset_name} from {file_path}...")

        # Load CSV with date parsing
        try:
            if date_cols:
                df = pd.read_csv(file_path, parse_dates=date_cols)
            else:
                df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"    Error reading {file_path.name}: {e}")
            raise e

    def get_train_data(self) -> dict:
        """Loads all training datasets."""
        print("\n Loading TRAIN Data...")
        return self._load_all(is_test=False)

    def get_test_data(self) -> dict:
        """Loads all test datasets."""
        print("\nLoading TEST Data...")
        return self._load_all(is_test=True)

    def _load_all(self, is_test: bool) -> dict:
        """Iterates through config and loads all available files."""
        data_dict = {}
        for name in self.files_config.keys():
            try:
                data_dict[name] = self._load_single_file(name, is_test=is_test)
            except FileNotFoundError:
                print(f"   Skipping {name} (File missing)")
            except Exception as e:
                print(f"    Skipping {name} (Error: {e})")
        return data_dict

    # Backwards compatibility wrapper (optional)
    def get_all_data(self) -> dict:
        return self.get_train_data()


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()

    # 1. Train
    train_data = loader.get_train_data()
    if 'web_visits' in train_data:
        print(f"Train Web Visits: {train_data['web_visits'].shape}")

    # 2. Test
    test_data = loader.get_test_data()
    if 'web_visits' in test_data:
        print(f"Test Web Visits: {test_data['web_visits'].shape}")