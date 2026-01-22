import os
from pathlib import Path
import pandas as pd
import logging
from typing import Optional, Union
from .error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data_reader_functions = {".csv": pd.read_csv, ".parguet": pd.read_parquet}


class DataLoader:
    """"A class for loading data from CSV files"""
    
    def load_data(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Loads data from the CSV file into a pandas DataFrame.
        """
        self._validate_file_path(file_path)
        ext = self._check_if_file_extension_supported(file_path)
        
        reader_func = data_reader_functions.get(ext)
        data: pd.DataFrame = reader_func(file_path)
        
        if data.empty:
            logger.error(EM.EMPTY_DATA.value)
            raise ValueError(EM.EMPTY_DATA.value)
        
        return data
    
    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.
        """
        if not isinstance(file_path, (str, Path)):
            logger.error(EM.INVALID_FILE_PATH.value.format(type=type(file_path)))
            raise TypeError(
                EM.INVALID_FILE_PATH.value.format(type=type(file_path))
            )
            
        if not os.path.exists(file_path):
            logger.error(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
            raise FileNotFoundError(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
        
    def _check_if_file_extension_supported(self, file_path: Union[str, Path]) -> str:
        """
        Checks if the file extension is supported.
        """
        ext = Path(file_path).suffix
        
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            logger.error(
                EM.INVALID_FILE_EXTENSION.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )
            raise ValueError(
                EM.INVALID_FILE_EXTENSION.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )
                
            return ext