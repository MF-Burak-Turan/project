from enum import Enum

SUPPORTED_FILE_EXTENSIONS = [".csv", ".parquet"]


class DataReadingErrorMessages(Enum):
    INVALID_FILE_EXTENSION = (
        f"Invalid file extension. Supported extensions are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
    )
    FILE_NOT_FOUND = "The specified file was not found."
    DATA_LOADING_ERROR = "An error occurred while loading the data."
    EMPTY_DATA = "The loaded data is empty."
    MISSING_REQUIRED_COLUMNS = "The data is missing required columns."
    DATA_TYPE_MISMATCH = "The data contains type mismatches."
    UNEXPECTED_ERROR = "An unexpected error occurred during data loading."
    INVALID_FILE_PATH = "The provided file path is invalid."
    
    