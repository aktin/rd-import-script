# -*- coding: utf-8 -*
# Created on Wed Oct 22 13:00 2025
# @VERSION=1.0
# @VIEWNAME=Rettungsdienst Importscript
# @MIMETYPE=zip
# @ID=rd

#
#      Copyright (c) 2025  Alexander Ivanets
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU Affero General Public License as
#      published by the Free Software Foundation, either version 3 of the
#      License, or (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU Affero General Public License for more details.
#
#      You should have received a copy of the GNU Affero General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#

import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

# TODO: Split function (too heavy)
def check_and_preprocess_csv_einsatzdaten(path_to_csv):
    """
    Validates an 'Einsatzdaten' CSV file based on specific column requirements.
    The function raises an Exception if validation fails.

    Checks for:
    1. File readability and format (semicolon-separated).
    2. Presence of mandatory columns: ["einsatznummer", "einsatzart", "einsatzstichwort"].
    3. For every row, at least one of the specified clock columns must have a value.

    Args:
        path_to_csv (str): The file path to the CSV to be checked.

    Returns:
        void (None): The function does not return a value. It raises an exception on failure.

    Raises:
        FileNotFoundError: If the file is not found.
        pd.errors.EmptyDataError: If the file is empty or malformed.
        ValueError: For specific validation errors (missing columns, invalid rows, empty file).
        Exception: For general read errors.
    """
    try:
        einsatzdaten_df = pd.read_csv(path_to_csv, sep=";", dtype=str)

        if einsatzdaten_df.empty:
            raise ValueError("CSV file is empty.")

    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError:
        raise
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    # Check mandatory columns
    mandatory_columns = ["einsatznummer", "einsatzart", "einsatzstichwort"]
    missing_cols = []
    for col in mandatory_columns:
        if col not in einsatzdaten_df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_cols)}")

    # Check if one of the clock values is on each row
    clock_columns = [
        "uhr_erstes_klingeln", "uhr_annahme", "uhr3", "uhr4",
        "uhr7", "uhr8", "uhr1", "uhr2"
    ]

    available_clock_cols = [col for col in clock_columns if col in einsatzdaten_df.columns]

    if not available_clock_cols:
        raise ValueError("No clock columns found in the CSV. At least one is required from the list.")

    clock_df_subset = einsatzdaten_df[available_clock_cols]

    missing_all_clocks = clock_df_subset.replace('', pd.NA).isna().all(axis=1)

    # Remove rows, there all clock
    if missing_all_clocks.any():
        num_bad_rows = missing_all_clocks.sum()
        first_bad_row_index = missing_all_clocks.idxmax()
        first_bad_row_number = first_bad_row_index + 1

        print(f"Found and removed {num_bad_rows} rows (starting from row {first_bad_row_number}) "
                        "that were missing all available clock columns.")
        einsatzdaten_df = einsatzdaten_df[~missing_all_clocks].reset_index(drop=True)

    return einsatzdaten_df


def main(zip_path):
    # Extract zip files in tmp folder
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Error: file {zip_path} does not exist")

    temp_dir = Path(tempfile.gettempdir())
    extract_dir = temp_dir / zip_path.stem

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Error: file {zip_path} does not contain a zip file") from e
    except Exception as e:
        raise RuntimeError(f"Error: an unexpected error occurred") from e

    # Check csv files
    einsatzdaten_df = check_and_preprocess_csv_einsatzdaten(extract_dir / "einsatzdaten.csv")

    # Transform each column to observation_fact


    # Load in database

    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python rd-import.py <zip-file>")
    main(sys.argv[1])
