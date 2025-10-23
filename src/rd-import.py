# -*- coding: utf-8 -*
# Created on Wed Oct 22 13:00 2025
# @VERSION=1.0
# @VIEWNAME=Rettungsdienst Importscript
# @MIMETYPE=zip
# @ID=rd
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


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


def validate_einsatzdaten(einsatzdaten_df):
    dict_column_pattern = {
        "einsatznummer": r"^1(2[3-9]|[3-9]\d)0\d{6}$",
        "einsatzart": r"^(A\d{2}|[AFHKLNPTUÜ])$",
        "uhr_erstes_klingeln": r"^\d{14}$",
        "uhr_annahme": r"^\d{14}$",
        "einsatzort_hausnummer": r"^\d+$",
        "zielort_hausnummer": r"^\d+$",
        "typ": r"^(KTW|RWT|NEF|LNA|LF|HLF|KLF|GW|LF|HAB|PTLF|DLK|KdoW|GW-A|GW-TIER|MTF|KEF|ELW)$",
        "uhralarm": r"^\d{14}$",
        "uhr3": r"^\d{14}$",
        "uhr4": r"^\d{14}$",
        "uhr7": r"^\d{14}$",
        "uhr8": r"^\d{14}$",
        "uhr1": r"^\d{14}$",
        "uhr2": r"^\d{14}$",
    }
    for col, pattern in dict_column_pattern.items():
        if col in einsatzdaten_df.columns:
            matched_column = einsatzdaten_df[col].astype(str).apply(
                lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None
            )
            # TODO: What to do if value not matching
            if len(matched_column) != len(einsatzdaten_df[col]):
                raise SystemExit(f"Value {col} does not match pattern {pattern}")

def find_earliest_timestamp(df):
    clock_columns = [
        "uhr_erstes_klingeln", "uhr_annahme", "uhr3", "uhr4",
        "uhr7", "uhr8", "uhr1", "uhr2"
    ]
    df_clocks = df[clock_columns].astype("datetime64[ns]")
    df["start_date"] = df_clocks.min(axis=1).apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
    return df


def create_i2b2_row():
    pass

def transform_einsatzdaten(einsatzdaten_df):
    # Find the smallest timestamp along rows -> timestamp
    einsatzdaten_df = find_earliest_timestamp(einsatzdaten_df)
    transformation_rules = {
        "einsatznummer": ""
    }


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

    # Validate dataframes
    validate_einsatzdaten(einsatzdaten_df)

    # Transform data for i2b2 format
    dict_dict_row = transform_einsatzdaten(einsatzdaten_df)
    # Load in database


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python rd-import.py <zip-file>")
    main(sys.argv[1])
