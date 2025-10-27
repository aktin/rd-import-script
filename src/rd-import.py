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
from functools import partial

import pandas as pd


#
#      Copyright (c) 2025  Alexander Ivanets, Markus Nissen
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
    einsatzdaten_df = import_csv_as_df(path_to_csv)
    check_df_for_mandatory_columns(
        einsatzdaten_df, ["einsatznummer", "einsatzart", "einsatzstichwort"]
    )
    confirm_clock_values(einsatzdaten_df)

    return einsatzdaten_df


def import_csv_as_df(path_to_csv):
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

    return einsatzdaten_df


def check_df_for_mandatory_columns(einsatzdaten_df, mandatory_columns):
    mandatory_columns = ["einsatznummer", "einsatzart", "einsatzstichwort"]
    missing_cols = []
    for col in mandatory_columns:
        if col not in einsatzdaten_df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_cols)}")


def confirm_clock_values(einsatzdaten_df):
    # Check if one of the clock values is on each row
    clock_columns = [
        "uhr_erstes_klingeln",
        "uhr_annahme",
        "uhr3",
        "uhr4",
        "uhr7",
        "uhr8",
        "uhr1",
        "uhr2",
    ]

    available_clock_cols = [
        col for col in clock_columns if col in einsatzdaten_df.columns
    ]

    if not available_clock_cols:
        raise ValueError(
            "No clock columns found in the CSV. At least one is required from the list."
        )

    clock_df_subset = einsatzdaten_df[available_clock_cols]

    missing_all_clocks = clock_df_subset.replace("", pd.NA).isna().all(axis=1)

    # Remove rows, there all clock
    if missing_all_clocks.any():
        num_bad_rows = missing_all_clocks.sum()
        first_bad_row_index = missing_all_clocks.idxmax()
        first_bad_row_number = first_bad_row_index + 1

        print(
            f"Found and removed {num_bad_rows} rows (starting from row {first_bad_row_number}) "
            "that were missing all available clock columns."
        )
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
            matched_column = (
                einsatzdaten_df[col]
                .astype(str)
                .apply(
                    lambda x: (
                        re.search(pattern, x).group(0)
                        if re.search(pattern, x)
                        else None
                    )
                )
            )
            # TODO: What to do if value not matching
            if len(matched_column) != len(einsatzdaten_df[col]):
                raise SystemExit(f"Value {col} does not match pattern {pattern}")


def find_earliest_timestamp(df):
    clock_columns = [
        "uhr_erstes_klingeln",
        "uhr_annahme",
        "uhr3",
        "uhr4",
        "uhr7",
        "uhr8",
        "uhr1",
        "uhr2",
    ]
    df_clocks = df[clock_columns].astype("datetime64[ns]")
    df["_metadata_start_date"] = df_clocks.min(axis=1).apply(
        lambda x: x.strftime("%Y%m%d%H%M%S") if pd.notna(x) else None
    )
    return df


def assign_instance_nummer(einsatzdaten_df):
    einsatzdaten_df["_metadata_start_date"] = pd.to_datetime(
        einsatzdaten_df["_metadata_start_date"]
    )
    einsatzdaten_df = einsatzdaten_df.sort_values(
        ["einsatznummer", "_metadata_start_date"]
    )
    einsatzdaten_df["_metadata_instance_num"] = (
        einsatzdaten_df.groupby("einsatznummer").cumcount() + 1
    )
    return einsatzdaten_df


def tval_transform(row, value, concept_cd):
    if pd.isna(value) or value == "":
        return None
    base = base_i2b2_row(row)
    base.update(
        {
            "concept_cd": concept_cd,
            "valtype": "T",
            "tval_char": value,
        }
    )
    return base


def code_transform(row, type, code):
    base = base_i2b2_row(row)
    concept_cd = f"AS:{type}" if code is None else f"AS:{type}:{code}"
    base.update(
        {
            "concept_cd": concept_cd,
        }
    )
    return base


def cd_transform(row, type, cd, tval_char):
    base = base_i2b2_row(row)
    base.update(
        {
            "concept_cd": f"AS:{type}",
            "modifier_cd": cd,
            "valtype": "T",
            "tval_char": tval_char,
        }
    )
    return base


def base_i2b2_row(row):
    """Build the common structure shared by all transforms."""
    return {
        "encounter_num": row["einsatznummer"],
        "patient_num": row["einsatznummer"],
        "provider_id": "@",
        "start_date": row["_metadata_start_date"],
        "instance_num": row.get("_metadata_instance_num", 1),
        "valtype": "",
        "tval_char": "",
        "nval_char": "",
        "valueflag_cd": "",
        "quantity_num": "",
        "units_cd": "",
        "end_date": "",
        "location_cd": "@",
        "observation_blob": "",
        "confidence_num": "",
        "update_date": "",
        "download_date": "",
        "import_date": "",
        "modifier_cd": "@",
    }


def transform_einsatzdaten(einsatzdaten_df):
    # Find the smallest timestamp along rows -> timestamp
    einsatzdaten_df = find_earliest_timestamp(einsatzdaten_df)
    # For each row assign appropriate instance_nummer
    einsatzdaten_df = assign_instance_nummer(einsatzdaten_df)

    transformers = {
        "einsatznummer": lambda row: tval_transform(row, row["einsatznummer"], "AS:ID"),
        "einsatzart": lambda row: code_transform(row, "TYPE", row["einsatzart"]),
        "einsatzstichwort": lambda row: tval_transform(
            row, row["einsatzstichwort"], "AS:KEYWORD"
        ),
        "einsatzstichwort_text": lambda row: tval_transform(
            row, row["einsatzstichwort"], "AS:KEYWORDTXT"
        ),
        "uhr_erstes_klingeln": lambda row: tval_transform(
            row, row["uhr_erstes_klingeln"], "AS:CLOCK_FIRSTRING"
        ),
        "uhr_annahme": lambda row: tval_transform(
            row, row["uhr_annahme"], "AS:CLOCK_ACCEPT"
        ),
        "einsatzort": lambda row: code_transform(row, "LOCATION", None),
        "einstzort_staat": lambda row: cd_transform(
            row, "LOCATION", "country", row["einsatzort_staat"]
        ),
        "einstzort_bundesland": lambda row: cd_transform(
            row, "LOCATION", "state", row["einsatzort_bundesland"]
        ),
        "einstzort_regierungsbezirk": lambda row: cd_transform(
            row, "LOCATION", "district", row["einsatzort_regierungsbezirk"]
        ),
        "einsatzort_region": lambda row: cd_transform(
            row, "LOCATION", "region", row["einsatzort_region"]
        ),
        "einsatzort_ort": lambda row: cd_transform(
            row, "LOCATION", "city", row["einsatzort_ort"]
        ),
        "einsatzort_ortsteil": lambda row: cd_transform(
            row, "LOCATION", "suburb", row["einsatzort_ortsteil"]
        ),
        "einsatzort_strasse": lambda row: cd_transform(
            row, "LOCATION", "street", row["einsatzort_strasse"]
        ),
        "einsatzort_hausnummer": lambda row: cd_transform(
            row, "LOCATION", "houseNummer", row["einsatzort_hausnummer"]
        ),
        "zielort": lambda row: code_transform(row, "DESTINATION", None),
        "zielort_staat": lambda row: cd_transform(
            row, "DESTINATION", "country", row["zielort_staat"]
        ),
        "zielort_bundesland": lambda row: cd_transform(
            row, "DESTINATION", "state", row["zielort_bundesland"]
        ),
        "zielort_regierungsbezirk": lambda row: cd_transform(
            row, "DESTINATION", "district", row["zielort_regierungsbezirk"]
        ),
        "zielort_region": lambda row: cd_transform(
            row, "DESTINATION", "region", row["zielort_region"]
        ),
        "zielort_ort": lambda row: cd_transform(
            row, "DESTINATION", "city", row["zielort_ort"]
        ),
        "zielort_ortsteil": lambda row: cd_transform(
            row, "DESTINATION", "suburb", row["zielort_ortsteil"]
        ),
        "zielort_strasse": lambda row: cd_transform(
            row, "DESTINATION", "street", row["zielort_strasse"]
        ),
        "zielort_hausnummer": lambda row: cd_transform(
            row, "DESTINATION", "houseNummer", row["zielort_hausnummer"]
        ),
        "zort_objekt": lambda row: cd_transform(
            row, "DESTINATION", "site", row["zort_objekt"]
        ),
        "diagnose": lambda row: tval_transform(row, row["diagnose"], "AS:DIAGNOSE"),
        "bemerkung": lambda row: tval_transform(row, row["bemerkung"], "AS:COMMENT"),
        "name": lambda row: tval_transform(row, row["name"], "AS:RESSOURCENAME"),
        "typ": lambda row: code_transform(row, "RESSOURCETYPE", row["typ"]),
        "uhralarm": lambda row: tval_transform(row, row["uhralarm"], "AS:CLOCK_ALERT"),
        "uhr3": lambda row: tval_transform(row, row["uhr3"], "AS:CLOCK3"),
        "uhr4": lambda row: tval_transform(row, row["uhr4"], "AS:CLOCK4"),
        "uhr7": lambda row: tval_transform(row, row["uhr7"], "AS:CLOCK7"),
        "uhr8": lambda row: tval_transform(row, row["uhr8"], "AS:CLOCK8"),
        "uhr1": lambda row: tval_transform(row, row["uhr1"], "AS:CLOCK1"),
        "uhr2": lambda row: tval_transform(row, row["uhr2"], "AS:CLOCK2"),
    }

    # Transform
    i2b2_df = dataframe_to_i2b2(einsatzdaten_df, transformers)
    print()


def dataframe_to_i2b2(df, transformers):
    results = []
    for _, row in df.iterrows():
        for func in transformers.values():
            transformed = func(row)
            if transformed:
                results.append(func(row))
    return pd.DataFrame(results)


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
    einsatzdaten_df = check_and_preprocess_csv_einsatzdaten(
        extract_dir / "einsatzdaten.csv"
    )

    # Validate dataframes
    validate_einsatzdaten(einsatzdaten_df)

    # Transform data for i2b2 format
    dict_dict_row = transform_einsatzdaten(einsatzdaten_df)
    print()
    # Load in database


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python rd-import.py <zip-file>")
    main(sys.argv[1])
