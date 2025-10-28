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
import logging
import sys
import os
from pathlib import Path
from functools import partial

import pandas as pd


"""
Rettungsdienst Import Script

This script processes zipped Rettungsdienst (emergency service) data,
validates CSV files, transforms them into i2b2-compatible format, and loads
the resulting dataset. It is designed to support AKTIN-style data imports.

Authors: Alexander Ivanets, Markus Nissen
License: GNU Affero General Public License v3.0
"""

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
CONFIG = {
    "files": {
        "einsatzdaten": {
            "filename": "einsatzdaten.csv",
            "mandatory_columns": ["einsatznummer", "einsatzart", "einsatzstichwort"],
            "clock_columns": [
                "uhr_erstes_klingeln",
                "uhr_annahme",
                "uhr3",
                "uhr4",
                "uhr7",
                "uhr8",
                "uhr1",
                "uhr2",
            ],
            "integer_cleanup_columns": ["einsatzort_hausnummer", "zielort_hausnummer"],
            "regex_patterns": {
                "einsatznummer": r"^1(2[3-9]|[3-9]\d)0\d{6}$",
                "einsatzart": r"^(A\d{2}|[AFHKLNPTUÜ])$",
                "uhr_erstes_klingeln": r"^\d{14}$",
                "uhr_annahme": r"^\d{14}$",
                "einsatzort_hausnummer": r"^\d+$",
                "zielort_hausnummer": r"^\d+$",
                "typ": r"^(?:$|\d+(?:-\d+)?|(?:KTW|RTW|RWT|NEF|LNA|LF|HLF|KLF|GW|HAB|PTLF|DLK|KdoW|GW-A|GW-TIER|MTF|KEF|ELW|PERSONAL)(?: ?\d+(?:-\d+)?)?)$",
                "uhralarm": r"^\d{14}$",
                "uhr3": r"^\d{14}$",
                "uhr4": r"^\d{14}$",
                "uhr7": r"^\d{14}$",
                "uhr8": r"^\d{14}$",
                "uhr1": r"^\d{14}$",
                "uhr2": r"^\d{14}$",
            },
        }
    },
    "i2b2_key_columns": {
        "encounter_num": "einsatznummer",
        "patient_num": "einsatznummer",
        "start_date": "_metadata_start_date",
        "instance_num": "_metadata_instance_num",
    },
    "i2b2_transforms": [
        # tval_transform instructions
        {
            "source_col": "einsatznummer",
            "transform_type": "tval",
            "concept_cd": "AS:ID",
        },
        {
            "source_col": "einsatzstichwort",
            "transform_type": "tval",
            "concept_cd": "AS:KEYWORD",
        },
        {
            "source_col": "einsatzstichwort_text",
            "transform_type": "tval",
            "concept_cd": "AS:KEYWORDTXT",
        },
        {
            "source_col": "uhr_erstes_klingeln",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_FIRSTRING",
        },
        {
            "source_col": "uhr_annahme",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_ACCEPT",
        },
        {
            "source_col": "diagnose",
            "transform_type": "tval",
            "concept_cd": "AS:DIAGNOSE",
        },
        {
            "source_col": "bemerkung",
            "transform_type": "tval",
            "concept_cd": "AS:COMMENT",
        },
        {
            "source_col": "name",
            "transform_type": "tval",
            "concept_cd": "AS:RESSOURCENAME",
        },
        {
            "source_col": "typ",
            "transform_type": "tval",
            "concept_cd": "AS:RESSOURCETYPE",
        },
        {
            "source_col": "uhralarm",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_ALERT",
        },
        {
            "source_col": "uhr3",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_3",
        },
        {
            "source_col": "uhr4",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_4",
        },
        {
            "source_col": "uhr7",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_7",
        },
        {
            "source_col": "uhr8",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_8",
        },
        {
            "source_col": "uhr1",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_1",
        },
        {
            "source_col": "uhr2",
            "transform_type": "tval",
            "concept_cd": "AS:CLOCK_2",
        },
        # code_transform instructions
        {
            "source_col": "einsatzart",
            "transform_type": "code",
            "concept_cd_base": "AS:TYPE",
        },
        {
            "source_col": None,
            "transform_type": "code",
            "concept_cd_base": "AS:LOCATION",
        },
        {
            "source_col": None,
            "transform_type": "code",
            "concept_cd_base": "AS:DESTINATION",
        },
        # cd_transform instructions
        {
            "source_col": "einstzort_staat",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "country",
        },
        {
            "source_col": "einstzort_bundesland",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "state",
        },
        {
            "source_col": "einsatzort_ort",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "city",
        },
        {
            "source_col": "einsatzort_ortsteil",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "suburb",
        },
        {
            "source_col": "einsatzort_strasse",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "street",
        },
        {
            "source_col": "einsatzort_hausnummer",
            "transform_type": "cd",
            "concept_cd": "AS:LOCATION",
            "modifier_cd": "houseNummer",
        },
        {
            "source_col": "zielort_staat",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "country",
        },
        {
            "source_col": "zielort_bundesland",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "state",
        },
        {
            "source_col": "zielort_ort",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "city",
        },
        {
            "source_col": "zielort_ortsteil",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "suburb",
        },
        {
            "source_col": "zielort_strasse",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "street",
        },
        {
            "source_col": "zielort_hausnummer",
            "transform_type": "cd",
            "concept_cd": "AS:DESTINATION",
            "modifier_cd": "houseNummer",
        },
    ],
}

# =============================================================================
# --- Script ---
# =============================================================================


def find_earliest_timestamp(df, clock_columns):
    """
    Determine the earliest timestamp for each row from all available clock columns in a DataFrame.

    Args:
        df: Input DataFrame containing time columns.
        clock_columns: List of column names representing timestamps.

    Returns:
        DataFrame with an additional column `_metadata_start_date` containing
        the earliest timestamp per row.
    """
    available_clocks = list(set(clock_columns) & set(df.columns))

    df_clocks = pd.DataFrame()
    for col in available_clocks:
        df_clocks[col] = pd.to_datetime(df[col], format="%Y%m%d%H%M%S", errors="coerce")

    min_timestamps = df_clocks.min(axis=1)
    df["_metadata_start_date"] = min_timestamps.dt.strftime("%Y%m%d%H%M%S")
    return df


def assign_instance_nummer(df, encounter_col, start_date_col):
    """
    Assign sequential instance numbers to encounters based on start time.

    Args:
        df: Input DataFrame.
        encounter_col: Column name representing encounter ID.
        start_date_col: Column name representing start date.

    Returns:
        DataFrame with a new column for instance numbering.
    """
    df[start_date_col] = pd.to_datetime(df[start_date_col])
    df = df.sort_values([encounter_col, start_date_col])

    instance_num_col = CONFIG["i2b2_key_columns"]["instance_num"]
    df[instance_num_col] = df.groupby(encounter_col).cumcount() + 1
    return df


def tval_transform(row, instruction, key_cols_map):
    """
    Transform a single row into a 'tval' i2b2 observation.

    Args:
        row: Input data row.
        instruction: Transformation instruction dict.
        key_cols_map: Mapping of i2b2 key columns.

    Returns:
        Dictionary for an i2b2 tval observation, or None if no value.
    """
    value = row.get(instruction["source_col"])
    if pd.isna(value) or value == "":
        return None

    base = base_i2b2_row(row, key_cols_map)
    base.update(
        {
            "concept_cd": instruction["concept_cd"],
            "valtype": "T",
            "tval_char": value,
        }
    )
    return base


def code_transform(row, instruction, key_cols_map):
    """
    Transform a row into a 'code' i2b2 observation.

    Args:
        row: Data row.
        instruction: Transformation instruction dict.
        key_cols_map: Mapping of i2b2 key columns.

    Returns:
        i2b2 observation dictionary.
    """
    code = row.get(instruction["source_col"]) if instruction["source_col"] else None
    concept_cd_base = instruction["concept_cd_base"]

    base = base_i2b2_row(row, key_cols_map)
    concept_cd = (
        f"{concept_cd_base}"
        if (pd.isna(code) or code == "")
        else f"{concept_cd_base}:{code}"
    )
    base.update(
        {
            "concept_cd": concept_cd,
        }
    )
    return base


def cd_transform(row, instruction, key_cols_map):
    """
    Transform a row into a 'cd' i2b2 observation (concept + modifier).

    Args:
        row: Data row.
        instruction: Transformation instruction dict.
        key_cols_map: Mapping of i2b2 key columns.

    Returns:
        i2b2 observation dictionary or None.
    """
    tval_char = row.get(instruction["source_col"])
    if pd.isna(tval_char) or tval_char == "":
        return None

    base = base_i2b2_row(row, key_cols_map)
    base.update(
        {
            "concept_cd": instruction["concept_cd"],
            "modifier_cd": instruction["modifier_cd"],
            "valtype": "T",
            "tval_char": tval_char,
        }
    )
    return base


def base_i2b2_row(row, key_cols_map):
    """
    Construct a base i2b2 observation row structure.

    Args:
        row: Source pandas row.
        key_cols_map: Mapping of i2b2 key column names to source columns.

    Returns:
        Dictionary representing an i2b2 base observation row.
    """
    return {
        "encounter_num": row.get(key_cols_map["encounter_num"]),
        "patient_num": row.get(key_cols_map["patient_num"]),
        "provider_id": "@",
        "start_date": row.get(key_cols_map["start_date"]),
        "instance_num": row.get(key_cols_map.get("instance_num"), 1),
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


TRANSFORM_DISPATCHER = {
    "tval": tval_transform,
    "code": code_transform,
    "cd": cd_transform,
}


def dataframe_to_i2b2(df, instructions_list, key_cols_map):
    """
    Apply transformation instructions to all rows in a DataFrame.

    Args:
        df: Input DataFrame.
        instructions_list: List of transformation instruction dicts.
        key_cols_map: Mapping of i2b2 key columns.

    Returns:
        Transformed DataFrame containing i2b2 facts.
    """
    results = []
    for _, row in df.iterrows():
        for instruction in instructions_list:
            transform_func = TRANSFORM_DISPATCHER.get(instruction["transform_type"])

            if not transform_func:
                log.warning(f"Unknown transform_type: {instruction['transform_type']}")
                continue

            transformed = transform_func(row, instruction, key_cols_map)
            if transformed:
                results.append(transformed)
    return pd.DataFrame(results)


def extract_zip(zip_path):
    """
    Extract a ZIP file into a temporary directory.

    Args:
        zip_path: Path to ZIP file.

    Returns:
        Path to the extraction directory.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If extraction fails.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Error: file {zip_path} does not exist")

    temp_dir = Path(tempfile.gettempdir())
    extract_dir = temp_dir / zip_path.stem

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            log.info(f"Extracting {zip_path} to {extract_dir}...")
            zip_file.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Error: file {zip_path} does not contain a zip file") from e
    except Exception as e:
        raise RuntimeError(f"Error: an unexpected error occurred") from e

    return extract_dir


def main(zip_path):
    """
    Main entry point for the import process.

    Args:
        zip_path: Path to ZIP file containing Rettungsdienst data.
    """
    log.info(f"Starting import for {zip_path}")
    extract_dir = extract_zip(zip_path)

    for _, file_config in CONFIG["files"].items():
        filename = file_config["filename"]
        log.info(f"Processing file: {filename}...")

        file_path = extract_dir / filename

        df = extract(file_path)
        df = preprocess(df, file_config)
        validate_dataframe(df, file_config["regex_patterns"])

        transformed_i2b2_data = transform_dataframe(df, file_config)

        log.info(f"Loading {len(transformed_i2b2_data)} i2b2 facts...")
        load(transformed_i2b2_data)
        log.info(f"Successfully loaded data for {filename}.")


def extract(filepath):
    """
    Read a CSV file into a DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with CSV contents.

    Raises:
        ValueError: If the file is empty.
    """
    try:
        einsatzdaten_df = pd.read_csv(filepath, sep=";", dtype=str)

        if einsatzdaten_df.empty:
            raise ValueError("CSV file is empty.")

    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError:
        raise
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    return einsatzdaten_df


def preprocess(df, file_config):
    """
    Preprocess input DataFrame before transformation.

    - Checks mandatory columns.
    - Cleans integer-like columns.
    - Removes rows missing all clock values.

    Args:
        df: Input DataFrame.
        file_config: File configuration dict.

    Returns:
        Cleaned DataFrame.
    """
    check_df_for_mandatory_columns(df, file_config["mandatory_columns"])

    if file_config.get("integer_cleanup_columns"):
        log.info(f"Cleaning integer columns: {file_config['integer_cleanup_columns']}")
        df = clean_integer_strings(df, file_config["integer_cleanup_columns"])

    df = check_clock_values(df, file_config["clock_columns"])

    return df


def check_df_for_mandatory_columns(df, mandatory_columns):
    """
    Ensure mandatory columns are present in DataFrame.

    Args:
        df: DataFrame to check.
        mandatory_columns: List of required column names.

    Raises:
        ValueError: If any columns are missing.
    """
    missing_cols = set(mandatory_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_cols)}")


def check_clock_values(df, clock_columns):
    """
    Remove rows missing all clock columns.

    Args:
        df: Input DataFrame.
        clock_columns: List of time columns to check.

    Returns:
        Filtered DataFrame.
    """
    available_clock_cols = [col for col in clock_columns if col in df.columns]

    clock_df_subset = df[available_clock_cols]
    missing_all_clocks = clock_df_subset.replace("", pd.NA).isna().all(axis=1)

    if missing_all_clocks.any():
        num_bad_rows = missing_all_clocks.sum()
        first_bad_row_index = missing_all_clocks.idxmax()
        first_bad_row_number = first_bad_row_index + 1

        log.warning(
            f"Found and removed {num_bad_rows} rows (starting from row {first_bad_row_number}) "
            "that were missing all available clock columns."
        )
        df = df[~missing_all_clocks].reset_index(drop=True)

    return df


def clean_integer_strings(df, cols_to_clean):
    """
    Clean numeric strings by removing decimal separators and text.

    Args:
        df: Input DataFrame.
        cols_to_clean: List of column names to clean.

    Returns:
        Cleaned DataFrame.
    """
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda x: re.split(r"[,\.]", x)[0])
                .replace("nan", "")
            )
    return df


def validate_dataframe(df, regex_patterns):
    """
    Validate DataFrame columns against regex patterns.

    Args:
        df: Input DataFrame.
        regex_patterns: Dict of column names to regex patterns.

    Raises:
        ValueError: If validation fails for any column.
    """
    for col, pattern in regex_patterns.items():
        if col in df.columns:
            series_to_check = df[col].fillna("").astype(str)

            is_empty = series_to_check == ""

            matches_pattern = series_to_check.str.match(pattern, na=False)

            is_valid = is_empty | matches_pattern

            if not is_valid.all():
                bad_rows_mask = ~is_valid
                bad_rows = df[bad_rows_mask]

                num_bad_rows = len(bad_rows)
                first_bad_val = bad_rows.iloc[0][col]

                log.error(
                    f"Validation failed for column '{col}'. Found {num_bad_rows} "
                    f"non-matching, non-empty rows. Example: '{first_bad_val}'"
                )
                raise ValueError(f"Validation failed for column '{col}'.")


def transform_dataframe(df, file_config):
    """
    Apply all transformation steps to convert DataFrame into i2b2 format.

    Args:
        df: Cleaned input DataFrame.
        file_config: File configuration dict.

    Returns:
        Transformed i2b2 DataFrame.
    """
    key_cols = CONFIG["i2b2_key_columns"]
    transform_list = CONFIG["i2b2_transforms"]

    clock_cols = file_config["clock_columns"]

    df = find_earliest_timestamp(df, clock_cols)
    df = assign_instance_nummer(df, key_cols["encounter_num"], key_cols["start_date"])

    return dataframe_to_i2b2(df, transform_list, key_cols)


def load(transformed_df):
    pass


# For testing purposes
def load_env():
    """
    Loads environment variables from a .env file if it exists.
    This is a basic parser and doesn't handle all .env syntax.
    """
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"Info: Found '{env_path}' file, loading environment variables.")
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()

                        if (value.startswith("'") and value.endswith("'")) or (
                            value.startswith('"') and value.endswith('"')
                        ):
                            value = value[1:-1]

                        os.environ[key] = value
        except Exception as e:
            print(f"Warning: Could not parse .env file. Error: {e}", file=sys.stderr)


if __name__ == "__main__":

    load_env()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python rd-import.py <zip-file>")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("rd-import.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    log = logging.getLogger(__name__)

    main(sys.argv[1])
