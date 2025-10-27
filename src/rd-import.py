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
            "source_col": "typ",
            "transform_type": "code",
            "concept_cd_base": "AS:RESSOURCETYPE",
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
    available_clocks = list(set(clock_columns) & set(df.columns))

    df_clocks = pd.DataFrame()
    for col in available_clocks:
        df_clocks[col] = pd.to_datetime(df[col], format="%Y%m%d%H%M%S", errors="coerce")

    min_timestamps = df_clocks.min(axis=1)
    df["_metadata_start_date"] = min_timestamps.dt.strftime("%Y%m%d%H%M%S")
    return df


def assign_instance_nummer(df, encounter_col, start_date_col):
    df[start_date_col] = pd.to_datetime(df[start_date_col])
    df = df.sort_values([encounter_col, start_date_col])

    instance_num_col = CONFIG["i2b2_key_columns"]["instance_num"]
    df[instance_num_col] = df.groupby(encounter_col).cumcount() + 1
    return df


def tval_transform(row, instruction, key_cols_map):
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
    """Build the common structure shared by all transforms."""
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
    check_df_for_mandatory_columns(df, file_config["mandatory_columns"])

    if file_config.get("integer_cleanup_columns"):
        log.info(f"Cleaning integer columns: {file_config['integer_cleanup_columns']}")
        df = clean_integer_strings(df, file_config["integer_cleanup_columns"])

    df = check_clock_values(df, file_config["clock_columns"])

    return df


def check_df_for_mandatory_columns(df, mandatory_columns):
    missing_cols = set(mandatory_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_cols)}")


def check_clock_values(df, clock_columns):
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
    key_cols = CONFIG["i2b2_key_columns"]
    transform_list = CONFIG["i2b2_transforms"]

    clock_cols = file_config["clock_columns"]

    df = find_earliest_timestamp(df, clock_cols)
    df = assign_instance_nummer(df, key_cols["encounter_num"], key_cols["start_date"])

    return dataframe_to_i2b2(df, transform_list, key_cols)


def load(transformed_df):
    pass


if __name__ == "__main__":
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
