# -*- coding: utf-8 -*
# Created on Wed Oct 22 13:00 2025
# @VERSION=1.1
# @VIEWNAME=Rettungsdienst Importscript (TOML)
# @MIMETYPE=zip
# @ID=rd
"""

      Copyright (c) 2025  AKTIN

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU Affero General Public License as
      published by the Free Software Foundation, either version 3 of the
      License, or (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU Affero General Public License for more details.

      You should have received a copy of the GNU Affero General Public License
      along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import base64
import hashlib
import logging
import os
import re
import sys
import tempfile
import json
import zipfile
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Any

import pandas as pd
import sqlalchemy as db
from sqlalchemy import exc


# =============================================================================
# --- CONFIGURATION LOADING ---
# =============================================================================

def load_config(config_path: str = "config.json") -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Try looking one directory up
        parent_path = os.path.join("..", config_path)
        if os.path.exists(parent_path):
            with open(parent_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON configuration: {e}")


# =============================================================================
# --- Script ---
# =============================================================================


def get_earliest_timestamp_per_row(
        timestamp_df: pd.DataFrame,
        date_format: str = "%Y%m%d%H%M%S"
) -> pd.Series:
    """
    Parses a DataFrame of timestamp strings and returns the earliest
    timestamp for each row as a datetime object.
    """
    dt_df = timestamp_df.apply(
        lambda col: pd.to_datetime(col, format=date_format, errors="coerce")
    )

    return dt_df.min(axis=1)


def assign_instance_number(df: pd.DataFrame, encounter_col: str, start_date_col: str) -> pd.DataFrame:
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


def tval_transform(row: dict, instruction: dict, key_cols_map: dict) -> dict | None:
    """
    Transform a single row into a 'tval' i2b2 observation.
    """
    source_col = instruction.get("source_col")

    if not source_col:
        return None

    value = row.get(source_col)
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


def code_transform(row: dict, instruction: dict, key_cols_map: dict) -> dict:
    """
    Transform a row into a 'code' i2b2 observation.
    """
    source_col = instruction.get("source_col")  # Safe access for TOML compatibility

    code = row.get(source_col) if source_col else None
    concept_cd_base = instruction["concept_cd_base"]

    base = base_i2b2_row(row, key_cols_map)
    concept_cd = (
        f"{concept_cd_base}"
        if (pd.isna(code) or code == "")
        else f"{concept_cd_base}:{code}"
    )
    base.update({"concept_cd": concept_cd, "valtype": "@", "valueflag_cd": "@"})
    return base


def cd_transform(row: dict, instruction: dict, key_cols_map: dict) -> dict | None:
    """
    Transform a row into a 'cd' i2b2 observation (concept + modifier). Handle metadata.
    """
    source_col = instruction.get("source_col")  # Safe access for TOML compatibility

    if not source_col:
        return None

    tval_char = row.get(source_col)
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


def metadata_cd_transform(row: dict, instruction: dict, key_cols_map: dict) -> dict | None:
    """
    Generates a 'cd' observation from environment variables,
    but attaches it to the current row's encounter.
    """
    if instruction["modifier_cd"] == "scriptId":
        value = os.getenv("uuid")
    elif instruction["modifier_cd"] == "scriptVersion":
        value = os.getenv("script_version")
    else:
        log.warning(f"Unknown metadata modifier: {instruction['modifier_cd']}")
        return None

    if not value:
        log.warning(f"Environment variable for {instruction['modifier_cd']} not set.")
        return None

    base = base_i2b2_row(row, key_cols_map)
    base.update(
        {
            "concept_cd": instruction["concept_cd"],
            "modifier_cd": instruction["modifier_cd"],
            "valtype": "T",
            "tval_char": value,
        }
    )
    return base


def base_i2b2_row(row: dict, key_cols_map: dict) -> dict:
    return {
        "encounter_num": row.get(key_cols_map["encounter_num"]),
        "patient_num": row.get(key_cols_map["patient_num"]),
        "provider_id": "@",
        "start_date": row.get(key_cols_map["start_date"]),
        "modifier_cd": "@",
        "instance_num": row.get(key_cols_map.get("instance_num"), 1),
        "valtype": "",
        "tval_char": "",
        "valueflag_cd": "",
        "units_cd": "@",
        "location_cd": "@",
        "observation_blob": "",
        "update_date": "",
        "import_date": "",
    }


def parse_json_transformations(config: dict) -> list[dict]:
    """
    Converts the concise JSON mapping into the standard list format.
    """
    mapping = config.get("transformations", {})
    statics = config.get("static_concepts", [])
    instructions = []

    for col_name, instruction in mapping.items():

        # 1. Simple String -> 'tval'
        if isinstance(instruction, str):
            instructions.append({
                "transform_type": "tval",
                "source_col": col_name,
                "concept_cd": instruction
            })

        # 2. Object/Dict -> Complex types
        elif isinstance(instruction, dict):
            t_type = instruction.get("type")

            if t_type == "code":
                instructions.append({
                    "transform_type": "code",
                    "source_col": col_name,
                    "concept_cd_base": instruction.get("concept")
                })

            elif t_type == "cd":
                instructions.append({
                    "transform_type": "cd",
                    "source_col": col_name,
                    "concept_cd": instruction.get("concept"),
                    "modifier_cd": instruction.get("mod")
                })

            elif t_type == "metadata":
                # Metadata doesn't use a CSV column, but for consistency
                instructions.append({
                    "transform_type": "metadata_cd",
                    "source_col": None,
                    "concept_cd": instruction.get("concept"),
                    "modifier_cd": instruction.get("mod")
                })

    # 3. Static Concepts
    for concept in statics:
        instructions.append({
            "transform_type": "code",
            "source_col": None,
            "concept_cd_base": concept
        })

    return instructions

def dataframe_to_i2b2(df: pd.DataFrame, instructions_list: list, key_cols_map: dict) -> pd.DataFrame:
    """
    Apply transformation instructions to all rows in a DataFrame.
    """
    dispatcher = {
        "tval": tval_transform,
        "code": code_transform,
        "cd": cd_transform,
        "metadata_cd": metadata_cd_transform,
    }
    results = []
    for row in df.itertuples(index=False):
        row_dict = dict(zip(df.columns, row))

        for instruction in instructions_list:
            transform_func = dispatcher.get(instruction["transform_type"])
            if not transform_func:
                log.warning(f"Unknown transform_type: {instruction['transform_type']}")
                continue

            transformed = transform_func(row_dict, instruction, key_cols_map)
            if transformed:
                results.append(transformed)

    return pd.DataFrame(results)


def extract_zip_into_tmp_dir(zip_path: str) -> Path:
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

    return extract_dir


def get_sourcesystem_cd_from_zip(filepath: str, prefix: str = "AS:") -> str | None:
    """
    Calculates the SHA-256 hash of a file, encodes it in
    URL-safe Base64, and prepends the given prefix.

    This format is secure and fits within a VARCHAR(50) field.
    (3-char prefix + 44-char hash = 47 chars)
    """

    sha256_hash = hashlib.sha256()

    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096 * 1024), b""):
                sha256_hash.update(byte_block)

        hash_bytes = sha256_hash.digest()
        base64_hash_bytes = base64.urlsafe_b64encode(hash_bytes)
        base64_hash_str = base64_hash_bytes.decode("utf-8")
        final_hash = base64_hash_str.rstrip("=")
        final_sourcesystem_cd = f"{prefix}{final_hash}"

        return final_sourcesystem_cd

    except FileNotFoundError:
        log.error(f"Error: file {filepath} does not exist")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        return None


def main(zip_path: str) -> None:
    if not CONFIG:
        log.error("Configuration not loaded. Aborting.")
        return

    log.info(f"Starting import for {zip_path}")
    extract_dir = extract_zip_into_tmp_dir(zip_path)

    for _, file_config in CONFIG["files"].items():
        filename = file_config["filename"]
        log.info(f"Processing file: {filename}...")

        file_path = extract_dir / filename

        df = load_csv_into_df(file_path)
        df = preprocess(df, file_config)
        validate_dataframe(df, file_config["regex_patterns"])

        transformed_i2b2_data = transform_dataframe(df, file_config)

        transformed_i2b2_data = add_general_i2b2_info(transformed_i2b2_data, zip_path)
        transformed_i2b2_data = convert_values_to_i2b2_format(transformed_i2b2_data)

        log.info(f"Loading {len(transformed_i2b2_data)} i2b2 facts...")
        load(transformed_i2b2_data)

        log.info(f"Successfully loaded data for {filename}.")


def convert_values_to_i2b2_format(df: pd.DataFrame) -> pd.DataFrame:
    date_columns = ["start_date", "update_date", "import_date"]
    result_df = df.copy()
    for column in date_columns:
        result_df[column] = (
            result_df[column]
            .astype(str)
            .apply(
                lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )
    return result_df


def add_general_i2b2_info(df: pd.DataFrame, zip_path: str) -> pd.DataFrame:
    result_df = df.copy()
    result_df["update_date"] = pd.Timestamp.now()
    result_df["import_date"] = pd.Timestamp.now()
    result_df["sourcesystem_cd"] = get_sourcesystem_cd_from_zip(zip_path)
    return result_df


def load_csv_into_df(filepath: PosixPath) -> pd.DataFrame:
    try:
        einsatzdaten_df = pd.read_csv(filepath, sep=";", dtype=str)

        if einsatzdaten_df.empty:
            raise ValueError("CSV file is empty.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: file {filepath} does not exist")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    return einsatzdaten_df


def preprocess(df: pd.DataFrame, file_config: dict) -> pd.DataFrame:
    """
    Preprocess input DataFrame before transformation.

    - Checks mandatory columns.
    - Cleans integer-like columns.
    - Removes rows missing all clock values.
    """
    check_df_for_mandatory_columns(df, file_config["mandatory_columns"])

    if file_config.get("integer_cleanup_columns"):
        log.info(f"Cleaning integer columns: {file_config['integer_cleanup_columns']}")
        # df = clean_integer_strings(df, file_config["integer_cleanup_columns"])

    df = check_clock_values(df, file_config["clock_columns"])

    return df


def check_df_for_mandatory_columns(df: pd.DataFrame, mandatory_columns: list) -> None:
    missing_cols = set(mandatory_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_cols)}")


def check_clock_values(df: pd.DataFrame, clock_columns: list) -> pd.DataFrame:
    """
    Remove rows missing all clock columns (handles empty strings and whitespace).
    """
    available_clock_cols = [col for col in clock_columns if col in df.columns]

    subset = df[available_clock_cols].replace(r"^\s*$", pd.NA, regex=True)

    missing_all_clocks = subset.isna().all(axis=1)

    if missing_all_clocks.any():
        num_bad_rows = missing_all_clocks.sum()
        first_bad_row_index = missing_all_clocks.idxmax()

        log.warning(
            f"Found and removed {num_bad_rows} rows (starting from index {first_bad_row_index}) "
            "that were missing all available clock columns."
        )
        df = df.loc[~missing_all_clocks].reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame, regex_patterns: dict) -> None:
    """
    Validate DataFrame columns against regex patterns.
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


def transform_dataframe(df: pd.DataFrame, file_config: dict) -> pd.DataFrame:
    key_cols = CONFIG["i2b2_key_columns"]
    transform_list = parse_json_transformations(CONFIG)

    df["_metadata_start_date"] = get_earliest_timestamp_per_row(df)
    df = assign_instance_number(df, key_cols["encounter_num"], key_cols["start_date"])

    return dataframe_to_i2b2(df, transform_list, key_cols)


def load(transformed_df: pd.DataFrame) -> None:
    # establish database conncetion
    USERNAME = os.environ["username"]
    PASSWORD = os.environ["password"]
    I2B2_CONNECTION_URL = os.environ["connection-url"]
    pattern = r"jdbc:postgresql://(.*?)(\?searchPath=.*)?$"
    connection = re.search(pattern, I2B2_CONNECTION_URL).group(1)
    ENGINE = db.create_engine(
        f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{connection}", pool_pre_ping=True
    )
    conn = ENGINE.connect()
    TABLE = db.Table("observation_fact", db.MetaData(), autoload_with=ENGINE)

    delete_from_db(conn, TABLE, transformed_df)
    upload_into_db(conn, TABLE, transformed_df)

    # cut db connection
    conn.close()
    ENGINE.dispose()


def delete_from_db(conn, TABLE, transformed_df):
    """
    delete existing combinations of encounter_nums/start_date/concept_cd from TABLE
    """
    transaction = conn.begin()
    try:
        unique_combinations = (
            transformed_df[["encounter_num", "start_date", "concept_cd"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        if unique_combinations:
            for row in unique_combinations:
                encounter = row["encounter_num"]
                # TODO: Remove function, because already done in previous step
                # date_i2b2 = convert_date_to_i2b2_format(str(row["start_date"]))
                date_i2b2 = row["start_date"]
                concept = row["concept_cd"]

                statement = (
                    TABLE.delete()
                    .where(TABLE.c["encounter_num"] == encounter)
                    .where(TABLE.c["start_date"] == date_i2b2)
                )
                statement = (
                    statement.where(TABLE.c["concept_cd"] == concept)
                    if concept
                    else statement
                )

                conn.execute(statement)

        transaction.commit()
    except exc.SQLAlchemyError as e:
        transaction.rollback()


def upload_into_db(conn: db.Connection, table: db.Table, transformed_df: pd.DataFrame) -> None:
    """
    load all dataframe lines into table
    """
    insert_transaction = conn.begin()
    try:
        temp = 5000
        for i in range(0, len(transformed_df), temp):
            stapel = transformed_df.iloc[i: i + temp]
            records = stapel.to_dict(orient="records")
            if records:
                conn.execute(table.insert(), records)
        insert_transaction.commit()
    except exc.SQLAlchemyError as e:
        insert_transaction.rollback()


def convert_date_to_i2b2_format(date: str) -> str:
    if len(date) > 19:
        date = date[:19]
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")


# For testing purposes
def load_env() -> None:
    env_path = os.path.join("../local", ".env")
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
        raise SystemExit("Usage: python rd_import.py <zip-file>")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("rd-import.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    log = logging.getLogger(__name__)

    try:
        CONFIG = load_config()
    except Exception as e:
        print(f"CRITICAL: {e}", file=sys.stderr)
        CONFIG = {}

    main(sys.argv[1])
