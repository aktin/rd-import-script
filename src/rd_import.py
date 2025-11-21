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
import json
import logging
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Any

import pandas as pd
import sqlalchemy as db
from sqlalchemy import exc, tuple_


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
    key_cols = file_config["i2b2_key_columns"]
    transform_list = parse_json_transformations(file_config)

    df["_metadata_start_date"] = get_earliest_timestamp_per_row(df)
    df = assign_instance_number(
        df, key_cols["encounter_num"], key_cols["start_date"], file_config
    )

    return dataframe_to_i2b2(df, transform_list, key_cols)


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
            instructions.append(
                {
                    "transform_type": "tval",
                    "source_col": col_name,
                    "concept_cd": instruction,
                }
            )

        # 2. Object/Dict -> Complex types
        elif isinstance(instruction, dict):
            t_type = instruction.get("type")

            if t_type == "code":
                instructions.append(
                    {
                        "transform_type": "code",
                        "source_col": col_name,
                        "concept_cd_base": instruction.get("concept"),
                    }
                )

            elif t_type == "cd":
                instructions.append(
                    {
                        "transform_type": "cd",
                        "source_col": col_name,
                        "concept_cd": instruction.get("concept"),
                        "modifier_cd": instruction.get("mod"),
                    }
                )

            elif t_type == "metadata":
                # Metadata doesn't use a CSV column, but for consistency
                instructions.append(
                    {
                        "transform_type": "metadata_cd",
                        "source_col": None,
                        "concept_cd": instruction.get("concept"),
                        "modifier_cd": instruction.get("mod"),
                    }
                )

    # 3. Static Concepts
    for concept in statics:
        instructions.append(
            {"transform_type": "code", "source_col": None, "concept_cd_base": concept}
        )

    return instructions


def get_earliest_timestamp_per_row(
        timestamp_df: pd.DataFrame, date_format: str = "%Y%m%d%H%M%S"
) -> pd.Series:
    """
    Parses a DataFrame of timestamp strings and returns the earliest
    timestamp for each row as a datetime object.
    """
    dt_df = timestamp_df.apply(
        lambda col: pd.to_datetime(col, format=date_format, errors="coerce")
    )

    return dt_df.min(axis=1)


def assign_instance_number(
        df: pd.DataFrame, encounter_col: str, start_date_col: str, file_config: dict
) -> pd.DataFrame:
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

    instance_num_col = file_config["i2b2_key_columns"]["instance_num"]
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


def metadata_cd_transform(
        row: dict, instruction: dict, key_cols_map: dict
) -> dict | None:
    """
    Generates a 'cd' observation from environment variables,
    but attaches it to the current row's encounter.
    """
    if instruction["modifier_cd"] == "scriptId":
        value = os.getenv("script_id")
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


def dataframe_to_i2b2(
        df: pd.DataFrame, instructions_list: list, key_cols_map: dict
) -> pd.DataFrame:
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
    result_df["sourcesystem_cd"] = "AS:" + os.getenv("uuid")
    return result_df


def load(transformed_df: pd.DataFrame) -> None:
    # Establish database connection
    username = os.environ["username"]
    password = os.environ["password"]
    i2b2_connection_url = os.environ["connection-url"]
    pattern = r"jdbc:postgresql://(.*?)(\?searchPath=.*)?$"
    connection = re.search(pattern, i2b2_connection_url).group(1)
    engine = db.create_engine(
        f"postgresql+psycopg2://{username}:{password}@{connection}", pool_pre_ping=True
    )
    with engine.connect() as conn:
        table = db.Table("observation_fact", db.MetaData(), autoload_with=engine)
        delete_from_db(conn, table, transformed_df)
        upload_into_db(conn, table, transformed_df)


def delete_from_db(conn, TABLE, transformed_df):
    """
    Bulk deletes existing records from TABLE based on encounter_num and concept_cd
    within the 'AS' source system scope.
    """
    keys_to_delete = (
        transformed_df[["encounter_num", "concept_cd"]]
        .drop_duplicates()
        .values.tolist()
    )

    if not keys_to_delete:
        log.info("No records to delete.")
        return

    with conn.begin() as transaction:
        try:
            stmt = (
                TABLE.delete()
                .where(TABLE.c.sourcesystem_cd.like("AS%"))
                .where(
                    tuple_(TABLE.c.encounter_num, TABLE.c.concept_cd).in_(
                        keys_to_delete
                    )
                )
            )

            result = conn.execute(stmt)
            log.info(f"Successfully deleted {result.rowcount} rows from database.")

            transaction.commit()

        except exc.SQLAlchemyError as e:
            transaction.rollback()
            log.error(
                f"Database error occurred. Transaction rolled back. Database state preserved. Error: {e}"
            )
            raise e


def upload_into_db(conn, table, transformed_df, batch_size=5000):
    """
    Loads dataframe rows into the database table in batches.
    """

    with conn.begin() as transaction:
        total_rows = len(transformed_df)

        try:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = start_idx + batch_size
                batch_df = transformed_df.iloc[start_idx:end_idx]

                records = batch_df.to_dict(orient="records")

                if records:
                    conn.execute(table.insert(), records)
                    log.debug(
                        f"Inserted batch {start_idx}-{min(end_idx, total_rows)} of {total_rows}"
                    )

            transaction.commit()
            log.info(f"Successfully uploaded {total_rows} records to database.")

        except exc.SQLAlchemyError as e:
            transaction.rollback()
            log.error(f"Database insert failed. Transaction rolled back. Error: {e}")

            raise e
        except Exception as e:
            transaction.rollback()
            log.error(
                f"Unexpected error during upload processing. Transaction rolled back. Error: {e}"
            )
            raise e


# For testing purposes
def load_env() -> None:
    env_path = os.path.join("../local", ".env")
    if os.path.exists(env_path):
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
            log.error(f"Warning: Could not parse .env file. Error: {e}", file=sys.stderr)


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
