# -*- coding: utf-8 -*
# Created on Wed Oct 22 13:00 2025
# @VERSION=1.1
# @VIEWNAME=Rettungsdienst Importscript
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
import os
import re
import sys
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as db
from sqlalchemy import tuple_


# =============================================================================
# --- CONFIGURATION LOADING ---
# =============================================================================


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Config file {config_path} not found.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON configuration: {e}")


# =============================================================================
# --- Script ---
# =============================================================================


def find_matching_file(search_dir: Path, mandatory_columns: list) -> Path | None:
    for file in search_dir.glob("*.csv"):
        if not file.is_file():
            continue
        try:
            df_header = pd.read_csv(file, sep=";", encoding="utf-8", dtype=str)
        except Exception as e:
            warnings.warn(f"Skipping file {file}: {e}")
            continue
        if set(mandatory_columns).issubset(df_header.columns):
            return file
    return None


def main(zip_path: str) -> None:
    config = load_config()

    extract_dir = extract_zip_into_tmp_dir(zip_path)
    file_path = find_matching_file(
        extract_dir, mandatory_columns=config["mandatory_columns"]
    )

    df = load_csv_into_df(file_path)
    df = preprocess(df, config)
    df = validate_dataframe(df, config["regex_patterns"])

    transformed_i2b2_data = transform_dataframe(df, config)

    transformed_i2b2_data = add_general_i2b2_info(transformed_i2b2_data, zip_path)
    transformed_i2b2_data = convert_values_to_i2b2_format(transformed_i2b2_data)

    delete_duplicate_entries_and_upload_into_db(transformed_i2b2_data)


def extract_zip_into_tmp_dir(zip_path: str) -> Path:
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise SystemExit(f"Error: file {zip_path} does not exist")

    extract_dir = Path(tempfile.mkdtemp(prefix=f"{zip_path.stem}_"))

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise SystemExit(f"Error: file {zip_path} does not contain a zip file") from e
    except Exception as e:
        raise SystemExit(f"Error: an unexpected error occurred: {e}") from e

    return extract_dir


def load_csv_into_df(filepath: Path) -> pd.DataFrame:
    try:
        einsatzdaten_df = pd.read_csv(filepath, sep=";", dtype=str)

        if einsatzdaten_df.empty:
            raise ValueError("CSV file is empty.")

    except FileNotFoundError:
        raise SystemExit(f"Error: file {filepath} does not exist")
    except Exception as e:
        raise SystemExit(f"Error reading CSV: {e}")

    return einsatzdaten_df


def preprocess(df: pd.DataFrame, file_config: dict) -> pd.DataFrame:
    check_df_for_mandatory_columns(df, file_config["mandatory_columns"])

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

        warnings.warn(
            f"Found and removed {num_bad_rows} rows (starting from index {first_bad_row_index}) "
            "that were missing all available clock columns."
        )
        df = df.loc[~missing_all_clocks].reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame, regex_patterns: dict) -> pd.DataFrame:
    """
    Validate DataFrame columns against regex patterns.
    Drops rows containing invalid values
    """
    df_clean = df.copy()

    for col, pattern in regex_patterns.items():
        if col in df_clean.columns:
            series_to_check = df_clean[col].fillna("").astype(str)

            is_empty = series_to_check == ""
            matches_pattern = series_to_check.str.match(pattern)

            is_valid = is_empty | matches_pattern

            if not is_valid.all():
                bad_rows_mask = ~is_valid
                num_bad_rows = bad_rows_mask.sum()

                first_bad_val = df_clean.loc[bad_rows_mask, col].iloc[0]

                warnings.warn(
                    f"Column '{col}': Found {num_bad_rows} invalid rows. "
                    f"Dropping them. Example invalid value: '{first_bad_val}'"
                )

                df_clean = df_clean[is_valid]

    return df_clean


def transform_dataframe(df: pd.DataFrame, file_config: dict) -> pd.DataFrame:
    key_cols = file_config["i2b2_key_columns"]
    transform_list = parse_json_transformations(file_config)

    df["_metadata_start_date"] = get_earliest_timestamp_per_row(df)
    df = assign_instance_number(
        df, key_cols["encounter_num"], key_cols["start_date"], file_config
    )

    return dataframe_to_i2b2(df, transform_list, key_cols)


def _handle_tval(col_name, instruction):
    return {
        "transform_type": "tval",
        "source_col": col_name,
        "concept_cd": instruction,
    }


def _handle_code(col_name, instruction):
    return {
        "transform_type": "code",
        "source_col": col_name,
        "concept_cd_base": instruction.get("concept"),
    }


def _handle_cd(col_name, instruction):
    return {
        "transform_type": "cd",
        "source_col": col_name,
        "concept_cd": instruction.get("concept"),
        "modifier_cd": instruction.get("mod"),
    }


def _handle_metadata(col_name, instruction):
    return {
        "transform_type": "metadata_cd",
        "source_col": None,
        "concept_cd": instruction.get("concept"),
        "modifier_cd": instruction.get("mod"),
    }


def parse_json_transformations(config: dict) -> list[dict]:
    mapping = config.get("transformations", {})
    statics = config.get("static_concepts", [])
    instructions = []

    transformation_handlers = {
        "code": _handle_code,
        "cd": _handle_cd,
        "metadata": _handle_metadata,
    }

    for col_name, instruction in mapping.items():
        # Handle simple string case
        if isinstance(instruction, str):
            instructions.append(_handle_tval(col_name, instruction))
            continue

        # Handle dict case via Registry
        if isinstance(instruction, dict):
            t_type = instruction.get("type")
            handler = transformation_handlers.get(t_type)

            if handler:
                instructions.append(handler(col_name, instruction))
            else:
                warnings.warn(
                    f"Unknown transformation type '{t_type}' for column '{col_name}' in configuration. This transformation will be ignored.",
                    UserWarning,
                )

    # Static concepts (unchanged)
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
    source_col = instruction.get("source_col")

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
        warnings.warn(f"Unknown metadata modifier: {instruction['modifier_cd']}")
        return None

    if not value:
        warnings.warn(f"Environment variable for {instruction['modifier_cd']} not set.")
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
                warnings.warn(
                    f"Unknown transform_type: {instruction['transform_type']}"
                )
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
    result_df["sourcesystem_cd"] = "AS:" + os.getenv("uuid", "unknown")
    return result_df


def delete_duplicate_entries_and_upload_into_db(transformed_df: pd.DataFrame) -> None:
    # Establish database connection
    username = os.getenv("username")
    password = os.getenv("password")
    i2b2_connection_url = os.getenv("connection-url")
    missing = [
        var
        for var, val in [
            ("username", username),
            ("password", password),
            ("connection-url", i2b2_connection_url),
        ]
        if not val
    ]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )
    pattern = r"jdbc:postgresql://(.*?)(\?searchPath=.*)?$"
    match = re.search(pattern, i2b2_connection_url)
    if not match:
        raise SystemExit(
            f"Invalid connection-url format: '{i2b2_connection_url}'. "
            "Expected format: 'jdbc:postgresql://<host>:<port>/<db>?searchPath=...'"
        )
    connection = match.group(1)
    engine = db.create_engine(
        f"postgresql+psycopg2://{username}:{password}@{connection}", pool_pre_ping=True
    )
    with engine.begin() as conn:
        table = db.Table("observation_fact", db.MetaData(), autoload_with=engine)

        delete_duplicate_entries(conn, table, transformed_df)
        upload_into_db(conn, table, transformed_df)


def delete_duplicate_entries(conn, table, transformed_df):
    keys_to_delete = (
        transformed_df[["encounter_num", "concept_cd"]]
        .drop_duplicates()
        .values.tolist()
    )

    if not keys_to_delete:
        warnings.warn("No records to delete.")
        return

    stmt = (
        table.delete()
        .where(table.c.sourcesystem_cd.like("AS%"))
        .where(tuple_(table.c.encounter_num, table.c.concept_cd).in_(keys_to_delete))
    )

    result = conn.execute(stmt)
    warnings.warn(f"Deleted {result.rowcount} rows.")


def upload_into_db(conn, table, transformed_df, batch_size=5000):
    total_rows = len(transformed_df)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = start_idx + batch_size
        batch_df = transformed_df.iloc[start_idx:end_idx]
        records = batch_df.to_dict(orient="records")

        if records:
            conn.execute(table.insert(), records)
            warnings.warn(
                f"Inserted batch {start_idx}-{min(end_idx, total_rows)} of {total_rows}"
            )

    print(f"Uploaded {total_rows} records.")


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
            raise SystemExit(f"Warning: Could not parse .env file. Error: {e}")


if __name__ == "__main__":

    load_env()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python rd_import.py <zip-file>")

    main(sys.argv[1])
