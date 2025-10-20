# -*- coding: utf-8 -*
# Created on Wed Okt 15 09:15:55 2025
# @VERSION=1.0.0
# @VIEWNAME=RD-Importskript
# @MIMETYPE=zip
# @ID=p21
#
#      Copyright (c) 2025  AKTIN
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
import os
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import sqlalchemy as db


class RDImporter:
    def __init__(self, path_zip: str):
        self.__zfe = ZipFileExtractor(path_zip)
        path_parent = os.path.dirname(path_zip)
        self.__tfm = TmpFolderManager(path_parent)
        self.__num_imports = 0
        self.__num_updates = 0

    def __extract_and_rename_zip_content(self) -> str:
        path_tmp = self.__tfm.create_tmp_folder()
        self.__zfe.extract_zip_to_folder(path_tmp)
        self.__tfm.rename_files_in_tmp_folder_to_lowercase()
        return path_tmp

    def __preprocess_and_check_csv_files(self, patOpDataVerifierh_folder: str) -> str:
        # TODO: Add MappingVerifier, MappingPreprocessor
        for v, p in [(OpDataVerifier, OpDataPreprocessor)]:
            verifier = v()
            preprocessor = p()
            if verifier.is_csv_in_folder():
                preprocessor.preprocess()
                verifier.check_column_names_of_csv()

    def __get_matched_encounters(self, list_valid_ids):
        try:
            extractor = EncounterInfoExtractorWithBillingId()
        except ValueError:
            print("Matching by billing id failed. Trying matching by encounter id...")
            extractor = EncounterInfoExtractorWithEncounterId()

        matcher = DatabaseEncounterMatcher(extractor)
        return matcher.get_matched_df(list_valid_ids)

    def __enrich_with_admission_dates(
            self, verifier_fall, df_mapping: pd.DataFrame
    ) -> pd.DataFrame:
        dict_admission_dates = (
            verifier_fall.get_unique_ids_of_valid_with_admission_dates()
        )
        df_admission_dates = pd.DataFrame(
            {
                "encounter_id": list(dict_admission_dates.keys()),
                "aufnahmedatum": list(dict_admission_dates.values()),
            }
        )
        return pd.merge(df_mapping, df_admission_dates, on="encounter_id")

    def __print_verification_stats(self, verifier_fall, list_valid_ids: list, df_mapping: pd.DataFrame):
        print(f"Fälle gesamt: {verifier_fall.count_total_encounter()}")
        print(f"Fälle gesamt: {len(list_valid_ids)}")
        print(f"Valide Fälle gematcht mit Datenbank: {df_mapping.shape[0]}")

    def __import_observation_facts(self, df_mapping: pd.DataFrame, path_tmp: str) -> pd.DataFrame:
        # TODO: Add MappingUploadManager
        for uploader_class in [OpDataUploadManager]:
            uploader = uploader_class(df_mapping, path_tmp)
            if uploader.VERIFIER.is_csv_in_folder():
                uploader.upload_csv()
            if isinstance(uploader, OpDataUploadManager):
                self.__num_imports = uploader.NUM_IMPORTS
                self.__num_updates = uploader.NUM_UPDATES

    def __print_import_results(self):
        print(f"Fälle hochgeladen: {self.__num_imports + self.__num_updates}")
        print(f"Neue Fälle hochgeladen: {self.__num_imports}")
        print(f"Bestehende Fälle aktualisiert: {self.__num_updates}")

    def import_file(self):
        try:
            path_tmp = self.__extract_and_rename_zip_content()
            self.__preprocess_and_check_csv_files(path_tmp)
            verifier_fall = OpDataVerifier(path_tmp)
            list_valid_ids = verifier_fall.get_unique_ids_of_valid_encounter()
            df_mapping = self.__get_matched_encounters(list_valid_ids)
            df_mapping = self.__enrich_with_admission_dates(verifier_fall, df_mapping)
            self.__print_verification_stats(verifier_fall, list_valid_ids, df_mapping)
            self.__import_observation_facts(df_mapping, path_tmp)
            self.__print_import_results()
        finally:
            self.__tmp.remove_tmp_folder()


class ZipFileExtractor:
    """Utility class for safely extracting ZIP archives."""

    def __init__(self, path_zip: str | Path):
        self.path_zip = Path(path_zip)
        self._check_zip_file_integrity()

    def _check_zip_file_integrity(self) -> None:
        """Validate that the provided path exists and is a valid ZIP file."""
        if not self.path_zip.exists():
            raise FileNotFoundError(f"File not found: {self.path_zip}")
        if not zipfile.is_zipfile(self.path_zip):
            raise zipfile.BadZipFile(f"Not a valid ZIP archive: {self.path_zip}")

    def extract_zip_to_folder(self, destination: str | Path) -> list[Path]:
        """Extract the ZIP archive into a destination folder.

        Args:
            destination: The path to the output directory.

        Returns:
            List of extracted file paths.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(self.path_zip) as zip_ref:
                zip_ref.extractall(path=destination)
                return [destination / name for name in zip_ref.namelist()]
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to extract {self.path_zip}: {e}")


class TmpFolderManager:
    """Utility class to manage a temporary subfolder."""

    def __init__(self, base_folder: str | Path):
        self.path_tmp = Path(base_folder) / "tmp"

    def create_tmp_folder(self) -> Path:
        """Create the temporary folder if it doesn't exist."""
        self.path_tmp.mkdir(parents=True, exist_ok=True)
        return self.path_tmp.resolve()

    def remove_tmp_folder(self) -> None:
        """Remove the temporary folder and all its contents."""
        if self.path_tmp.is_dir():
            shutil.rmtree(self.path_tmp)

    def rename_files_to_lowercase(self) -> list[Path]:
        """Rename all files in tmp folder to lowercase.

        Returns:
            A list of renamed file paths.
        """
        renamed_files = []
        for file_path in self._get_files_in_tmp_folder():
            new_path = file_path.with_name(file_path.name.lower())

            # Avoid overwriting existing files with same lowercase name
            if new_path.exists() and new_path != file_path:
                continue  # skip or log if desired

            file_path.rename(new_path)
            renamed_files.append(new_path)
        return renamed_files

    def _get_files_in_tmp_folder(self) -> list[Path]:
        """Return a list of all files in the tmp folder."""
        if not self.path_tmp.exists():
            return []
        return [p for p in self.path_tmp.iterdir() if p.is_file()]

    def __enter__(self):
        self.create_tmp_folder()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_tmp_folder()


from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class CSVReader(ABC):
    """Abstract base class for reading and writing CSV files."""

    SIZE_CHUNKS: int = 10_000
    CSV_SEPARATOR: str = ";"
    CSV_NAME: str = ""

    def __init__(self, folder_path: str | Path):
        self.folder_path = Path(folder_path)
        self.path_csv = self.folder_path / self.CSV_NAME

    @staticmethod
    def get_csv_encoding() -> str:
        """Return the default encoding for CSV files (can be overridden)."""
        return "utf-8"

    @abstractmethod
    def read_csv(self) -> pd.DataFrame:
        """Read the CSV file and return a pandas DataFrame."""
        pass

    def save_df_as_csv(self, df: pd.DataFrame, output_path: str | Path, encoding: str | None = None) -> Path:
        """Save a DataFrame to CSV with the defined separator and encoding."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if encoding is None:
            encoding = self.get_csv_encoding()

        df.to_csv(output_path, sep=self.CSV_SEPARATOR, encoding=encoding, index=False)
        return output_path


class CSVPreprocessor(CSVReader, ABC):
    LEADING_ZEROS = 0

    def preprocess(self):
        header = self._get_csv_file_header_in_lowercase()
        header = self._remove_dashes_from_header(header)
        header += '\n'
        self._write_header_to_csv(header)
        self._append_zeros_to_internal_id()

    def _get_csv_file_header_in_lowercase(self) -> str:
        df = pd.read_csv(self.path_csv, nrows=0, index_col=None, sep=self.CSV_SEPARATOR,
                         encoding=self.get_csv_encoding(), dtype=str)
        df.rename(columns=str.lower, inplace=True)
        return ";".join(df.columns)

    @staticmethod
    def _remove_dashes_from_header(header: str) -> str:
        return header.replace("-", " ").strip()

    def _write_header_to_csv(self, header: str) -> None:
        path_parent = os.path.dirname(self.path_csv)
        path_dummy = os.path.sep.join([path_parent, 'dummy.csv'])
        encoding = self.get_csv_encoding()
        with open(self.path_csv, "r+", encoding=encoding) as f1, open(path_dummy, "w", encoding=encoding) as f2:
            f1.readline()
            f2.write(header)
            shutil.copyfileobj(f1, f2)
        os.remove(self.path_csv)
        os.rename(path_dummy, self.path_csv)

    def _rename_column_in_header(self, header: str, column_old: str, column_new: str) -> str:
        list_header = header.split(self.CSV_SEPARATOR)
        if list_header.count(column_new) == 1:
            return header
        pattern = "".join([r'^', column_old, r'(\.)?(\d*)?$'])
        idx_match = [i for i, item in enumerate(list_header) if re.search(pattern, item)]
        if len(idx_match) != 1:
            raise RuntimeError(f'Invalid count for column of {column_old} during adjustment')
        list_header[idx_match[0]] = column_new
        return self.CSV_SEPARATOR.join(list_header)

    def _append_zeros_to_internal_id(self):
        path_parent = os.path.dirname(self.path_csv)
        path_dummy = os.path.sep.join([path_parent, 'dummy.csv'])
        encoding = self.get_csv_encoding()
        df_tmp = pd.DataFrame()
        for chunk in pd.read_csv(self.path_csv, chunksize=self.SIZE_CHUNKS, sep=self.CSV_SEPARATOR, encoding=encoding,
                                 dtype=str):
            # chunk['khinterneskennzeichen'] = chunk['khinterneskennzeichen'].fillna('')
            # chunk['khinterneskennzeichen'] = chunk['khinterneskennzeichen'].apply(
            # lambda x: ''.join([str('0' * self.LEADING_ZEROS), x]))
            df_tmp = pd.concat([df_tmp, chunk])
        self.save_df_as_csv(df_tmp, path_dummy, encoding)
        os.remove(self.path_csv)
        os.rename(path_dummy, self.path_csv)


class CSVFileVerifier(CSVReader, ABC):
    DICT_COLUMN_PATTERN: dict
    MANDATORY_COLUMN_VALUES: list

    def is_csv_in_folder(self) -> bool:
        if not os.path.isfile(self.path_csv):
            print(f'{self.path_csv} does not exist')
            return False
        return True

    def check_column_names_of_csv(self):
        df = pd.read_csv(self.path_csv, nrows=0, index_col=None, sep=self.CSV_SEPARATOR,
                         encoding=self.get_csv_encoding(), dtype=str)
        set_required_columns = set(self.DICT_COLUMN_PATTERN.keys())
        set_matched_columns = set_required_columns.intersection(set(df.columns))
        if set_matched_columns != set_required_columns:
            raise SystemExit(
                f"Following columns are missing in {self.CSV_NAME}: {set_required_columns.difference(set_matched_columns)}")

    def get_unique_ids_of_valid_columns(self) -> list:
        set_valid_ids = set()
        for chunk in pd.read_csv(self.path_csv, chunksize=self.SIZE_CHUNKS, sep=self.CSV_SEPARATOR,
                                 encoding=self.get_csv_encoding(), dtype=str):
            chunk = chunk[list(self.DICT_COLUMN_PATTERN.keys())]
            chunk = chunk.fillna('')
            for column in chunk.columns.values:
                chunk = self.clear_invalid_column_fields_in_chunk(chunk, column)
            # set_valid_ids.update(chunk['khinterneskennzeichen'].unique()]

    def clear_invalid_column_fields_in_chunk(self, chunk: pd.Series, column_name: str) -> pd.Series:
        pattern = self.DICT_COLUMN_PATTERN[column_name]
        indices_empty_fields = chunk[chunk[column_name] == ''].index
        indices_wrong_syntax = chunk[(chunk[column_name] != '') & (~chunk[column_name].str.match(pattern))].index
        if len(indices_wrong_syntax):
            if column_name not in self.MANDATORY_COLUMN_VALUES:
                chunk.loc[indices_empty_fields, column_name] = ''
            else:
                chunk = chunk.drop(indices_wrong_syntax)
        if len(indices_empty_fields) and column_name in self.MANDATORY_COLUMN_VALUES:
            chunk = chunk.drop(indices_empty_fields)
        return chunk


class OpDataPreprocessor(CSVPreprocessor):
    CSV_NAME = 'OpData.csv'

    def preprocess(self):
        super().preprocess()


class OpDataVerifier:
    CSV_NAME = 'OpData.csv'
    DICT_COLUMN_PATTERN = {
        ''
    }


class OpDataUploadManager:
    pass



class DatabaseConnection(ABC):
    ENGINE: db.engine.Engine = None

    def __init__(self):
        self.username = os.environ['username']
        self.password = os.environ['password']
        self.i2b2_connection_url = os.environ['connection_url']
        self.__init_engine()

    def __init_engine(self):
        pattern = r'jdbc:postgresql://(.*?)(\?searchPath=.*)?$'
        connection = re.search(pattern, self.i2b2_connection_url).group(1)
        self.engine = db.create_engine(f"postgresql+psycopg2://{self.username}:{self.password}@{connection}",
                                       pool_pre_ping=True)

    def open_connection(self):
        return self.engine.connect()

    def __del__(self):
        if self.engine is not None:
            self.engine.dispose()


class DatabaseExtractor(DatabaseConnection, ABC):
    SIZE_CHUNKS: int = 10000

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        pass

    def _stream_query_into_df(self, query: db.sql.expression) -> pd.DataFrame:
        df = pd.DataFrame()
        with self.open_connection() as connection:
            result = connection.execution_options(stream_results=True).execute(query)
            while True:
                chunk = result.fetchmany(size=self.SIZE_CHUNKS)
                if not chunk:
                    break
                if df.empty:
                    df = pd.DataFrame(chunk)
                else:
                    df = df.append(chunk, ignore_index=True)
            if df.empty:
                raise ValueError("No entries for database query was found")
            df.columns = result.keys()
            return df

class EncounterInfoExtractorWithEncounterId(DatabaseExtractor):
    """
    SQLAlchemy-Query to extract encounter_id, encounter_num and patient_num for AKTIN
    optin encounter from database. Column for encounter_id is renmaed to 'match_id'
    to streamline the matching in DatabaseEncounterMatcher.
    """

    def extract(self) -> pd.DataFrame:
        enc = db.Table(
            "encounter_mapping", db.MetaData(), autoload_with=self.ENGINE
        )
        pat = db.Table("patient_mapping", db.MetaData(), autoload_with=self.ENGINE)
        opt = db.Table(
            "optinout_patients", db.MetaData(), autoload_with=self.ENGINE
        )
        query = (
            db.select(
                enc.c["encounter_ide"],
                enc.c["encounter_num"],
                pat.c["patient_num"],
            )
            .select_from(
                enc.join(pat, enc.c["patient_ide"] == pat.c["patient_ide"]).join(
                    opt, pat.c["patient_ide"] == opt.c["pat_psn"], isouter=True
                )
            )
            .where(db.or_(opt.c["study_id"] != "AKTIN", opt.c["pat_psn"].is_(None)))
        )
        df = self._stream_query_into_df(query)
        df.rename(columns={"encounter_ide": "match_id"}, inplace=True)
        return df


class EncounterInfoExtractorWithBillingId(DatabaseExtractor):
    pass

class DatabaseEncounterMatcher:
    pass
