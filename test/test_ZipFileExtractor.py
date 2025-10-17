import unittest
import tempfile
import zipfile
from pathlib import Path
from src.rdimport import ZipFileExtractor


class TestZipFileExtractor(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        self.valid_zip = self.tmp_path / "test.zip"
        with zipfile.ZipFile(self.valid_zip, "w") as zipf:
            (self.tmp_path / "file1.txt").write_text("Hello World!")
            (self.tmp_path / "file2.txt").write_text("Python Testing")
            zipf.write(self.tmp_path / "file1.txt", arcname="file1.txt")
            zipf.write(self.tmp_path / "file2.txt", arcname="file2.txt")

        self.invalid_zip = self.tmp_path / "not_a_zip.txt"
        self.invalid_zip.write_text("I'm not a zip file")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_valid_zip_extraction(self):
        extractor = ZipFileExtractor(self.valid_zip)
        output_folder = self.tmp_path / "output"
        extracted_files = extractor.extract_zip_to_folder(output_folder)

        self.assertTrue((output_folder / "file1.txt").exists())
        self.assertTrue((output_folder / "file2.txt").exists())
        self.assertEqual(len(extracted_files), 2)

    def test_missing_file_raises(self):
        missing_path = self.tmp_path / "does_not_exist.zip"
        with self.assertRaises(FileNotFoundError):
            ZipFileExtractor(missing_path)

    def test_invalid_zip_raises(self):
        with self.assertRaises(zipfile.BadZipFile):
            ZipFileExtractor(self.invalid_zip)

    def test_extract_returns_correct_file_list(self):
        extractor = ZipFileExtractor(self.valid_zip)
        output_folder = self.tmp_path / "extract_here"
        extracted = extractor.extract_zip_to_folder(output_folder)
        extracted_names = [p.name for p in extracted]
        self.assertListEqual(sorted(extracted_names), ["file1.txt", "file2.txt"])


if __name__ == "__main__":
    unittest.main()
