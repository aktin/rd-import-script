import unittest
import tempfile
from pathlib import Path
from src.rdimport import TmpFolderManager  # adjust to your actual module path


class TestTmpFolderManager(unittest.TestCase):
    """Unit tests for TmpFolderManager."""

    def setUp(self):
        # Create a temporary base directory for all tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.tmp_manager = TmpFolderManager(self.base_path)

    def tearDown(self):
        # Cleanup the temporary directory
        self.temp_dir.cleanup()

    def test_create_tmp_folder_creates_directory(self):
        """Ensure the tmp folder is created successfully."""
        tmp_path = self.tmp_manager.create_tmp_folder()
        self.assertTrue(tmp_path.exists())
        self.assertTrue(tmp_path.is_dir())
        self.assertEqual(tmp_path.name, "tmp")

    def test_remove_tmp_folder_removes_directory(self):
        """Ensure remove_tmp_folder deletes the tmp directory."""
        tmp_path = self.tmp_manager.create_tmp_folder()
        (tmp_path / "dummy.txt").write_text("hello")

        self.tmp_manager.remove_tmp_folder()
        self.assertFalse(tmp_path.exists())

    def test_rename_files_to_lowercase(self):
        """Ensure files are renamed to lowercase."""
        tmp_path = self.tmp_manager.create_tmp_folder()

        # Create mixed-case filenames
        filenames = ["HELLO.TXT", "Data.JSON", "ReadMe.MD"]
        for name in filenames:
            (tmp_path / name).write_text("content")

        renamed = self.tmp_manager.rename_files_to_lowercase()

        # Check all lowercase versions exist
        for name in filenames:
            self.assertTrue((tmp_path / name.lower()).exists())

        # Check return value matches renamed files
        renamed_names = sorted([p.name for p in renamed])
        expected_names = sorted([f.lower() for f in filenames])
        self.assertListEqual(renamed_names, expected_names)

    def test_rename_skips_conflicts(self):
        """If lowercase file already exists, renaming should skip it."""
        tmp_path = self.tmp_manager.create_tmp_folder()

        file_upper = tmp_path / "TEST.TXT"
        file_lower = tmp_path / "test.txt"

        file_upper.write_text("UPPER")
        file_lower.write_text("LOWER")

        renamed = self.tmp_manager.rename_files_to_lowercase()

        self.assertTrue(file_lower.exists())
        self.assertTrue(file_upper.exists())
        self.assertIn(file_lower, renamed)

    def test_context_manager_creates_and_removes_tmp(self):
        """Ensure context manager auto-cleans the tmp folder."""
        with TmpFolderManager(self.base_path) as manager:
            tmp_path = manager.create_tmp_folder()
            (tmp_path / "temp.txt").write_text("temporary")

            self.assertTrue(tmp_path.exists())
            self.assertTrue((tmp_path / "temp.txt").exists())

        # Folder should be deleted after exiting context
        self.assertFalse(tmp_path.exists())

    def test_get_files_in_tmp_folder_returns_only_files(self):
        """Ensure only files are returned, not directories."""
        tmp_path = self.tmp_manager.create_tmp_folder()
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "subdir").mkdir()

        files = self.tmp_manager._get_files_in_tmp_folder()
        names = sorted([f.name for f in files])
        self.assertListEqual(names, ["file1.txt", "file2.txt"])


if __name__ == "__main__":
    unittest.main()
