import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("download_model", Path(__file__).parents[1] / "download_model.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadModelTest(unittest.TestCase):
    def test_registered_aliases_preserve_binary_and_print_one_path(self):
        # Include explicit --filename: it must not accidentally return ambiguous model.pte.
        for repo, alias in downloader.IMPORT_FILENAMES.items():
            for explicit in (False, True):
                with self.subTest(repo=repo, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                    def download(**kwargs):
                        path = kwargs["local_dir"] / kwargs["filename"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"model binary fixture")
                        return str(path)

                    hub = types.SimpleNamespace(hf_hub_download=download, snapshot_download=None)
                    args = ["download_model.py", "--repo-id", repo,
                            "--output-dir", directory, "--print-path"]
                    if explicit:
                        args += ["--filename", downloader.VERIFIED_MODELS[repo]]
                    output = io.StringIO()
                    with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(sys, "argv", args):
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                            downloader.main()
                    lines = output.getvalue().splitlines()
                    self.assertEqual(len(lines), 1)
                    result = Path(lines[0])
                    self.assertEqual(result.name, alias)
                    self.assertEqual(result.read_bytes(), b"model binary fixture")


if __name__ == "__main__":
    unittest.main()
