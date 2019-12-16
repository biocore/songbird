import unittest
import types
from importlib.machinery import SourceFileLoader
from click.testing import CliRunner
import shutil
import pkg_resources

# _Black magic from the stack overflow gods_:
# https://stackoverflow.com/questions/19009932/import-arbitrary-python
#  -source-file-python-3-3/41595552
loader = SourceFileLoader('songbird', './scripts/songbird')
songbird = types.ModuleType(loader.name)
loader.exec_module(songbird)


class TestSongbirdCLI(unittest.TestCase):

    def setUp(self) -> None:
        self.path = pkg_resources.resource_filename('songbird',
                                                    'testing_logs')

    def tearDown(self) -> None:
        shutil.rmtree(self.path)

    def test_cli_no_seed_set(self):
        runner = CliRunner()
        test_args = ['--input-biom', 'data/redsea/redsea.biom',
                     '--metadata-file', 'data/redsea/redsea_metadata.txt',
                     '--formula',
                     'Depth+Temperature+Salinity+Oxygen+Fluorescence'
                     '+Nitrate',
                     '--epochs', '100',
                     '--differential-prior', '0.5',
                     '--summary-interval', '1',
                     '--summary-dir', self.path]

        result = runner.invoke(songbird.multinomial, test_args)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error .with_traceback(ex.__traceback__)

    def test_cli_set_split_seed_int(self):
        runner = CliRunner()
        test_args = ['--input-biom', 'data/redsea/redsea.biom',
                     '--metadata-file', 'data/redsea/redsea_metadata.txt',
                     '--formula',
                     'Depth+Temperature+Salinity+Oxygen+Fluorescence'
                     '+Nitrate',
                     '--epochs', '100',
                     '--differential-prior', '0.5',
                     '--summary-interval', '1',
                     '--summary-dir', self.path,
                     '--split-seed', 42,
                     ]

        result = runner.invoke(songbird.multinomial, test_args)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error .with_traceback(ex.__traceback__)

    def test_cli_set_split_seed_None(self):
        runner = CliRunner()
        test_args = ['--input-biom', 'data/redsea/redsea.biom',
                     '--metadata-file', 'data/redsea/redsea_metadata.txt',
                     '--formula',
                     'Depth+Temperature+Salinity+Oxygen+Fluorescence'
                     '+Nitrate',
                     '--epochs', '100',
                     '--differential-prior', '0.5',
                     '--summary-interval', '1',
                     '--summary-dir', self.path,
                     '--split-seed', None,
                     ]

        result = runner.invoke(songbird.multinomial, test_args)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error .with_traceback(ex.__traceback__)

    def test_cli_set_tf_seed_int(self):
        runner = CliRunner()
        test_args = ['--input-biom', 'data/redsea/redsea.biom',
                     '--metadata-file', 'data/redsea/redsea_metadata.txt',
                     '--formula',
                     'Depth+Temperature+Salinity+Oxygen+Fluorescence'
                     '+Nitrate',
                     '--epochs', '100',
                     '--differential-prior', '0.5',
                     '--summary-interval', '1',
                     '--summary-dir', self.path,
                     '--split-seed', None,
                     '--tf-seed', 42,
                     ]

        result = runner.invoke(songbird.multinomial, test_args)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error .with_traceback(ex.__traceback__)

    def test_cli_set_split_seed_tf_seed_int(self):
        runner = CliRunner()
        test_args = ['--input-biom', 'data/redsea/redsea.biom',
                     '--metadata-file', 'data/redsea/redsea_metadata.txt',
                     '--formula',
                     'Depth+Temperature+Salinity+Oxygen+Fluorescence'
                     '+Nitrate',
                     '--epochs', '100',
                     '--differential-prior', '0.5',
                     '--summary-interval', '1',
                     '--summary-dir', self.path,
                     '--tf-seed', 42,
                     '--split-seed', 42,
                     ]

        result = runner.invoke(songbird.multinomial, test_args)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error .with_traceback(ex.__traceback__)


if __name__ == '__main__':
    unittest.main()
