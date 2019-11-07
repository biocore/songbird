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

    def test_cli(self):
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
        # Uncommenting the below lines is useful for debugging if
        #  you are getting result.exit_code != 0
        # import traceback
        # ex = result.exception
        # traceback.print_exception(type(ex), ex, ex.__traceback__)
        self.assertEqual(0, result.exit_code)


if __name__ == '__main__':
    unittest.main()
