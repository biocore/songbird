from io import StringIO
import unittest
from songbird.util import read_metadata
import pandas as pd
import pandas.util.testing as pdt


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.data1 = StringIO(
            '\n'.join(
                [
                    "sampleid\tcol1\tcol2\tcol3",
                    "0\t0\t2\t4",
                    "1\t1\t3\t5",
                    "3\t4\t8\t12"
                ]
            )
        )

        self.data2 = StringIO(
            '\n'.join(
                [
                    "sampleid\tcol1\tcol2\tcol3",
                    "a\t0\t1\t4",
                    "1\t1\ta\t5",
                    "3\t4\tc\t12"
                ]
            )
        )

    def test_read_index(self):
        res = read_metadata(self.data1)
        exp = pd.DataFrame(
            [
                ['0', 0.0, 2.0,  4.0],
                ['1', 1.0, 3.0,  5.0],
                ['3', 4.0, 8.0, 12.0]
            ],
            columns=['sampleid', 'col1', 'col2', 'col3'])

        exp = exp.set_index('sampleid')
        pdt.assert_frame_equal(exp, res)

    def test_read_categorical(self):
        res = read_metadata(self.data2)
        exp = pd.DataFrame(
            [
                ['a', 0.0, '1',  4.0],
                ['1', 1.0, 'a',  5.0],
                ['3', 4.0, 'c', 12.0]
            ],
            columns=['sampleid', 'col1', 'col2', 'col3'])

        exp = exp.set_index('sampleid')
        pdt.assert_frame_equal(exp, res)


if __name__ == "__main__":
    unittest.main()
