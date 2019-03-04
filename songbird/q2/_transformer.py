import qiime2
import pandas as pd

from songbird.q2 import SongbirdStatsFormat, DifferentialFormat
from songbird.q2.plugin_setup import plugin


@plugin.register_transformer
def _1(ff: SongbirdStatsFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _2(obj: qiime2.Metadata) -> SongbirdStatsFormat:
    ff = SongbirdStatsFormat()
    obj.save(str(ff))
    return ff


@plugin.register_transformer
def _3(ff: DifferentialFormat) -> pd.DataFrame:
    df = pd.read_csv(str(ff), sep='\t', comment='#', skip_blank_lines=True,
                     header=0, dtype=object, index_col=0)
    return df


@plugin.register_transformer
def _4(df: pd.DataFrame) -> DifferentialFormat:
    ff = DifferentialFormat()
    df.to_csv(str(ff), sep='\t', header=True, index=True)
    return ff
