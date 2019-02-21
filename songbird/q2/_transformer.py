import qiime2

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
def _3(ff: DifferentialFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))

@plugin.register_transformer
def _4(obj: qiime2.Metadata) -> DifferentialFormat:
    ff = SongbirdStatsFormat()
    obj.save(str(ff))
    return ff
