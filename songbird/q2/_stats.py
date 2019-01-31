from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData


SongbirdStats = SemanticType('SongbirdStats',
                             variant_of=SampleData.field['type'])


class SongbirdStatsFormat(model.TextFileFormat):
    def validate(*args):
        pass


SongbirdStatsDirFmt = model.SingleFileDirectoryFormat(
    'SongbirdStatsDirFmt', 'stats.tsv', SongbirdStatsFormat)
