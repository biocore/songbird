from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData


# songbird stats summarizing loss and cv error
SongbirdStats = SemanticType('SongbirdStats',
                             variant_of=SampleData.field['type'])


class SongbirdStatsFormat(model.TextFileFormat):
    def validate(*args):
        pass


SongbirdStatsDirFmt = model.SingleFileDirectoryFormat(
    'SongbirdStatsDirFmt', 'stats.tsv', SongbirdStatsFormat)
