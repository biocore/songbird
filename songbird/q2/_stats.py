from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData


# songbird stats summarizing loss and cv error
SongbirdStats = SemanticType('SongbirdStats',
                             variant_of=SampleData.field['type'])


class SongbirdStatsFormat(model.TextFileFormat):
    def validate(*args):
        pass


SongbirdStatsDirFmt = model.SingleFileDirectoryFormat(
    'SongbirdStatsDirFmt', 'stats.tsv', SongbirdStatsFormat)

# songbird differentials
Differential = SemanticType('Differential',
                            variant_of=FeatureData.field['type'])


class DifferentialFormat(model.TextFileFormat):
    def validate(*args):
        pass


DifferentialDirFmt = model.SingleFileDirectoryFormat(
    'DifferentialDirFmt', 'differentials.tsv', DifferentialFormat)
