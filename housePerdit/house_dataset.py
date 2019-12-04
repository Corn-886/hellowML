import tensorflow as tf

_CSV_COLUMNS_0ME = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
                    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                    'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
                    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                    'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
                    'SaleType', 'SaleCondition', 'SalePrice']
_CSV_COLUMN_DEFAULTS = [['0'], [0], ['0'], ['0'], [0], ['0'], ['0'], ['0'],
                        ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
                        ['0'], ['0'], ['0'], [0], [0], [0],
                        [0], ['0'], ['0'], ['0'], ['0'], ['0'],
                        ['0'], ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
                        ['0'], [0], ['0'], [0], [0], [0],
                        ['0'], ['0'], ['0'], ['0'], [0], [0], [0],
                        [0], [0], [0], [0], [0], [0],
                        [0], ['0'], [0], ['0'], [0], ['0'],
                        ['0'], [0], ['0'], [0], [0], ['0'],
                        ['0'], ['0'], [0], [0], [0], [0],
                        [0], [0], ['0'], ['0'], ['0'], [0], [0], [0],
                        ['0'], ['0'], [0]]

MiscFeature = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
    'MiscFeature', ['Elev', 'Gar2', 'Othr']))
SaleType = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
    'SaleType', ['WD', 'CWD,', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']))
MiscVal = tf.feature_column.numeric_column(
    'MiscVal', default_value=0)
MoSold = tf.feature_column.numeric_column(
    'MoSold', default_value=0)
YrSold = tf.feature_column.numeric_column(
    'YrSold', default_value=0)
SaleCondition = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        'SaleCondition', ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']))
MSSubClass = tf.feature_column.numeric_column('MSSubClass')
MSZoning = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
    'MSZoning', ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']))
LotFrontage_emb = tf.feature_column.embedding_column(
    categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
        'LotFrontage', hash_bucket_size=60), dimension=9)
LotArea = tf.feature_column.numeric_column('LotArea')
Street = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Street', ['Grvl', 'Pave']))
Alley = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Alley', ['Grvl', 'Pave', '0']))
LotShape = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('LotShape', ['Reg', 'IR1', 'IR2', 'IR3']))
LandContour = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('LandContour', ['Lvl', 'Bnk', 'HLS', 'Low']))
LotConfig = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('LotConfig',
                                                              ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']))
LandSlope = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('LandSlope', ['Gtl', 'Mod', 'Sev']))

Neighborhood_emb = tf.feature_column.embedding_column(
    categorical_column=tf.feature_column.categorical_column_with_hash_bucket('Neighborhood', hash_bucket_size=25),
    dimension=9)
Condition1 = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Condition1',
                                                              ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN',
                                                               'PosA', 'RRNe', 'RRAe']))
OverallCond = tf.feature_column.numeric_column('OverallCond')
Condition2 = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Condition2',
                                                              ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN',
                                                               'PosA', 'RRNe', 'RRAe']))
BldgType = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('BldgType',
                                                              ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI']))
HouseStyle = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('HouseStyle',
                                                              ['1.5Fin', '1.5Unf', '2Story', '2.5Fin', '5Unf', 'SFoyer',
                                                               'SLvl']))
OverallQual = tf.feature_column.numeric_column('OverallQual')
YearBuilt = tf.feature_column.numeric_column('YearBuilt')
YearRemodAdd = tf.feature_column.numeric_column('YearRemodAdd')
RoofStyle = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('RoofStyle',
                                                              ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']))
RoofMatl = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('RoofMatl',
                                                              ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll',
                                                               'Tar&Grv', 'WdShake', 'WdShngl']))
Exterior1st_emb = tf.feature_column.embedding_column(
    categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
        'Exterior1st', hash_bucket_size=25), dimension=9)
Exterior2nd_emb = tf.feature_column.embedding_column(
    categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
        'Exterior2nd', hash_bucket_size=25
    ), dimension=9)
MasVnrType = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('MasVnrType',
                                                              ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone']))
BsmtFinType1 = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('BsmtFinType1',
                                                              ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', '0']))
BsmtFinSF1 = tf.feature_column.numeric_column('BsmtFinSF1')
BsmtFinType2 = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('BsmtFinType2',
                                                              ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', '0']))
BsmtFinSF2 = tf.feature_column.numeric_column('BsmtFinSF2')
BsmtUnfSF = tf.feature_column.numeric_column('BsmtUnfSF')
TotalBsmtSF = tf.feature_column.numeric_column('TotalBsmtSF')
Heating = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Heating',
                                                              ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']))
HeatingQC = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('HeatingQC',
                                                              ['Ex', 'Gd', 'TA', 'Fa', 'Po']))
Electrical = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Electrical',
                                                              ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']))
stFlrSF_1 = tf.feature_column.numeric_column('1stFlrSF')
ndFlrSF_2 = tf.feature_column.numeric_column('2ndFlrSF')
LowQualFinSF = tf.feature_column.numeric_column('LowQualFinSF')
GrLivArea = tf.feature_column.numeric_column("GrLivArea")
BsmtFullBath = tf.feature_column.numeric_column("BsmtFullBath")
BsmtHalfBath = tf.feature_column.numeric_column("BsmtHalfBath")
FullBath = tf.feature_column.numeric_column("FullBath")
HalfBath = tf.feature_column.numeric_column("HalfBath")
BedroomAbvGr = tf.feature_column.numeric_column("BedroomAbvGr")
KitchenAbvGr = tf.feature_column.numeric_column("KitchenAbvGr")
KitchenQual = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('KitchenQual',
                                                              ['Ex', 'Gd', 'TA', 'Fa', 'Po']))
TotRmsAbvGrd = tf.feature_column.numeric_column("TotRmsAbvGrd")
Functional = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Functional',
                                                              ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev',
                                                               'Sal']))
Fireplaces = tf.feature_column.numeric_column("Fireplaces")
FireplaceQu = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('FireplaceQu',
                                                              ['Ex', 'Gd', 'TA', 'Fa', 'Po', '0']))
GarageType = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('GarageType',
                                                              ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort',
                                                               'Detchd', '0']))
GarageYrBlt = tf.feature_column.numeric_column("GarageYrBlt")
GarageCars = tf.feature_column.numeric_column("GarageCars")
GarageArea = tf.feature_column.numeric_column("GarageArea")
GarageQual = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('GarageQual',
                                                              ['Ex', 'Gd', 'TA', 'Fa', 'Po', '0']))
GarageCond = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('GarageCond',
                                                              ['Ex', 'Gd', 'TA', 'Fa', 'Po', '0']))
PavedDrive = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('PavedDrive',
                                                              ['Y', 'P', 'N']))
WoodDeckSF = tf.feature_column.numeric_column("WoodDeckSF")
OpenPorchSF = tf.feature_column.numeric_column("OpenPorchSF")
EnclosedPorch = tf.feature_column.numeric_column("EnclosedPorch")
tSsnPorch = tf.feature_column.numeric_column("3SsnPorch")
ScreenPorch = tf.feature_column.numeric_column("ScreenPorch")
PoolArea = tf.feature_column.numeric_column("PoolArea")
PoolQC = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('PoolQC',
                                                              ['Ex', 'Gd', 'TA', 'Fa', '0']))

Fence = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('Fence',
                                                              ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', '0']))


def get_base_column():
    base_column = [
        MiscFeature, SaleType, MiscVal, MoSold, YrSold, SaleCondition, MSSubClass, MSZoning, LotArea, Street, Alley,
        LotShape, LandContour, LotConfig, LandSlope, Neighborhood_emb, Condition1, OverallCond, Condition2, BldgType,
        HouseStyle, OverallQual, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, Exterior1st_emb, Exterior2nd_emb,
        MasVnrType, BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating, HeatingQC,
        Electrical, stFlrSF_1, ndFlrSF_2, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath,
        BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType,
        GarageYrBlt, GarageCars, GarageArea, GarageQual, GarageCond, PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch,
        tSsnPorch, ScreenPorch, PoolArea, PoolQC, Fence
    ]

    return base_column


# 构造输入数据
def construct_input_fn(data_path, branch_size, num_epochs, shuffle):
    def parse_csv(line):
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个转为字典类型{'a':'b'}。record_defaults用于指明每一列的缺失值用什么填充,
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS_0ME, columns))
        lables = features.pop('SalePrice')
        return features, lables

    # 返回数据
    dataset = tf.data.TextLineDataset(data_path).skip(1)
    dataset = dataset.map(parse_csv, num_parallel_calls=16)
    # 打乱顺序
    if shuffle:
        dataset = dataset.shuffle(1460)

    dataset = dataset.batch(branch_size)
    # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
    # 为了配合输出次数，一般默认repeat()空
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
