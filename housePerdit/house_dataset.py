import tensorflow as tf

_CSV_COLUMNS_NAME = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
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
_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [0]]

MiscFeature = tf.feature_column.categorical_column_with_hash_bucket(
    'MiscFeature', hash_bucket_size=100
)
MiscFeature_emb = tf.feature_column.embedding_column(
    categorical_column=MiscFeature, dimension=9)
MiscVal = tf.feature_column.categorical_column_with_hash_bucket(
    'MiscVal', hash_bucket_size=100
)
MiscVal_emb = tf.feature_column.embedding_column(
    categorical_column=MiscVal, dimension=9)
MoSold = tf.feature_column.categorical_column_with_hash_bucket(
    'MoSold', hash_bucket_size=100
)
MoSold_emb = tf.feature_column.embedding_column(
    categorical_column=MoSold, dimension=9)
YrSold = tf.feature_column.categorical_column_with_hash_bucket(
    'YrSold', hash_bucket_size=100
)
YrSold_emb = tf.feature_column.embedding_column(
    categorical_column=YrSold, dimension=9)
SaleType = tf.feature_column.categorical_column_with_hash_bucket(
    'SaleType', hash_bucket_size=100
)
SaleType_emb = tf.feature_column.embedding_column(
    categorical_column=SaleType, dimension=9)
SaleCondition = tf.feature_column.categorical_column_with_hash_bucket(
    'SaleCondition', hash_bucket_size=100
)
SaleCondition_emb = tf.feature_column.embedding_column(
    categorical_column=SaleCondition, dimension=9)
MSSubClass = tf.feature_column.categorical_column_with_hash_bucket(
    'MSSubClass', hash_bucket_size=100
)
MSSubClass_emb = tf.feature_column.embedding_column(
    categorical_column=MSSubClass, dimension=9)
MSZoning = tf.feature_column.categorical_column_with_hash_bucket(
    'MSZoning', hash_bucket_size=100
)
MSZoning_emb = tf.feature_column.embedding_column(
    categorical_column=MSZoning, dimension=9)
LotFrontage = tf.feature_column.categorical_column_with_hash_bucket(
    'LotFrontage', hash_bucket_size=100
)
LotFrontage_emb = tf.feature_column.embedding_column(
    categorical_column=LotFrontage, dimension=9)
LotArea = tf.feature_column.categorical_column_with_hash_bucket(
    'LotArea', hash_bucket_size=100
)
LotArea_emb = tf.feature_column.embedding_column(
    categorical_column=LotArea, dimension=9)
Street = tf.feature_column.categorical_column_with_hash_bucket(
    'Street', hash_bucket_size=100
)
Street_emb = tf.feature_column.embedding_column(
    categorical_column=Street, dimension=9)
Alley = tf.feature_column.categorical_column_with_hash_bucket(
    'Alley', hash_bucket_size=100
)
Alley_emb = tf.feature_column.embedding_column(
    categorical_column=Alley, dimension=9)
LotShape = tf.feature_column.categorical_column_with_hash_bucket(
    'LotShape', hash_bucket_size=100
)
LotShape_emb = tf.feature_column.embedding_column(
    categorical_column=LotShape, dimension=9)
LandContour = tf.feature_column.categorical_column_with_hash_bucket(
    'LandContour', hash_bucket_size=100
)
LandContour_emb = tf.feature_column.embedding_column(
    categorical_column=LandContour, dimension=9)
LotConfig = tf.feature_column.categorical_column_with_hash_bucket(
    'LotConfig', hash_bucket_size=100
)
LotConfig_emb = tf.feature_column.embedding_column(
    categorical_column=LotConfig, dimension=9)
LandSlope = tf.feature_column.categorical_column_with_hash_bucket(
    'LandSlope', hash_bucket_size=100
)
LandSlope_emb = tf.feature_column.embedding_column(
    categorical_column=LandSlope, dimension=9)
Neighborhood = tf.feature_column.categorical_column_with_hash_bucket(
    'Neighborhood', hash_bucket_size=100
)
Neighborhood_emb = tf.feature_column.embedding_column(
    categorical_column=Neighborhood, dimension=9)
OverallCond = tf.feature_column.categorical_column_with_hash_bucket(
    'OverallCond', hash_bucket_size=100
)
OverallCond_emb = tf.feature_column.embedding_column(
    categorical_column=OverallCond, dimension=9)
Condition2 = tf.feature_column.categorical_column_with_hash_bucket(
    'Condition2', hash_bucket_size=100
)
Condition2_emb = tf.feature_column.embedding_column(
    categorical_column=Condition2, dimension=9)
BldgType = tf.feature_column.categorical_column_with_hash_bucket(
    'BldgType', hash_bucket_size=100
)
BldgType_emb = tf.feature_column.embedding_column(
    categorical_column=BldgType, dimension=9)
HouseStyle = tf.feature_column.categorical_column_with_hash_bucket(
    'HouseStyle', hash_bucket_size=100
)
HouseStyle_emb = tf.feature_column.embedding_column(
    categorical_column=HouseStyle, dimension=9)
OverallQual = tf.feature_column.categorical_column_with_hash_bucket(
    'OverallQual', hash_bucket_size=100
)
OverallQual_emb = tf.feature_column.embedding_column(
    categorical_column=OverallQual, dimension=9)
YearBuilt = tf.feature_column.categorical_column_with_hash_bucket(
    'YearBuilt', hash_bucket_size=100
)
YearBuilt_emb = tf.feature_column.embedding_column(
    categorical_column=YearBuilt, dimension=9)
YearRemodAdd = tf.feature_column.categorical_column_with_hash_bucket(
    'YearRemodAdd', hash_bucket_size=100
)
YearRemodAdd_emb = tf.feature_column.embedding_column(
    categorical_column=YearRemodAdd, dimension=9)
RoofStyle = tf.feature_column.categorical_column_with_hash_bucket(
    'RoofStyle', hash_bucket_size=100
)
RoofStyle_emb = tf.feature_column.embedding_column(
    categorical_column=RoofStyle, dimension=9)
RoofMatl = tf.feature_column.categorical_column_with_hash_bucket(
    'RoofMatl', hash_bucket_size=100
)
RoofMatl_emb = tf.feature_column.embedding_column(
    categorical_column=RoofMatl, dimension=9)
Exterior1st = tf.feature_column.categorical_column_with_hash_bucket(
    'Exterior1st', hash_bucket_size=100
)
Exterior1st_emb = tf.feature_column.embedding_column(
    categorical_column=Exterior1st, dimension=9)
Exterior2nd = tf.feature_column.categorical_column_with_hash_bucket(
    'Exterior2nd', hash_bucket_size=100
)
Exterior2nd_emb = tf.feature_column.embedding_column(
    categorical_column=Exterior2nd, dimension=9)
MasVnrType = tf.feature_column.categorical_column_with_hash_bucket(
    'MasVnrType', hash_bucket_size=100
)
MasVnrType_emb = tf.feature_column.embedding_column(
    categorical_column=MasVnrType, dimension=9)
BsmtFinType1 = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtFinType1', hash_bucket_size=100
)
BsmtFinType1_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtFinType1, dimension=9)
BsmtFinSF1 = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtFinSF1', hash_bucket_size=100
)
BsmtFinSF1_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtFinSF1, dimension=9)
BsmtFinType2 = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtFinType2', hash_bucket_size=100
)
BsmtFinType2_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtFinType2, dimension=9)
BsmtFinSF2 = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtFinSF2', hash_bucket_size=100
)
BsmtFinSF2_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtFinSF2, dimension=9)
BsmtUnfSF = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtUnfSF', hash_bucket_size=100
)
BsmtUnfSF_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtUnfSF, dimension=9)
TotalBsmtSF = tf.feature_column.categorical_column_with_hash_bucket(
    'TotalBsmtSF', hash_bucket_size=100
)
TotalBsmtSF_emb = tf.feature_column.embedding_column(
    categorical_column=TotalBsmtSF, dimension=9)
Heating = tf.feature_column.categorical_column_with_hash_bucket(
    'Heating', hash_bucket_size=100
)
Heating_emb = tf.feature_column.embedding_column(
    categorical_column=Heating, dimension=9)
HeatingQC = tf.feature_column.categorical_column_with_hash_bucket(
    'HeatingQC', hash_bucket_size=100
)
HeatingQC_emb = tf.feature_column.embedding_column(
    categorical_column=HeatingQC, dimension=9)
Electrical = tf.feature_column.categorical_column_with_hash_bucket(
    'Electrical', hash_bucket_size=100
)
Electrical_emb = tf.feature_column.embedding_column(
    categorical_column=Electrical, dimension=9)
lstFlrSF = tf.feature_column.categorical_column_with_hash_bucket(
    '1stFlrSF', hash_bucket_size=100
)
lstFlrSF_emb = tf.feature_column.embedding_column(
    categorical_column=lstFlrSF, dimension=9)
tondFlrSF = tf.feature_column.categorical_column_with_hash_bucket(
    '2ndFlrSF', hash_bucket_size=100
)
tondFlrSF_emb = tf.feature_column.embedding_column(
    categorical_column=tondFlrSF, dimension=9)
LowQualFinSF = tf.feature_column.categorical_column_with_hash_bucket(
    'LowQualFinSF', hash_bucket_size=100
)
LowQualFinSF_emb = tf.feature_column.embedding_column(
    categorical_column=LowQualFinSF, dimension=9)
GrLivArea = tf.feature_column.categorical_column_with_hash_bucket(
    'GrLivArea', hash_bucket_size=100
)
GrLivArea_emb = tf.feature_column.embedding_column(
    categorical_column=GrLivArea, dimension=9)
BsmtFullBath = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtFullBath', hash_bucket_size=100
)
BsmtFullBath_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtFullBath, dimension=9)
BsmtHalfBath = tf.feature_column.categorical_column_with_hash_bucket(
    'BsmtHalfBath', hash_bucket_size=100
)
BsmtHalfBath_emb = tf.feature_column.embedding_column(
    categorical_column=BsmtHalfBath, dimension=9)
FullBath = tf.feature_column.categorical_column_with_hash_bucket(
    'FullBath', hash_bucket_size=100
)
FullBath_emb = tf.feature_column.embedding_column(
    categorical_column=FullBath, dimension=9)
HalfBath = tf.feature_column.categorical_column_with_hash_bucket(
    'HalfBath', hash_bucket_size=100
)
HalfBath_emb = tf.feature_column.embedding_column(
    categorical_column=HalfBath, dimension=9)
BedroomAbvGr = tf.feature_column.categorical_column_with_hash_bucket(
    'BedroomAbvGr', hash_bucket_size=100
)
BedroomAbvGr_emb = tf.feature_column.embedding_column(
    categorical_column=BedroomAbvGr, dimension=9)
KitchenAbvGr = tf.feature_column.categorical_column_with_hash_bucket(
    'KitchenAbvGr', hash_bucket_size=100
)
KitchenAbvGr_emb = tf.feature_column.embedding_column(
    categorical_column=KitchenAbvGr, dimension=9)
KitchenQual = tf.feature_column.categorical_column_with_hash_bucket(
    'KitchenQual', hash_bucket_size=100
)
KitchenQual_emb = tf.feature_column.embedding_column(
    categorical_column=KitchenQual, dimension=9)
TotRmsAbvGrd = tf.feature_column.categorical_column_with_hash_bucket(
    'TotRmsAbvGrd', hash_bucket_size=100
)
TotRmsAbvGrd_emb = tf.feature_column.embedding_column(
    categorical_column=TotRmsAbvGrd, dimension=9)
Functional = tf.feature_column.categorical_column_with_hash_bucket(
    'Functional', hash_bucket_size=100
)
Functional_emb = tf.feature_column.embedding_column(
    categorical_column=Functional, dimension=9)
Fireplaces = tf.feature_column.categorical_column_with_hash_bucket(
    'Fireplaces', hash_bucket_size=100
)
Fireplaces_emb = tf.feature_column.embedding_column(
    categorical_column=Fireplaces, dimension=9)
FireplaceQu = tf.feature_column.categorical_column_with_hash_bucket(
    'FireplaceQu', hash_bucket_size=100
)
FireplaceQu_emb = tf.feature_column.embedding_column(
    categorical_column=FireplaceQu, dimension=9)
GarageType = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageType', hash_bucket_size=100
)
GarageType_emb = tf.feature_column.embedding_column(
    categorical_column=GarageType, dimension=9)
GarageYrBlt = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageYrBlt', hash_bucket_size=100
)
GarageYrBlt_emb = tf.feature_column.embedding_column(
    categorical_column=GarageYrBlt, dimension=9)
GarageCars = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageCars', hash_bucket_size=100
)
GarageCars_emb = tf.feature_column.embedding_column(
    categorical_column=GarageCars, dimension=9)
GarageArea = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageArea', hash_bucket_size=100
)
GarageArea_emb = tf.feature_column.embedding_column(
    categorical_column=GarageArea, dimension=9)
GarageQual = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageQual', hash_bucket_size=100
)
GarageQual_emb = tf.feature_column.embedding_column(
    categorical_column=GarageQual, dimension=9)
GarageCond = tf.feature_column.categorical_column_with_hash_bucket(
    'GarageCond', hash_bucket_size=100
)
GarageCond_emb = tf.feature_column.embedding_column(
    categorical_column=GarageCond, dimension=9)
PavedDrive = tf.feature_column.categorical_column_with_hash_bucket(
    'PavedDrive', hash_bucket_size=100
)
PavedDrive_emb = tf.feature_column.embedding_column(
    categorical_column=PavedDrive, dimension=9)
WoodDeckSF = tf.feature_column.categorical_column_with_hash_bucket(
    'WoodDeckSF', hash_bucket_size=100
)
WoodDeckSF_emb = tf.feature_column.embedding_column(
    categorical_column=WoodDeckSF, dimension=9)
OpenPorchSF = tf.feature_column.categorical_column_with_hash_bucket(
    'OpenPorchSF', hash_bucket_size=100
)
OpenPorchSF_emb = tf.feature_column.embedding_column(
    categorical_column=OpenPorchSF, dimension=9)
EnclosedPorch = tf.feature_column.categorical_column_with_hash_bucket(
    'EnclosedPorch', hash_bucket_size=100
)
EnclosedPorch_emb = tf.feature_column.embedding_column(
    categorical_column=EnclosedPorch, dimension=9)
tSsnPorch = tf.feature_column.categorical_column_with_hash_bucket(
    '3SsnPorch', hash_bucket_size=100
)
tSsnPorch_emb = tf.feature_column.embedding_column(
    categorical_column=tSsnPorch, dimension=9)
ScreenPorch = tf.feature_column.categorical_column_with_hash_bucket(
    'ScreenPorch', hash_bucket_size=100
)
ScreenPorch_emb = tf.feature_column.embedding_column(
    categorical_column=ScreenPorch, dimension=9)
PoolArea = tf.feature_column.categorical_column_with_hash_bucket(
    'PoolArea', hash_bucket_size=100
)
PoolArea_emb = tf.feature_column.embedding_column(
    categorical_column=PoolArea, dimension=9)
PoolQC = tf.feature_column.categorical_column_with_hash_bucket(
    'PoolQC', hash_bucket_size=100
)
PoolQC_emb = tf.feature_column.embedding_column(
    categorical_column=PoolQC, dimension=9)
Fence = tf.feature_column.categorical_column_with_hash_bucket(
    'Fence', hash_bucket_size=100
)
Fence_emb = tf.feature_column.embedding_column(
    categorical_column=Fence, dimension=9)


def get_base_column():
    base_column = [MiscFeature_emb, MiscVal_emb, MoSold_emb, YrSold_emb, SaleType_emb, SaleCondition_emb,
                   MSSubClass_emb, MSZoning_emb
        , LotFrontage_emb, LotArea_emb, Street_emb, Alley_emb, LotShape_emb, LandContour_emb, LotConfig_emb,
                   LandSlope_emb
        , Neighborhood_emb, OverallCond_emb, BldgType_emb, HouseStyle_emb, OverallQual_emb,
                   YearBuilt_emb
        , YearRemodAdd_emb, RoofStyle_emb, RoofMatl_emb, Exterior1st_emb, Exterior2nd_emb, MasVnrType_emb,
                   BsmtFinType1_emb
        , BsmtFinSF1_emb, BsmtFinType2_emb, BsmtFinSF2_emb, BsmtUnfSF_emb, TotalBsmtSF_emb, Heating_emb, HeatingQC_emb
        , Electrical_emb, lstFlrSF_emb, tondFlrSF_emb, Condition2_emb, LowQualFinSF_emb, GrLivArea_emb, BsmtFullBath_emb
        , BsmtHalfBath_emb, FullBath_emb, HalfBath_emb, BedroomAbvGr_emb, KitchenAbvGr_emb, KitchenQual_emb,
                   TotRmsAbvGrd_emb
        , Functional_emb, Fireplaces_emb, FireplaceQu_emb, GarageType_emb, GarageYrBlt_emb, GarageCars_emb,
                   GarageArea_emb
        , GarageQual_emb, GarageCond_emb, PavedDrive_emb, WoodDeckSF_emb, OpenPorchSF_emb, EnclosedPorch_emb,
                   tSsnPorch_emb
        , ScreenPorch_emb, PoolArea_emb, PoolQC_emb, Fence_emb]

    return base_column


# 构造输入数据
def construct_input_fn(data_path, branch_size, repeat, shuffle):
    def parse_csv(line):
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个转为字典类型{'a':'b'}。record_defaults用于指明每一列的缺失值用什么填充,
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS_NAME, columns))
        lables = features.pop('SalePrice')
        return features, lables

    # 返回数据
    dataset = tf.data.TextLineDataset(data_path).skip(1)
    dataset = dataset.batch(branch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=16)
    # 打乱顺序
    if shuffle:
        dataset = dataset.shuffle(100)

    dataset = dataset.repeat(repeat)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
