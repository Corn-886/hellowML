import pandas as pd
from pandas.io.json import json_normalize
import json


def main():
    data_file = '/Users/suyuming/Desktop/datarecome.csv'
    names = ['cust_pty_no', 'branch_no', ' assetstype', 'assetstype_name', 'assetstype_pref', 'assetstype_rank',
             'prod_id', 'fund_code ', 'fund_name', 'fund_score', ' fund_rank', 'holdingrelation', 'behavioralprdf',
             'prodperformance',
             'busi_date']
    df = pd.read_csv(data_file, names=names, sep=' ')

    for i, row in df.iterrows():
        #提取数据
        holdJ = json.loads(row['holdingrelation'])
        #提取客户持仓股票
        for i1 in range(len(holdJ['holdstock'])):
            df.at[i1, 'hold_code_' + str(i1)] = holdJ['holdstock'][i1]['code']
        #提取客户偏好持仓管理公司
        for i2 in range(len(holdJ['managercom'])):
            df.at[i2, 'managercom_code_' + str(i2)] = holdJ['managercom'][i2]['code']


    print(df)


if __name__ == '__main__':
    main()
