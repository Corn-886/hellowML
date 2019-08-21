import pandas as pd
import json

def main():
    bufs = '{"holdstock":[{"name":"格力电器","code":"000651"},{"name":"贵州茅台","code":"600519"}],"manager":[],"managercom":[{"name":"易方达新经济","code":"易方达基金管理有限公司"},{"name":"易方达生物","code":"易方达基金管理有限公司"},{"name":"易方达新丝路","code":"易方达基金管理有限公司"},{"name":"易方达重组","code":"易方达基金管理有限公司"},{"name":"易方达消费行业","code":"易方达基金管理有限公司"},{"name":"创业ETF联接A","code":"易方达基金管理有限公司"}],"industri":"电子|家用电器|食品饮料","holdclass":"","holdcycle":""}'
    cls = json.loads(bufs)
    print(cls['holdstock'])
    for i in range(len(cls['holdstock'])):

        print(cls['holdstock'][i]['code'])



if __name__ == '__main__':
    main()
