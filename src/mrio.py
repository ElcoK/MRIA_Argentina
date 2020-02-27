import os,sys
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from ras_method import ras_method

import warnings
warnings.filterwarnings('ignore')

def est_trade_value(x,output_new,sector):
    """
    Function to estimate the trade value between two sectors
    """
    if (sector is not 'other1') & (sector is not 'other2'):
        sec_output = output_new.sum(axis=1).loc[output_new.sum(axis=1).index.get_level_values(1) == sector].reset_index()
    else:
        sec_output = output_new.sum(axis=1).loc[output_new.sum(axis=1).index.get_level_values(1) == 'IMP'].reset_index()
    x['gdp'] = x.gdp*min(sec_output.loc[sec_output.region==x.reg1].values[0][2],sec_output.loc[sec_output.region==x.reg2].values[0][2])
    return x


def estimate(table='INDEC',print_output=False):
    """
    Function to create a province-level MRIO table, based on a national IO table. The default is the INDEC table.
    """
    data_path = os.path.join('..','data')

    # load sector data
    sectors = list(pd.read_excel(os.path.join(data_path,'other_sources',
                    'industry_high_level_classification.xlsx'))['SEC_CODE'].values)

    # load provincial mappers
    reg_mapper = pd.read_excel(os.path.join(data_path,'INDEC','sh_cou_06_16.xls'),sheet_name='reg_mapper',header=None).iloc[:,:2]
    reg_mapper = dict(zip(reg_mapper[0],reg_mapper[1]))

    # load provincial data
    prov_data = pd.read_excel(os.path.join(data_path,'INDEC','PIB_provincial_06_17.xls'),sheet_name='VBP',
                         skiprows=3,index_col=[0],header=[0],nrows=71)
    prov_data = prov_data.loc[[x.isupper() for x in prov_data.index],:]
    prov_data.columns = [x.replace(' ','_') for x in ['Ciudad de Buenos Aires', 'Buenos Aires', 'Catamarca', 'Cordoba',
        'Corrientes', 'Chaco', 'Chubut', 'Entre Rios', 'Formosa', 'Jujuy',
        'La Pampa', 'La Rioja', 'Mendoza', 'Misiones', 'Neuquen', 'Rio Negro',
        'Salta', 'San Juan', 'San Luis', 'Santa Cruz', 'Santa Fe',
        'Santiago del Estero', 'Tucuman', 'Tierra del Fuego',
        'No distribuido', 'Total']]
    region_names = list(prov_data.columns)[:-2]

    prov_data.index = sectors+['TOTAL']
    prov_data = prov_data.replace(0, 1)

    ### Create proxy data for first iteration

    sectors+['other1','other2']

    # proxy level 2
    proxy_reg_arg = pd.DataFrame(prov_data.iloc[-1,:24]/prov_data.iloc[-1,:24].sum()).reset_index()
    proxy_reg_arg['year'] = 2016
    proxy_reg_arg = proxy_reg_arg[['year','index','TOTAL']]
    proxy_reg_arg.columns = ['year','id','gdp']
    proxy_reg_arg.to_csv(os.path.join('..','mrio_downscaling','proxy_reg_arg.csv'),index=False)

    # proxy level 4
    for iter_,sector in enumerate(sectors+['other1','other2']):
        if (sector is not 'other1') & (sector is not 'other2'):
            proxy_sector = pd.DataFrame(prov_data.iloc[iter_,:24]/prov_data.iloc[iter_,:24].sum()).reset_index()
            proxy_sector['year'] = 2016
            proxy_sector['sector'] = 'sec{}'.format(sector)
            proxy_sector = proxy_sector[['year','sector','index',sector]]
            proxy_sector.columns = ['year','sector','region','gdp']
            proxy_sector.to_csv(os.path.join('..','mrio_downscaling','proxy_sec{}.csv'.format(sector)),index=False)
        else:
            proxy_sector = pd.DataFrame(prov_data.iloc[-1,:24]/prov_data.iloc[-1,:24].sum()).reset_index()
            proxy_sector['year'] = 2016
            proxy_sector['sector'] = sector+'1'
            proxy_sector = proxy_sector[['year','sector','index','TOTAL']]
            proxy_sector.columns = ['year','sector','region','gdp']
            proxy_sector.to_csv(os.path.join('..','mrio_downscaling','proxy_{}.csv'.format(sector)),index=False)

    # proxy level 18

    def change_name(x):
        if x in sectors:
            return 'sec'+x
        elif x == 'other1':
            return 'other11'
        else:
            return 'other21' 

    mi_index = pd.MultiIndex.from_product([sectors+['other1','other2'], region_names, sectors+['other1','other2'], region_names],
                                        names=['sec1', 'reg1','sec2','reg2'])
    for iter_,sector in enumerate(sectors+['other1','other2']):
        if (sector is not 'other1') & (sector is not 'other2'):
            proxy_trade = pd.DataFrame(columns=['year','gdp'],index= mi_index).reset_index()
            proxy_trade['year'] = 2016
            proxy_trade['gdp'] = 0
            proxy_trade = proxy_trade.query("reg1 != reg2")
            proxy_trade = proxy_trade.loc[proxy_trade.sec1 == sector]
            proxy_trade['sec1'] = proxy_trade.sec1.apply(change_name)
            proxy_trade['sec2'] = proxy_trade.sec2.apply(change_name)
            proxy_trade = proxy_trade[['year','sec1','reg1','sec2','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','sector','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade_sec{}.csv'.format(sector)),index=False)    
        else:
            proxy_trade = pd.DataFrame(columns=['year','gdp'],index= mi_index).reset_index()
            proxy_trade['year'] = 2016
            proxy_trade['gdp'] = 0
            proxy_trade = proxy_trade.query("reg1 != reg2")    
            proxy_trade = proxy_trade.loc[proxy_trade.sec1 == sector]
            proxy_trade['sec1'] = proxy_trade.sec1.apply(change_name)
            proxy_trade['sec2'] = proxy_trade.sec2.apply(change_name)       
            proxy_trade = proxy_trade[['year','sec1','reg1','sec2','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','sector','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade_{}.csv'.format(sector)),index=False)

    """
    Create first version of MRIO for Argentina, without trade
    """

    ### run libmrio
    p = subprocess.Popen([r'..\mrio_downscaling\mrio_disaggregate', 'settings_notrade.yml'],
                    cwd=os.path.join('..','mrio_downscaling'))
    p.wait()

    ### load data and reorder
    region_names_list = [item for sublist in [[x]*(len(sectors)+2) for x in region_names]
                     for item in sublist]

    rows = ([x for x in sectors+['VA','IMP']])*len(region_names)
    cols = ([x for x in sectors+['FD','EXP']])*len(region_names)

    index_mi = pd.MultiIndex.from_arrays([region_names_list, rows], names=('region', 'row'))
    column_mi = pd.MultiIndex.from_arrays([region_names_list, cols], names=('region', 'col'))

    MRIO = pd.read_csv(os.path.join('..','mrio_downscaling','output1.csv'),header=None,index_col=None)
    MRIO.index = index_mi
    MRIO.columns = column_mi

    # create predefined index and col, which is easier to read
    sector_only = [x for x in sectors]*len(region_names)
    col_only = ['FD']*len(region_names)

    region_col = [item for sublist in [[x]*len(sectors) for x in region_names] for item in sublist] + \
        [item for sublist in [[x]*1 for x in region_names] for item in sublist]

    column_mi_reorder = pd.MultiIndex.from_arrays(
        [region_col, sector_only+col_only], names=('region', 'col'))

    # sum va and imports
    valueA = MRIO.xs('VA', level=1, axis=0).sum(axis=0)
    valueA.drop('FD', level=1,axis=0,inplace=True)
    valueA.drop('EXP', level=1,axis=0,inplace=True)
    imports = MRIO.xs('IMP', level=1, axis=0).sum(axis=0)
    imports.drop('FD', level=1,axis=0,inplace=True)
    imports.drop('EXP', level=1,axis=0,inplace=True)
    FinalD = MRIO.xs('FD', level=1, axis=1).sum(axis=1)
    FinalD.drop('VA', level=1,axis=0,inplace=True)
    FinalD.drop('IMP', level=1,axis=0,inplace=True)
    Export = MRIO.xs('EXP', level=1, axis=1).sum(axis=1)
    Export.drop('VA', level=1,axis=0,inplace=True)
    Export.drop('IMP', level=1,axis=0,inplace=True)

    output_new = MRIO.copy()

    """
    Balance first MRIO version
    """

    # convert to numpy matrix
    X0 = MRIO.as_matrix()

    # get sum of rows and columns
    u = X0.sum(axis=1)
    v = X0.sum(axis=0)

    # and only keep T
    v[:(len(u)-2)] = u[:-2]

    # apply RAS method to rebalance the table
    X1 = ras_method(X0, u, v, eps=1e-5,print_out=print_output)

    #translate to pandas dataframe
    output_new = pd.DataFrame(X1)
    output_new.index = index_mi
    output_new.columns = column_mi


    """
    Create second version of MRIO for Argentina, with trade
    """

    ### Load OD matrix

    od_matrix_total = pd.DataFrame(pd.read_excel(os.path.join(data_path,'OD_data','province_ods.xlsx'),
                            sheet_name='total',index_col=[0,1],usecols =[0,1,2,3,4,5,6,7])).unstack(1).fillna(0)
    od_matrix_total.columns.set_levels(['A','G','C','D','B','I'],level=0,inplace=True)
    od_matrix_total.index = od_matrix_total.index.map(reg_mapper)
    od_matrix_total = od_matrix_total.stack(0)
    od_matrix_total.columns = od_matrix_total.columns.map(reg_mapper)
    od_matrix_total = od_matrix_total.swaplevel(i=-2, j=-1, axis=0)
    od_matrix_total = od_matrix_total.loc[:, od_matrix_total.columns.notnull()]

    ### Create proxy data
    # proxy level 14 
    mi_index = pd.MultiIndex.from_product([sectors+['other1','other2'], region_names, region_names],
                                        names=['sec1', 'reg1','reg2'])

    for iter_,sector in enumerate(tqdm(sectors+['other1','other2'])):
        if sector in ['A','G','C','D','B','I']:
            proxy_trade = (od_matrix_total.sum(level=1).divide(od_matrix_total.sum(level=1).sum(axis=1),axis='rows')).stack(0).reset_index()
            proxy_trade.columns = ['reg1','reg2','gdp']
            proxy_trade['year'] = 2016
            proxy_trade = proxy_trade.apply(lambda x: est_trade_value(x,output_new,sector),axis=1)
            proxy_trade['sec1'] = 'sec{}'.format(sector)
            proxy_trade = proxy_trade[['year','sec1','reg1','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade14_sec{}.csv'.format(sector)),index=False)
        elif (sector is not 'other1') &  (sector is not 'other2') & (sector not in ['A','G','C','D','B','I']): # &  (sector not in ['L','M','N','O','P']):
            proxy_trade = (od_matrix_total.sum(level=1).divide(od_matrix_total.sum(level=1).sum(axis=1),axis='rows')).stack(0).reset_index()
            #proxy_trade[0].loc[(proxy_trade.origin_province == proxy_trade.destination_province)] = 0.9
            #proxy_trade[0].loc[~(proxy_trade.origin_province == proxy_trade.destination_province)] = 0.1
            proxy_trade.columns = ['reg1','reg2','gdp']
            proxy_trade['year'] = 2016
            proxy_trade = proxy_trade.apply(lambda x: est_trade_value(x,output_new,sector),axis=1)
            proxy_trade['sec1'] = 'sec{}'.format(sector)
            proxy_trade = proxy_trade[['year','sec1','reg1','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade14_sec{}.csv'.format(sector)),index=False)

        else:
            proxy_trade = (od_matrix_total.sum(level=1).divide(od_matrix_total.sum(level=1).sum(axis=1),axis='rows')).stack(0).reset_index()
            proxy_trade.columns = ['reg1','reg2','gdp']
            proxy_trade['year'] = 2016
            proxy_trade = proxy_trade.apply(lambda x: est_trade_value(x,output_new,sector),axis=1)
            proxy_trade['sec1'] = sector+'1'
            proxy_trade = proxy_trade[['year','sec1','reg1','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade14_{}.csv'.format(sector)),index=False)    

    # proxy level 18
    mi_index = pd.MultiIndex.from_product([sectors+['other1','other2'], region_names, sectors+['other1','other2'], region_names],
                                        names=['sec1', 'reg1','sec2','reg2'])
    for iter_,sector in enumerate(tqdm(sectors+['other1','other2'])):
        if (sector is not 'other1') & (sector is not 'other2'):
            proxy_trade = pd.DataFrame(columns=['year','gdp'],index= mi_index).reset_index()
            proxy_trade['year'] = 2016
            proxy_trade['gdp'] = 0
            proxy_trade = proxy_trade.query("reg1 != reg2")
            proxy_trade = proxy_trade.loc[proxy_trade.sec1 == sector]
            proxy_trade = proxy_trade.loc[proxy_trade.sec2.isin(['L','M','N','O','P'])]
            proxy_trade['sec1'] = proxy_trade.sec1.apply(change_name)
            proxy_trade['sec2'] = proxy_trade.sec2.apply(change_name) 
            
            proxy_trade = proxy_trade.query("reg1 == reg2")    

            proxy_trade = proxy_trade[['year','sec1','reg1','sec2','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','sector','region','gdp']
            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade_sec{}.csv'.format(sector)),index=False)
        
        else:
            proxy_trade = pd.DataFrame(columns=['year','gdp'],index= mi_index).reset_index()
            proxy_trade['year'] = 2016
            proxy_trade['gdp'] = 0
            proxy_trade = proxy_trade.query("reg1 != reg2")    
            proxy_trade = proxy_trade.loc[proxy_trade.sec1 == sector]
            proxy_trade = proxy_trade.loc[proxy_trade.sec2.isin(['L','M','N','O','P'])]
            proxy_trade['sec1'] = proxy_trade.sec1.apply(change_name)
            proxy_trade['sec2'] = proxy_trade.sec2.apply(change_name) 
            
            proxy_trade = proxy_trade.query("reg1 == reg2")    

            proxy_trade = proxy_trade[['year','sec1','reg1','sec2','reg2','gdp']]
            proxy_trade.columns = ['year','sector','region','sector','region','gdp']

            proxy_trade.to_csv(os.path.join('..','mrio_downscaling','proxy_trade_{}.csv'.format(sector)),index=False)   


    ### run libmrio
    p = subprocess.Popen([r'..\mrio_downscaling\mrio_disaggregate', 'settings_trade.yml'],
            cwd=os.path.join('..','mrio_downscaling'))
    p.wait()

    # load data and reorder
    region_names_list = [item for sublist in [[x]*(len(sectors)+2) for x in region_names]
                     for item in sublist]

    rows = ([x for x in sectors+['VA','IMP']])*len(region_names)
    cols = ([x for x in sectors+['FD','EXP']])*len(region_names)

    index_mi = pd.MultiIndex.from_arrays([region_names_list, rows], names=('region', 'row'))
    column_mi = pd.MultiIndex.from_arrays([region_names_list, cols], names=('region', 'col'))

    MRIO = pd.read_csv(os.path.join('..','mrio_downscaling','output2.csv'),header=None,index_col=None)
    MRIO.index = index_mi
    MRIO.columns = column_mi

    # create predefined index and col, which is easier to read
    sector_only = [x for x in sectors]*len(region_names)
    col_only = ['FD','EXP']*len(region_names)

    region_col = [item for sublist in [[x]*len(sectors) for x in region_names] for item in sublist] + \
        [item for sublist in [[x]*2 for x in region_names] for item in sublist]

    column_mi_reorder = pd.MultiIndex.from_arrays(
        [region_col, sector_only+col_only], names=('region', 'col'))

    # sum va and imports
    valueA = pd.DataFrame(MRIO.loc[MRIO.index.get_level_values(1) == 'VA'].sum(axis='index'))
    valueA.columns = pd.MultiIndex.from_product([['Total'],['ValueA']],names=['region','row'])

    IMP = pd.DataFrame(MRIO.loc[MRIO.index.get_level_values(1) == 'IMP'].sum(axis='index'))
    IMP.columns = pd.MultiIndex.from_product([['Total'],['IMP']],names=['region','row'])

    output = pd.concat([MRIO.loc[~MRIO.index.get_level_values(1).isin(['FD','EXP'])]])
    output = output.drop(['VA','IMP'], level=1)
    output = pd.concat([output,valueA.T,IMP.T])

    output = output.reindex(column_mi_reorder, axis='columns')

    mrio_arg = ras_method(np.array(output).T,np.array(list(output.sum(axis=1))[:384]+list(output.sum(axis=0)[-48:])),
                        np.array(list(output.sum(axis=1))[:384]+[output.loc[('Total','ValueA'),:].sum(),output.loc[('Total','IMP'),:].sum()]), 
                        eps=1e-3,print_out=print_output)

    mrio_argentina = pd.DataFrame(mrio_arg.T,index=output.index,columns=output.columns)

    mrio_argentina.to_csv(os.path.join(data_path,'MRIO','mrio_argentina.csv'))


def prepare_table_mria():
    """
    Convert MRIO table to an excel file in which all elements of the table are disaggregated.
    """
    data_path = os.path.join('..','data')

    # load table
    MRIO = pd.read_csv(os.path.join(data_path,'MRIO','mrio_argentina.csv'),index_col=[0,1],header=[0,1])

    Xnew = MRIO.copy()
    Xnew = Xnew+1e-6

    # write to excel
    writer = pd.ExcelWriter(os.path.join(data_path,'MRIO','mrio_argentina_disaggregated.xlsx'))

    # write T
    df_T = Xnew.iloc[:384, :384]
    df_T.columns = df_T.columns.droplevel()
    df_labels_T = pd.DataFrame(df_T.reset_index()[['region', 'row']])
    df_T.reset_index(inplace=True, drop=True)
    df_T.to_excel(writer, 'T', index=False, header=False)
    df_labels_T.to_excel(writer, 'labels_T', index=False, header=False)

    # write FD
    df_FD = Xnew.iloc[:384, 384:].iloc[:, Xnew.iloc[:384, 384:].columns.get_level_values(1)=='FD']
    df_labels_FD = pd.DataFrame(list(df_FD.columns))
    df_FD.columns = df_FD.columns.droplevel()
    df_FD.reset_index(inplace=True, drop=True)
    df_FD.to_excel(writer, 'FD', index=False, header=False)
    df_labels_FD.to_excel(writer, 'labels_FD', index=False, header=False)

    # write ExpROW
    df_ExpROW =  pd.DataFrame(Xnew.iloc[:384, 384:].iloc[:, Xnew.iloc[:384, 384:].columns.get_level_values(1)=='EXP'].sum(axis=1))
    df_labels_ExpROW = pd.DataFrame(['Export'])
    df_ExpROW.reset_index(inplace=True, drop=True)
    df_ExpROW.to_excel(writer, 'ExpROW', index=False, header=False)
    df_labels_ExpROW.reset_index(inplace=True, drop=True)
    df_labels_ExpROW.columns = ['Export']
    df_labels_ExpROW.to_excel(writer, 'labels_ExpROW', index=False, header=False)

    # write VA
    df_VA = pd.DataFrame(Xnew.iloc[384:, :409].T[('Total', 'ValueA')])
    df_VA.columns = ['VA']
    df_VA['imports'] = pd.DataFrame(Xnew.iloc[384:, :].T[('Total', 'IMP')])
    df_VA.reset_index(inplace=True, drop=True)
    df_VA.to_excel(writer, 'VA', index=False, header=False)
    df_labels_VA = pd.DataFrame(['Import', 'VA']).T
    df_labels_VA.to_excel(writer, 'labels_VA', index=False, header=False)

    # save excel
    writer.save()


if __name__ == "__main__":

    estimate()
    prepare_table_mria()