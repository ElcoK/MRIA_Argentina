import os
import pandas as pd
import numpy as np



def prepare_table_mria():

    data_path = os.path.join('..','data')

    # load table
    MRIO = pd.read_csv(os.path.join(data_path,'MRIO','mrio_argentina.csv'),index_col=[0,1],header=[0,1])

    Xnew = MRIO.copy()
    Xnew = Xnew+1e-6

    # write to excel
    writer = pd.ExcelWriter(os.path.join(data_path,'MRIO', 'mrio_argentina_disaggregated.xlsx'))

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

    prepare_table_mria()