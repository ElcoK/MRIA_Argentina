
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ras_method import ras_method
import subprocess

data_path = os.path.join('C:\\','Projects','MRIA_Argentina','data')

import warnings
warnings.filterwarnings('ignore')

def INDEC(set_year=2015,save_output=True):
  
  """
  Function to create a national Input-Output table using INDEC data as the baseline 

  Source for the supply-use table: https://www.indec.gob.ar/indec/web/Nivel4-Tema-3-9-114 
  Source for the Gross Value Added and Total Production time series: https://www.indec.gob.ar/indec/web/Nivel4-Tema-3-9-47
  """

  # load mapping function for industries
  ind_mapper = pd.read_excel(os.path.join(data_path,'INDEC',
                          'sh_cou_06_16.xls'),
                        sheet_name='ind_mapper',header=None)

  ind_mapper = dict(zip(ind_mapper[0],ind_mapper[1]))

  # load mapping function for products
  com_mapper = pd.read_excel(os.path.join(data_path,'INDEC',
                          'sh_cou_06_16.xls'),
                        sheet_name='com_mapper',header=None)
  com_mapper = dict(zip(com_mapper[0],['P_'+x for x in com_mapper[1]]))

  #create list of sectors
  sectors = [chr(i) for i in range(ord('A'),ord('P')+1)]

  """
  Load supply table and aggregate
  """

  sup_table_in = pd.read_excel(os.path.join(data_path,'INDEC',
                  'sh_cou_06_16.xls'), sheet_name='Mat Oferta pb',
                  skiprows=2,header=[0,1],index_col=[0,1],nrows=271)
  
  sup_table_in = sup_table_in.drop('Total',level=0,axis=1)
  
  sup_table = sup_table_in.copy()

  sup_table.columns = sup_table.columns.get_level_values(0)
  sup_table.columns = sup_table.columns.map(ind_mapper)
  sup_table = sup_table.T.groupby(level=0,axis=0).sum()
  sup_table.columns = sup_table.columns.get_level_values(0)
  sup_table.columns = sup_table.columns.map(com_mapper)
  sup_table = sup_table.T.groupby(level=0,axis=0).sum()

  """
  Load use table and aggregate
  """

  use_table = pd.read_excel(os.path.join(data_path,'INDEC',
                            'sh_cou_06_16.xls'),
                            sheet_name='Mat Utilizacion pc',
                            skiprows=2,header=[0,1],
                            index_col=[0,1],nrows=271)

  basic_prod_prices = use_table[[#'PRODUCCION NACIONAL A PRECIOS BASICOS',
                                'IMPORTACIONES  (CIF a nivel de producto y FOB a nivel total)',
                                'AJUSTE CIF/FOB DE LAS IMPORTACIONES','DERECHOS DE IMPORTACION',
                                'IMPUESTOS A LOS PRODUCTOS NETOS DE SUBSIDIOS','MARGENES DE COMERCIO',
                                'MARGENES DE TRANSPORTE','IMPUESTO AL VALOR AGREGADO NO DEDUCIBLE',
                                #'OFERTA TOTAL A PRECIOS DE  COMPRADOR'
                                  ]]*-1

  use_table = use_table.drop(['PRODUCCION NACIONAL A PRECIOS BASICOS',
                              'IMPORTACIONES  (CIF a nivel de producto y FOB a nivel total)',
                              'AJUSTE CIF/FOB DE LAS IMPORTACIONES','DERECHOS DE IMPORTACION',
                              'IMPUESTOS A LOS PRODUCTOS NETOS DE SUBSIDIOS','MARGENES DE COMERCIO',
                              'MARGENES DE TRANSPORTE','IMPUESTO AL VALOR AGREGADO NO DEDUCIBLE',
                              'OFERTA TOTAL A PRECIOS DE  COMPRADOR','UTILIZACION INTERMEDIA',
                              'UTILIZACION FINAL','DEMANDA TOTAL'],level=0,axis=1)

  # change to basic prices
  basic_prod_prices.columns = basic_prod_prices.columns.get_level_values(0)
  basic_prod_prices = basic_prod_prices.T.groupby(level=0,axis=0).sum()
  basic_prod_prices.columns = basic_prod_prices.columns.get_level_values(0)
  basic_prod_prices.columns = basic_prod_prices.columns.map(com_mapper)
  basic_prod_prices = basic_prod_prices.T.groupby(level=0,axis=0).sum()
  basic_prod_prices = basic_prod_prices.astype(int)

  use_table.columns = use_table.columns.get_level_values(0)
  use_table.columns = use_table.columns.map(ind_mapper)
  use_table = use_table.T.groupby(level=0,axis=0).sum()
  use_table.columns = use_table.columns.get_level_values(0)
  use_table.columns = use_table.columns.map(com_mapper)
  use_table = use_table.T.groupby(level=0,axis=0).sum()

  use_table= pd.concat([use_table,basic_prod_prices],axis=1)       

  """
  Create Industry-Industry IO table
  """                      
  
  # GET VARIABLES
  x = np.array(sup_table.sum(axis=0)) # total production on industry level
  g = np.array(sup_table.sum(axis=1)) # total production on product level
  F = use_table.iloc[:16,16:].sum(axis=1)

  #Numpify
  Sup_array = np.asarray(sup_table.iloc[:16,:16]) # numpy array of supply matrix
  Use_array = np.asarray(use_table.iloc[:16,:16]) # numpy array of use matrix

  g_diag_inv = np.linalg.inv(np.diag(g)) # inverse of g (and diagolinized)
  x_diag_inv = np.linalg.inv(np.diag(x)) # inverse of x (and diagolinized)

  # Calculate the matrices
  B = np.dot(Use_array,x_diag_inv) # B matrix (U*x^-1)
  D = np.dot(Sup_array.T,g_diag_inv) # D matrix (V*g^-1)
  I_i = np.identity((len(x))) # Identity matrix for industry-to-industry

  # Inverse for industry-to-industry
  A_ii = np.dot(D,B)
  IDB_inv = np.linalg.inv((I_i-np.dot(D,B))) # (I-DB)^-1 

  # And canclulate sum of industries
  ind = np.dot(IDB_inv,np.dot(D,F)/1e6) # (I-DB)^-1 * DF

  # split FD in local, import and export
  LFD = np.dot(D,use_table.iloc[:16,[16,18,19,21,22,23,24]].sum(axis=1) )/1e6
  Exp = np.dot(D,use_table.iloc[:16,17])/1e6
  Imp = np.dot(D,use_table.iloc[:16,20])/1e6

  # create combined table for the year 2004
  IO_ARG = pd.concat([pd.DataFrame(np.dot(A_ii,np.diag(ind))),
            pd.DataFrame(LFD),pd.DataFrame(Exp)],axis=1)

  IO_ARG.columns = list(use_table.columns[:18])
  IO_ARG.index = list(use_table.columns[:16])
  VA = np.array(list(ind)+[0,0])-np.array(IO_ARG.sum(axis=0))
  IMP = np.array(list(Imp*-1)+[0,0])
  VA[-2:] = 0
  IO_ARG.loc['ValueA'] = VA
  IO_ARG.loc['Imports'] = IMP
  IO_ARG.rename({'UTILIZACION FINAL':'FD',
                 'U_EXPORTACIONES':'EXP' },axis=1,inplace=True)
  IO_ARG[IO_ARG < 1e-5] = 0

  if set_year == 2004:
    return IO_ARG

  """
  Update table to preferred year
  """
  
  # load value added and total production time-series
  ValueA_series = pd.read_excel(os.path.join(data_path,
                              'INDEC','sh_VBP_VAB_06_19.xls'),
                              sheet_name='Cuadro 4',skiprows=3,index_col=[0])/1e3

  Total_Prod = pd.read_excel(os.path.join(data_path,
                            'INDEC','sh_VBP_VAB_06_19.xls'),
                            sheet_name='Cuadro 2',skiprows=3,index_col=[0])/1e3                  

  # split table 
  FD = np.array(IO_ARG['FD'][:16]*np.array((Total_Prod[2004]))/np.array(IO_ARG.sum(1)[:16]))
  Exports = np.array(IO_ARG['EXP'][:16][:16]*np.array((Total_Prod[2004]))/np.array(IO_ARG.sum(1)[:16]))
  Imports = np.array(IO_ARG.loc['Imports'][:16]*np.array((Total_Prod[2004]))/np.array(IO_ARG.sum(1)[:16]))
  ValueA = np.array(ValueA_series[2004])

  # convert to numpy matrix
  X0 = IO_ARG.values[:,:]

  # get sum of T
  u = np.array(list(Total_Prod[2004])+[sum(ValueA),sum(Imports)])
  v = np.array(list(Total_Prod[2004])+[x*((sum(ValueA)+sum(Imports))/(sum(FD)+sum(Exports))) for x in [sum(FD),sum(Exports)]])
  v[v < 0] = 0
  # and only keep T

  # apply RAS method to rebalance the table for 2004
  new_IO = ras_method(X0,u,v,1e-5,print_out=False)

  NEW_IO = pd.DataFrame(new_IO,index=sectors+['ValueA','Imports'],columns=sectors+['FD','EXP'])

  for year in [int(x) for x in np.linspace(2004, set_year, set_year-2004)]:

    FD = np.array(NEW_IO['FD'][:16]*np.array((Total_Prod[year]))/np.array(NEW_IO.sum(1)[:16]))
    Exports = np.array(NEW_IO['EXP'][:16][:16]*np.array((Total_Prod[year]))/np.array(NEW_IO.sum(1)[:16]))
    Imports = np.array(NEW_IO.loc['Imports'][:16]*np.array((Total_Prod[year]))/np.array(NEW_IO.sum(1)[:16]))
    ValueA = np.array(ValueA_series[year])

    # convert to numpy matrix
    X0 = NEW_IO.values[:,:]

    # get sum of T
    u = np.array(list(Total_Prod[year])+[sum(ValueA),sum(Imports)])
    v = np.array(list(Total_Prod[year])+[x*((sum(ValueA)+sum(Imports))/(sum(FD)+sum(Exports))) for x in [sum(FD),sum(Exports)]])
    v[v < 0] = 0
    # and only keep T

    # apply RAS method to rebalance the table
    new_IO = ras_method(X0,u,v,1e-5,print_out=False)
    
    INDEC = pd.DataFrame(new_IO,index=sectors+['ValueA','Imports'],columns=sectors+['FD','EXP'])*1e3

  if save_output:
    INDEC.to_csv(os.path.join(data_path,'national_tables','{}_INDEC.csv'.format(set_year)))
  
  return INDEC

def OECD(save_output=True):
  """
  Function to create a national IO-table using OECD data 
  """

  # load mapping function for industries
  mapper_OECD = pd.read_excel(os.path.join(data_path,'other_sources','mappers.xlsx'),
                              sheet_name="OECD_INDEC")
  mapper_OECD = dict(zip(mapper_OECD['OECD'],mapper_OECD['INDEC'])) 

  # load table:
  OECD = pd.read_excel(os.path.join(data_path,'OECD','OECD_Argentina_2015.xlsx'),skiprows=6,index_col=[0])
  OECD.index = [x.split(':')[1] for x in OECD.index]
  OECD.columns = [x.split(':')[1] for x in OECD.columns]

  
  OECD.columns = OECD.columns.map(mapper_OECD)
  OECD = OECD.groupby(level=0,axis=1).sum()
  OECD.index = OECD.index.map(mapper_OECD)
  OECD = OECD.groupby(level=0,axis=0).sum()
  OECD.loc['B',:] = 0
  OECD['B'] = 0

  OECD = OECD[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['FinalD','Exports']]
  OECD = OECD.T[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['ValueA','Imports']]
  OECD = OECD.T*10

  if save_output:
    OECD.to_csv(os.path.join(data_path,'national_tables','2015_OECD.csv'))
  
  return OECD

def EORA(save_output=True):
  """
  Function to create a national IO-table using EORA data as the baseline 
  """

  # load mapping function for industries
  mapper_EORA = pd.read_excel(os.path.join(data_path,'other_sources','mappers.xlsx'),
                              sheet_name="EORA_INDEC")
  mapper_EORA = dict(zip(mapper_EORA['EORA'],mapper_EORA['INDEC']))

  ### Load supply and use table
  EORA_SUP = pd.read_excel(os.path.join(data_path,'EORA',
            'EORA_IO_ARG_2015_BasicPrice.xlsx'),sheet_name='SUP',
            index_col=[0,1,2],header=[0,1,2])

  EORA_USE = pd.read_excel(os.path.join(data_path,'EORA',
            'EORA_IO_ARG_2015_BasicPrice.xlsx'),sheet_name='USE',
            index_col=[0,1,2],header=[0,1,2]).fillna(0)

  # GET VARIABLES
  x = np.array(EORA_SUP.sum(axis=1)) # total production on industry level
  g = np.array(EORA_SUP.sum(axis=0)) # total production on product level
  F = EORA_USE.iloc[:196,125:].sum(axis=1)

  #Numpify
  Sup_array = np.asarray(EORA_SUP.iloc[:125,:196]) # numpy array of supply matrix
  Use_array = np.asarray(EORA_USE.iloc[:196,:125]) # numpy array of use matrix

  g_diag_inv = np.linalg.inv(np.diag(g)) # inverse of g (and diagolinized)
  x_diag_inv = np.linalg.inv(np.diag(x)) # inverse of x (and diagolinized)

  # Calculate the matrices
  B = np.dot(Use_array,x_diag_inv) # B matrix (U*x^-1)
  D = np.dot(Sup_array,g_diag_inv) # D matrix (V*g^-1)
  I_i = np.identity((len(x))) # Identity matrix for industry-to-industry

  # Inverse for industry-to-industry
  A_ii = np.dot(D,B)
  IDB_inv = np.linalg.inv((I_i-np.dot(D,B))) # (I-DB)^-1 

  # And canclulate sum of industries
  ind = np.dot(IDB_inv,np.dot(D,F)) # (I-DB)^-1 * DF

  # split FD in local, import and export
  LFD = np.dot(D,EORA_USE.iloc[:196,125:131].sum(axis=1) )
  Exp = np.dot(D,EORA_USE.iloc[:196,131])

  # combine all elements into one table
  EORA = pd.concat([pd.DataFrame(np.dot(A_ii,np.diag(ind))),pd.DataFrame(LFD),
                  pd.DataFrame(Exp)],axis=1)
  EORA.columns = [x[2] for x in EORA_USE.columns[:125]]+['FinalD','Exports']
  EORA.index = [x[2] for x in EORA_USE.columns[:125]]

  VA = list(EORA_USE.iloc[196:202,:125].sum(axis=0))+[0,0]
  IMP = list(EORA_USE.iloc[202,:125])+[0,0]
  EORA.loc['ValueA'] = VA
  EORA.loc['Imports'] = IMP
  EORA[EORA < 1e-5] = 0

  # and map into the INDEC classes:
  EORA.columns = EORA.columns.map(mapper_EORA)
  EORA = EORA.groupby(level=0,axis=1).sum()
  EORA.index = EORA.index.map(mapper_EORA)
  EORA = EORA.groupby(level=0,axis=0).sum()

  EORA = EORA[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['FinalD','Exports']]
  EORA = EORA.T[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['ValueA','Imports']]
  EORA = EORA.T

  # and balance the table
  X0 = EORA.values[:,:]

  # get sum of T
  # u = np.array(list(EORA.sum(axis=1)[:16])+[EORA.loc['ValueA'].sum(),
  #                 EORA.loc['Imports'].sum()])
  # v = np.array(list(EORA.sum(axis=1)[:16])+[EORA['FinalD'].sum(),EORA['Exports'].sum()])

  # and only keep T

  # apply RAS method to rebalance the table
  new_IO = ras_method(X0,X0.sum(axis=1),X0.sum(axis=1),1e-5,print_out=False)

  EORA.iloc[:,:] = new_IO/1e6*10
  
  if save_output:
    EORA.to_csv(os.path.join(data_path,'national_tables','2015_EORA.csv'))
  
  return EORA

def GTAP(save_output=True):
  """
  Function to create a national IO-table using GTAP data as the baseline 
  """

  # load mapping function for industries  
  mapper_GTAP = pd.read_excel(os.path.join(data_path,'other_sources','mappers.xlsx'),
                              sheet_name="GTAP_INDEC")
  mapper_GTAP = dict(zip(mapper_GTAP['GTAP'],mapper_GTAP['INDEC']))

  ### Load supply and use table
  GTAP_SUP = pd.read_excel(os.path.join(data_path,'GTAP','GTAP_Argentina_2014.xlsx'),
            sheet_name='Supply',index_col=[0],header=[0]).fillna(0)
  GTAP_USE = pd.read_excel(os.path.join(data_path,'GTAP','GTAP_Argentina_2014.xlsx'),
            sheet_name='Use',index_col=[0],header=[0]).fillna(0)

  # GET VARIABLES
  x = np.array(GTAP_SUP.sum(axis=1)) # total production on industry level
  g = np.array(GTAP_SUP.sum(axis=0)) # total production on product level
  F = GTAP_USE.iloc[:65,65:].sum(axis=1)

  #Numpify
  Sup_array = np.asarray(GTAP_SUP.iloc[:65,:65]) # numpy array of supply matrix
  Use_array = np.asarray(GTAP_USE.iloc[:65,:65]) # numpy array of use matrix

  g_diag_inv = np.linalg.inv(np.diag(g)) # inverse of g (and diagolinized)
  x_diag_inv = np.linalg.inv(np.diag(x)) # inverse of x (and diagolinized)

  # Calculate the matrices
  B = np.dot(Use_array,x_diag_inv) # B matrix (U*x^-1)
  D = np.dot(Sup_array,g_diag_inv) # D matrix (V*g^-1)
  I_i = np.identity((len(x))) # Identity matrix for industry-to-industry

  # Inverse for industry-to-industry
  A_ii = np.dot(D,B)
  IDB_inv = np.linalg.inv((I_i-np.dot(D,B))) # (I-DB)^-1 

  # And canclulate sum of industries
  ind = np.dot(IDB_inv,np.dot(D,F)) # (I-DB)^-1 * DF

  # split FD in local, import and export
  LFD = np.dot(D,GTAP_USE.iloc[:65,65:68].sum(axis=1) )
  Exp = np.dot(D,GTAP_USE.iloc[:65,68])

  # combine all elements into one table
  GTAP = pd.concat([pd.DataFrame(np.dot(A_ii,np.diag(ind))),pd.DataFrame(LFD),
                    pd.DataFrame(Exp)],axis=1)
  GTAP.columns = [x for x in GTAP_USE.columns[:65]]+['FinalD','Exports']
  GTAP.index = [x for x in GTAP_USE.columns[:65]]

  VA = list(GTAP_USE.iloc[65,:65])+[0,0]
  IMP = list(GTAP_USE.iloc[66,:65])+[0,0]
  GTAP.loc['ValueA'] = VA
  GTAP.loc['Imports'] = IMP
  GTAP[GTAP < 1e-5] = 0

  # and map into the INDEC classes:
  GTAP.columns = GTAP.columns.map(mapper_GTAP)
  GTAP = GTAP.groupby(level=0,axis=1).sum()
  GTAP.index = GTAP.index.map(mapper_GTAP)
  GTAP = GTAP.groupby(level=0,axis=0).sum()

  GTAP = GTAP[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['FinalD','Exports']]
  GTAP = GTAP.T[[chr(i).upper() for i in range(ord('a'),ord('p')+1)]+['ValueA','Imports']]
  GTAP = GTAP.T*8

  if save_output:
    GTAP.to_csv(os.path.join(data_path,'national_tables','2015_GTAP.csv'))

  return GTAP

if __name__ == "__main__":

  print('INDEC 2015 ARS: {}'.format(INDEC().sum().sum()))
  print('OECD 2015 ARS: {}'.format(OECD().sum().sum()))
  print('EORA 2015 ARS: {}'.format(EORA().sum().sum()))
  print('GTAP 2014 ARS: {}'.format(GTAP().sum().sum()))