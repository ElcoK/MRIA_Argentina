import os,sys
import pyomo
import numpy as np
import pandas as pd
from tqdm import tqdm

from national_io import INDEC,GTAP,OECD,EORA
from mrio import estimate,prepare_table_mria
from mria import MRIA_IO as MRIA
from table import io_basic

def run(data_source='INDEC',
        set_year=2015,solver='IPOPT',   
        disr_dict_fd = {},
        disr_dict_sup = {},
        print_output=False,
        loss_in_va=True,
        ):

    # set data paths
    data_path = os.path.join('..','data')

    output_path = os.path.join('..','results')

    """Specify disruption"""
    output_dir = os.path.join(
        output_path,
        'agriculture_impacts')

    ### Get regional mapper dictionary, to make everything compatible
    reg_mapper = pd.read_excel(os.path.join(data_path,'INDEC','sh_cou_06_16.xls'),
                          sheet_name='reg_mapper',header=None)
    reg_mapper = dict(zip(reg_mapper[0], reg_mapper[1]))

    if not os.path.exists(os.path.join(data_path,'MRIO', 'mrio_argentina_disaggregated_{}_{}.xlsx'.format(data_source,set_year))):

        ### Create MRIO table from national database
        if data_source == 'INDEC':
            INDEC()
        elif data_source == 'OECD':
            OECD()
        elif data_source == 'GTAP':
            GTAP()
        elif data_source == 'EORA':
            EORA()

        estimate(table=data_source, year = set_year)
        prepare_table_mria(table=data_source, year = set_year)    

    ### Create input-output database file to be used with optimization model
    """ Specify file path """
    filepath = os.path.join(data_path,'MRIO', 'mrio_argentina_disaggregated_{}_{}.xlsx'.format(data_source,set_year))

    regions = ['Ciudad de Buenos Aires', 'Buenos Aires', 'Catamarca', 'Cordoba',
        'Corrientes', 'Chaco', 'Chubut', 'Entre Rios', 'Formosa', 'Jujuy',
        'La Pampa', 'La Rioja', 'Mendoza', 'Misiones', 'Neuquen', 'Rio Negro',
        'Salta', 'San Juan', 'San Luis', 'Santa Cruz', 'Santa Fe',
        'Santiago del Estero', 'Tucuman', 'Tierra del Fuego']

    regions = [x.replace(' ','_') for x in regions]

    """Create data input"""
    DATA = io_basic('Argentina', filepath,regions)
    DATA.prep_data()

    ###RUN MRIA model for baseline

    output_in = pd.DataFrame()

    disr_dict_fd_base = {}
    disr_dict_sup_base = {}

    """Create model"""
    MRIA_RUN = MRIA(DATA.name, DATA.regions, DATA.sectors, data_source, list_fd_cats=['FD'])

    """Define sets and alias"""
    # CREATE SETS
    MRIA_RUN.create_sets()

    # CREATE ALIAS
    MRIA_RUN.create_alias()

    """ Define tables and parameters"""
    MRIA_RUN.baseline_data(DATA, disr_dict_sup_base, disr_dict_fd_base)
    MRIA_RUN.impact_data(DATA, disr_dict_sup_base, disr_dict_fd_base)

    status = MRIA_RUN.run_impactmodel(solver=solver)

    """Get base line values"""
    output_in['x_in'] = pd.Series(MRIA_RUN.X.get_values())
    output_in.index.names = ['region', 'sector']


    ###Run impact model
    collect_outputs = {}

    total_losses = {}

    event = '{}_{}'.format(data_source,set_year)

    output = output_in.copy()

    """Run model and create some output"""
    

    """Create model"""
    MRIA_RUN = MRIA(DATA.name, DATA.regions, DATA.sectors, data_source, list_fd_cats=['FD'])

    """Define sets and alias"""
    # CREATE SETS
    MRIA_RUN.create_sets()

    # CREATE ALIAS
    MRIA_RUN.create_alias()
    """ Define tables and parameters"""
    MRIA_RUN.baseline_data(DATA, disr_dict_sup, disr_dict_fd)
    MRIA_RUN.impact_data(DATA, disr_dict_sup, disr_dict_fd)

    """Get direct losses """
    disrupt = pd.DataFrame.from_dict(disr_dict_sup, orient='index')
    disrupt.reset_index(inplace=True)
    disrupt[['region', 'sector']] = disrupt['index'].apply(pd.Series)
    disrupt.drop('index', axis=1, inplace=True)
    disrupt = 1 - disrupt.groupby(['region', 'sector']).sum()
    disrupt.columns = ['shock']

    output['dir_losses'] = (disrupt['shock']*output['x_in']).fillna(0)*-1

    if print_output:
        output_print = 'print'
    else:
        output_print = None

    status = MRIA_RUN.run_impactmodel(output=output_print,solver=solver,tol=1e-6, DisWeight=1.75, RatWeight=2.0)
    print(status)

    output['x_out'] = pd.Series(MRIA_RUN.X.get_values())
    output['total_losses'] = (output['x_out'] - output['x_in'])
    output['ind_losses'] = (output['total_losses'] - output['dir_losses'])
    output = output.sort_index(level=[0,1])
    output.to_csv(os.path.join(output_dir, '{}.csv'.format(event)))

    prov_impact = output.groupby(level=0, axis=0).sum()[['dir_losses','total_losses','ind_losses']]
    collect_outputs[solver] = prov_impact

    total_losses_sum = (output['total_losses'].sum().sum())/1000
    total_losses[solver] = total_losses_sum

    print('{} results in {} billion pesos losses for {}'.format(event,total_losses_sum,solver))

    return output

if __name__ == "__main__":
    run('INDEC',2015)