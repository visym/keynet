from vipy.util import try_import
import tempfile

GLOBAL = {'PROCESSES': 1, 'VERBOSE': True, 'DASK_CLIENT': None}

def backend():
    return 'scipy'

def num_processes(n=None, backend='joblib'):
    if n is not None:
        GLOBAL['PROCESSES'] = n
        if n > 1 and backend == 'dask':
            try_import('dask', 'dask distributed')
            from dask.distributed import Client
            client = Client(name='keynet',
                            scheduler_port=0,
                            dashboard_address=None,
                            processes=True,
                            threads_per_worker=1,
                            n_workers=n,
                            direct_to_workers=True,
                            local_directory=tempfile.mkdtemp())
            GLOBAL['DASK_CLIENT'] = client

    return GLOBAL['PROCESSES']

def dask_client():
    assert GLOBAL['DASK_CLIENT'] is not None, "Must set keynet.globals.num_processes(n>1)"
    return GLOBAL['DASK_CLIENT'] 

def verbose(b=None):
    if b is not None:
        GLOBAL['VERBOSE'] = b
    return GLOBAL['VERBOSE']

