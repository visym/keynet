GLOBAL = {'PROCESSES': 1, 'VERBOSE': True}

def num_processes(n=None):
    if n is not None:
        GLOBAL['PROCESSES'] = n
    return GLOBAL['PROCESSES']

def verbose(b=None):
    if b is not None:
        GLOBAL['VERBOSE'] = b
    return GLOBAL['VERBOSE']

