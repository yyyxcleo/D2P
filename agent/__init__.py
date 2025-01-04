from .er import *
from .scr import *
from .mosepro_distill import *
from .joint import *
from .buf import *
# from .mose import *
from .mosepro_mkd import *

METHODS = {
    'er': ER,
    'scr': SCR,
    'mose': MOSE,
    'd2p': D2P,
    'joint': Joint,
    'buf': Buf,
}


def get_agent(method_name, *args, **kwargs):
    if method_name in METHODS.keys():
        return METHODS[method_name](*args, **kwargs)
    else:
        raise Exception('unknown method!')