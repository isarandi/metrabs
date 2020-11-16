import os

DATA_ROOT = os.environ.get('DATA_ROOT', default='/home/sarandi/data_reprod')
CACHE_DIR = os.environ.get('CACHE_DIR', default=f'{DATA_ROOT}/cache')

