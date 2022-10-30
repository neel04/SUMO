import logging
logging.basicConfig(level=logging.INFO, filename='./renamer_logs.log')
import ray
from ray.util.multiprocessing import Pool
import random
from tqdm import tqdm
import tensorflow_io as tfio
import os
import s3fs

#prints aws logs for every request
os.environ['AWS_TRACE'] = "0"

#rename each file in the folder
s3 = s3fs.S3FileSystem(anon=False)
paths = ['s3://'+_ for _ in s3.glob('s3://s-laion/ssd-videos/new_tfrecs/*tfrecord*')]
random.shuffle(paths)

assert len(paths) > 0, 'No TFRecords found'

pool = Pool()

def name_resetter(path):
    if len(path) > 60:
        s3.mv(path, f's3://s-laion/ssd-videos/new_tfrecs/{random.randint(0, 999999999)}.tfrecord')
    print('Resetted:', path)

def file_renamer(x):
    '''
    Rename path to TFDS format:
    sample input: s3://s-laion/ssd-videos/new_tfrecs/432154363.tfrecord
    sample output: s3://s-laion/ssd-videos/new_tfrecs/bdd100k-train.tfrecord-{str(idx).zfill(5)}-of-{len(path)}
    '''
    idx, path = x
    test_split_cutoff = len(paths) - (len(paths) // 10) #10% of the data is used for validation

    if idx > test_split_cutoff:
        idx -= test_split_cutoff # fix testing indexing
        s3.mv(path, f's3://s-laion/ssd-videos/new_tfrecs/bdd100k-test.tfrecord-{str(idx).zfill(5)}-of-{str(len(paths)//10).zfill(5)}')
        logging.info(f'{path} moved to bdd100k-test.tfrecord-{str(idx).zfill(5)}-of-{str(len(paths)//10).zfill(5)}')
    else:
        s3.mv(path, f's3://s-laion/ssd-videos/new_tfrecs/bdd100k-train.tfrecord-{str(idx).zfill(5)}-of-{test_split_cutoff}')
        logging.info(f'{path} moved to bdd100k-train.tfrecord-{str(idx).zfill(5)}-of-{str(len(paths)//10).zfill(5)}')
    
    print('File renamed!')
    
#parallelize name resetter which loops over paths with ray
def my_func_par(f, large_list):
    pool.map(f, large_list) #~1-2hr runtime

#my_func_par(name_resetter, tqdm(paths, total=len(paths), desc='Resetting files'))
paths = ['s3://'+_ for _ in s3.glob('s3://s-laion/ssd-videos/new_tfrecs/*tfrecord*')]
my_func_par(file_renamer, tqdm(enumerate(paths), total=len(paths), desc='Renaming files'))