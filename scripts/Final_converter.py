#!/usr/bin/env python
# coding: utf-8
'''
Working with datasets to process sample tfrecs and setup stuff for training

===========================================================
Use gausian distribution to push model to regress longitudinal speed accurately
explore scaling via simplest ways - TFRecords, converted to TFDS datasets and speed extracted
And find that paper to explore recurrence

i[1] ==> column 1: latitudinal control (angular velocity); column 2: longitudinal control (speed)

--> Visualize and clean up worspace
'''

import gc
import ray
import warnings
warnings.filterwarnings("ignore")
#import time
import psutil
import logging
logging.basicConfig(level=logging.INFO, filename='./my_logs.log')

import tensorflow_datasets as tfds
import os
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import s3fs
import time
import sys
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from tqdm import tqdm
#from glob import glob

#Importing stuff outside the dir, absolute path is hardcoded for simplicity
sys.path.append('/home/awesome/scripts/BDD_Driving_Model/')
from data_providers.nexar_large_speed import FLAGS, MyDataset
from batching import FLAGS as BFLAGS


BFLAGS.data_dir = '/home/awesome/data/sample_tfrecs'

s3 = s3fs.S3FileSystem(anon=False)
path = ['s3://'+_ for _ in s3.glob('s3://s-laion/ssd-videos/*.tfrecord')]
#path = glob("/home/awesome/data/sample_tfrecs/train/*")

logging.info(path)
assert len(path) > 0, 'No TFRecords found'

'''
Maybe requires covnversion to float_list/bytest_list with .value ??
Definitely the only way to properly produce trainiable data
'''
tf.compat.v1.enable_eager_execution()

features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Tensor(
            shape=(963, 256, 256, 6), dtype=tf.uint8),
    'label':
        tfds.features.Tensor(
            shape=(963, 51), dtype=tf.float32
        ),

})

data_obj = MyDataset('train')
ray.init(_temp_dir='/fsx/awesome/temp')
 
def gaussian_1d(index, sigma, length):
    x = np.arange(length)
    return np.exp(-((x-index)**2)/(2*sigma**2))

def generate_onehot(source_tensor):
    source_tensor = tf.keras.utils.to_categorical(source_tensor, 51, dtype='float32')
    idx = tf.where(source_tensor == 1.0) #getting index of onehot encoded vector
    gaussian_kernel = gaussian_1d(idx, 2, len(source_tensor))
    source_tensor = tf.repeat(1.0, len(source_tensor)) #making same-sized tensor filled with 1s
    return tf.math.multiply(source_tensor, gaussian_kernel) #hadamard product

def binner(inp):
    '''
    inp: list of tuples/iterables pairs 
    bins values in each tuple into bins of size 0.5, in the range: [0, 25]
    '''
    #create lookup dict with keys 0, 0.5, 1, ... 25 for values 0,1,2,...,51
    lookup = dict( zip([_/2 for _ in range(51)], [k for k in range(51)]) )
    
    #ensuring array is postiive
    #round off an integer to the closest 0.5
    inp = np.clip(np.round(np.abs(inp)/0.5)*0.5, 0, 25)
    return np.vectorize(lookup.get)(inp).astype(np.float32)


def parser(example):
    inp, out = data_obj.parse_example_proto(example)
    return inp+out

def splicer(inp, n):
    '''choses adjacent image pairs temporally 'n' frames apart
    inp: The tensor to splice
    n: The number of frames for each pair to be apart
    '''
    return tf.convert_to_tensor(list(zip( inp[:len(inp)-(n+1)] , inp[(n+1):]))) #convert to Tf Tensor later

def error_handler(gen, thrd):
    '''
    Handles errors when iterating through an iterable
    '''
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except Exception as e:
            logging.info(f'\n--------\n{thrd} Errored out - Batch has been lost. {e}\n-------------')
            continue
'''
52488it
'''
@ray.remote
def input_fn(sample):
    start = time.time()
    old_ctrl, spliced_ctrl = zip(*splicer(tf.reshape(sample[1], [-1, 2]), n=60))
    spliced_images = splicer(tf.reshape(sample[0], [-1, 512, 512, 3]), n=60) #~800ms for splicing both vars

    #concatenate a list of tuples of tensors channelwise
    spliced_images = tf.convert_to_tensor(list(map(lambda x: tf.concat([x[0], x[1]], axis=2), spliced_images)))
    spliced_images = tf.cast(tf.image.resize(spliced_images, [256, 256], preserve_aspect_ratio=True), tf.uint8)

    spliced_ctrl = tf.map_fn(generate_onehot, tf.reshape(binner(spliced_ctrl)[:, :1], -1) ) #converting to onehot gaussian=smoothed
    spliced_ctrl = tf.reshape(spliced_ctrl, [-1, 51])
    
    # Write the `tf.train.Example` observations to the file
    #ser_img = io.BytesIO(resized_images) #serialized image
    #COMPRESSION DISABLED!
    with tf.io.TFRecordWriter(f's3://s-laion/ssd-videos/new_tfrecs/{np.random.randint(0, 9999999999999)}.tfrecord') as writer:
        example = features.serialize_example({'image': spliced_images.numpy(), 'label': spliced_ctrl.numpy()}) #im = image, ctrl = longitudinal control/speed
        writer.write(example)
    
    del spliced_images, spliced_ctrl, example, sample, old_ctrl
    gc.collect()

    logging.info(f'written! - taken {time.time()-start} seconds\ntotal memory usage: {psutil.Process(os.getpid()).memory_info().rss/1000000} MB')
    print('written!')

raw_dataset = tf.data.TFRecordDataset(
                        path, num_parallel_reads=100,
                    ).map(
                        parser, num_parallel_calls=100, deterministic=False,
                    ).prefetch(tf.data.experimental.AUTOTUNE) #increasing calls for more parallelism

def parmap(fn, itereble):
    return ray.get([fn.remote(x) for x in tqdm(itereble, total=51480)])

if __name__ == '__main__':
    parmap(input_fn, raw_dataset)

logging.info(f'\n-------------\nFinished!\n----------------')
print('All Done!')
