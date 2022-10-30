from itertools import cycle
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, filename='./my_logs.log')

import tensorflow_datasets as tfds
from tqdm import tqdm

import s3fs
import time
import sys
import tensorflow as tf
import tensorflow_io as tfio

#Importing stuff outside the dir, absolute path is hardcoded for simplicity
sys.path.append('/home/awesome/scripts/BDD_Driving_Model/')
from data_providers.nexar_large_speed import MyDataset
from batching import FLAGS as BFLAGS

BFLAGS.data_dir = '/home/awesome/data/sample_tfrecs'

s3 = s3fs.S3FileSystem(anon=False)
path = ['s3://'+_ for _ in s3.glob('s3://s-laion/ssd-videos/*.tfrecord')]
#path = glob("/home/awesome/data/sample_tfrecs/train/*")

logging.info(path)
assert len(path) > 0, 'No TFRecords found'

features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Tensor(
            shape=(963, 256, 256, 6), dtype=tf.uint8),
    'label':
        tfds.features.Tensor(
            shape=(963, 51), dtype=tf.float32
        ),

})

tf.compat.v1.enable_eager_execution()
data_obj = MyDataset('train')
 
def gaussian_1d(index, sigma, length):
    x = tf.range(length, dtype=tf.int32)
    index = tf.cast(index, tf.int32)
    return tf.reshape(tf.exp(-((x-index)**2)/(2*sigma**2)), (51,))

def generate_onehot(source_tensor):
    source_tensor = tf.reshape(tf.one_hot(int(source_tensor), 51, dtype='float32'), (51,))

    idx = tf.where(source_tensor == 1.0) #getting index of onehot encoded vector
    gaussian_kernel = gaussian_1d(idx, 2, 51) #generating gaussian kernel
    source_tensor = tf.repeat(1.0, len(source_tensor)) #making same-sized tensor filled with 1s
    return tf.math.multiply(source_tensor, tf.cast(gaussian_kernel, tf.float32)) #hadamard product

def binner(inp):
    '''
    inp: list of tuples/iterables pairs 
    bins values in each tuple into bins of size 0.5, in the range: [0, 25]
    '''
    #create lookup dict with keys 0, 0.5, 1, ... 25 for values 0,1,2,...,51
    lookup = dict( zip([_/2 for _ in range(51)], [k for k in range(51)]) )
    
    #ensuring array is postiive
    #round off an integer to the closest 0.5
    return tf.vectorized_map(lambda x: tf.clip_by_value(tf.cast(tf.round(tf.abs(x)/0.5)*0.5, tf.float32), 0, 25), inp)


def parser(example):
    inp, out = data_obj.parse_example_proto(example)
    return input_fn(inp+out)

def splicer(inp, n):
    '''choses adjacent image pairs temporally 'n' frames apart
    inp: The tensor to splice
    n: The number of frames for each pair to be apart
    '''
    return tf.stack( (inp[:len(inp)-(n+1)] , inp[(n+1):]), axis=-1)  #convert to Tf Tensor later

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

def input_fn(*sample):
    sample = sample[0]
    start = time.time()
    # unpack a tensor of shape 963,2,2 into two variables a and b both of shape 963,2 abd 963,2
    old_ctrl, spliced_ctrl = tf.unstack(splicer(tf.reshape(sample[1], [-1, 2]), n=60), axis=1)
    spliced_images = splicer(tf.reshape(sample[0], [-1, 512, 512, 3]), n=60) #~800ms for splicing both vars

    #concatenate a list of tuples of tensors channelwise
    spliced_images = tf.reshape(spliced_images, [-1, 512, 512, 6])
    spliced_images = tf.cast(tf.image.resize(spliced_images, [256, 256], preserve_aspect_ratio=True), tf.uint8)

    spliced_ctrl = binner(spliced_ctrl)[:, :1]

    spliced_ctrl = tf.map_fn(generate_onehot, spliced_ctrl) #converting to onehot gaussian=smoothed
    spliced_ctrl = tf.reshape(spliced_ctrl, [-1, 51])

    #print(f'shape of spliced_ctrl: {spliced_ctrl.shape}\nshape of spliced_images: {spliced_images.shape}')
    return spliced_images, spliced_ctrl

def SUMO_dataset(num_calls):
    # convert a list named path to tf.data.Dataset
    #paths = ['s3://'+_ for _ in s3.glob('s3://s-laion/ssd-videos/new_tfrecs/*.tfrecord')]
    #raw_dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=num_calls)

    raw_dataset = tfds.builder_from_directory('s3://s-laion/ssd-videos/new_tfrecs/').as_dataset(
        split='train', shuffle_files=True, as_supervised=True,
        read_config=tfds.ReadConfig(skip_prefetch=True, num_parallel_calls_for_decode=num_calls, input_context=tf.distribute.InputContext(num_input_pipelines=64, input_pipeline_id=1))
    )
    
    return raw_dataset

def num_examples():
    #old = 1.85s/it --> 55
    data = SUMO_dataset(100).prefetch(64)

    for _ in tqdm(data, total=51480):
        pass

if __name__ == '__main__':
    num_examples()