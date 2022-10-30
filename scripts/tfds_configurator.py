import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import os
#prints aws logs for every request
os.environ['AWS_TRACE'] = "1"

'''
singularity shell --nv --cleanenv -B /fsx/awesome/temp:/fsx/awesome/temp tensorflow_latest.sif
clear; python3 keras_cv_attention_models/train_script.py -m coatnet.CoAtNet0 --seed 0 --batch_size 128 -s CoAtNet0_160 -d 'bdd100k:1.0.0' --rescale_mode='tf' --num_layers=0 --mixup_alpha=0 --cutmix_alpha=0
'''
print(tfio.__version__)
features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Tensor(
            shape=(963, 256, 256, 6), dtype=tf.uint8), #tf.uint8
    'label':
        tfds.features.Tensor(
            shape=(963, 51), dtype=tf.float32),
})

tfds.folder_dataset.write_metadata(
    data_dir='s3://s-laion/ssd-videos/new_tfrecs/',
    features=features,
    # You can also explicitly pass a list of `tfds.core.SplitInfo`
    split_infos=[
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=[963]*47736,
            num_bytes=0),
        
        tfds.core.SplitInfo(
            name='test',
            shard_lengths=[963]*5303,
            num_bytes=0),
    ],
    description="""Preprocessed TFRecords of BDD100K dataset, longitudinal data only, delayed by 60 frames for each tuple.""",
    supervised_keys=('image', 'label'),
    check_data=True,
)
