import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import argparse
import wandb

from keras_cv_attention_models.convnext.convnext import *
from wandb.keras import WandbCallback
from tensorflow.keras import mixed_precision
from tqdm import tqdm

# ======================================================================================================================
# Initialization
# ======================================================================================================================
gpus = tf.config.list_physical_devices('GPU')

#TODO:Limiting memory growth in GPUs
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  pass

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
print(policy)

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='Convnext_baseline', help='model')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--pretrained', type=str, default='imagenet', help='pretrained model path')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')

parser.add_argument("--lr_base_512", type=float, default=3e-4, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
parser.add_argument('--optimizer', type=str, default='LAMB', help='optimizer')
parser.add_argument('--log_frequency', type=int, default=15, help='log frequency')
parser.add_argument('--val_frequency', type=int, default=1, help='eval frequency')


parser.add_argument('--input_shape', type=int, default=[256, 256], help='input shape')
parser.add_argument('--rescale_mode', type=str, default="tf", help='rescale mode')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

# print all the arguments
args = parser.parse_args()
print(args)

#==================================================
# HELPER FUNCTIONS
#==================================================
wandb.init(project="SUMO", entity="neel", name=args.model_name, config=args, magic=True)

def init_mean_std_by_rescale_mode(rescale_mode):
    if isinstance(rescale_mode, (list, tuple)):  # Specific mean and std
        mean, std = rescale_mode
    elif rescale_mode == "torch":
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = tf.constant([0.229, 0.224, 0.225]) * 255.0
    elif rescale_mode == "tf":  # [0, 255] -> [-1, 1]
        mean, std = 128, 128
        # mean, std = 127.5, 128.0
    elif rescale_mode == "tf128":  # [0, 255] -> [-1, 1]
        mean, std = 128.0, 128.0
    elif rescale_mode == "raw01":
        mean, std = 0, 255.0  # [0, 255] -> [0, 1]
    else:
        mean, std = 0, 1  # raw inputs [0, 255]
    return mean, std

def init_optimizer(optimizer, lr_base, weight_decay, momentum=0.9):
    optimizer = str(optimizer).lower()
    # norm_weights = ["bn/gamma", "bn/beta", "ln/gamma", "ln/beta", "/positional_embedding", "/bias"]  # ["bn/moving_mean", "bn/moving_variance"] not in weights
    no_weight_decay = ["/gamma", "/beta", "/bias", "/positional_embedding", "/no_weight_decay"]  # ["bn/moving_mean", "bn/moving_variance"] not in weights
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_base, momentum=momentum)
    elif optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_base, momentum=momentum)
    elif optimizer == "lamb":
        optimizer = tfa.optimizers.LAMB(learning_rate=lr_base, weight_decay_rate=weight_decay, exclude_from_weight_decay=no_weight_decay, global_clipnorm=1.0)
    elif optimizer == "adamw":
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_base, weight_decay=lr_base * weight_decay, global_clipnorm=1.0)
        if hasattr(optimizer, "exclude_from_weight_decay"):
            setattr(optimizer, "exclude_from_weight_decay", no_weight_decay)
    elif optimizer == "sgdw":
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_base, momentum=momentum, weight_decay=lr_base * weight_decay)
        if hasattr(optimizer, "exclude_from_weight_decay"):
            setattr(optimizer, "exclude_from_weight_decay", no_weight_decay)
    else:
        optimizer = getattr(tf.keras.optimizers, optimizer.capitalize())(learning_rate=lr_base)
    return optimizer

#==================================================
# DATASET SETUP
#==================================================
input_context = tf.distribute.InputContext(
    input_pipeline_id=1,  # Worker id
    num_input_pipelines=16,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)

dataset = tfds.load(name='dataset_bdd100k', data_dir='s3://s-laion/ssd-videos/', as_supervised=True, read_config=read_config)

mean, std = init_mean_std_by_rescale_mode('tf')
post_batch = lambda xx, yy: ((xx - mean) / std, yy)

args.batch_size *= 8 #8 GPUs

train_ds = dataset['train'].unbatch().map(post_batch, num_parallel_calls=100).batch(args.batch_size).prefetch(24)
val_ds = dataset['test'].unbatch().map(post_batch, num_parallel_calls=100).batch(args.batch_size)

print(f"\nTraining Dataset: {train_ds.element_spec}\n")

#==============================
# MODEL SETUP
#==============================
WBCallback = WandbCallback(save_model=True, generator=val_ds.as_numpy_iterator(), log_evaluation_frequency=1,
    input_type='images', log_evaluation=True, log_batch_frequency=15, validation_steps=50)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./checkpoints/', save_best_only=True, monitor="val_acc")

Terminate_on_NaN = tf.keras.callbacks.TerminateOnNaN()

callbacks = [WBCallback, checkpoint_callback, Terminate_on_NaN]

steps_per_epoch = 11249952 // (args.batch_size*8)

lr = (args.lr_base_512 * 512 / args.batch_size) / 8

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    optimizer = init_optimizer(args.optimizer, lr, args.weight_decay)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, initial_scale=2**15 , dynamic=True)
    # Accuracy metric
    metrics = tf.keras.metrics.CategoricalAccuracy()
    model = ConvNeXtSmall(input_shape=(*args.input_shape, 6), num_classes=51, pretrained=args.pretrained)

# Distributed dataset
dist_dataset = strategy.experimental_distribute_dataset(train_ds)
# loss objects
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

#==============================
#Compiling Model and running training
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compute_loss(labels, predictions):
  per_example_loss = loss(labels, predictions)
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = compute_loss(labels, logits)
        #loss_value = optimizer.get_scaled_loss(loss_value)

    # Backward pass
    grads = tape.gradient(loss_value, model.trainable_variables)
    #grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    #Update metrics
    metrics.update_state(labels, logits)

    return loss_value

@tf.function
def distributed_train_step(x,y):
    per_replica_losses = strategy.run(train_step, args=(x,y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for epoch in range(args.epochs):

    for step, (images, labels) in tqdm(enumerate(dist_dataset)):
        loss_value = distributed_train_step(images, labels)
        if step % args.log_frequency == 0:
            print(f"Epoch {epoch}, Step {step}/{steps_per_epoch}, Loss {loss_value}, Acc {metrics.result()}")
            wandb.log({'loss': loss_value, 'train_acc':metrics.result() ,'step': step, 'epoch': epoch})
    
    print(f"\n{'='*50}Epoch Ended {epoch}, Train_acc: {metrics.result()}\n{'='*50}\n")
    wandb.log({'End of Epoch train_acc':metrics.result()})

    if epoch % args.val_frequency == 0:
        output = model.evaluate(val_ds, callbacks=callbacks, use_multiprocessing=True, return_dict=True)
        wandb.log(output) #logging the validation metrics
        model.save_weights(f"./checkpoints/{epoch}.h5")
        print(f"Saved checkpoint {epoch}")

'''# Mirrored strategy
with strategy.scope():
    loss, metrics = tf.keras.losses.CategoricalCrossentropy(from_logits=True), ['acc']
    optimizer = init_optimizer(args.optimizer, lr, args.weight_decay)

    model = ConvNeXtSmall(input_shape=(*args.input_shape, 6), num_classes=51, pretrained=args.pretrained)

    print(model.summary())


model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

hist = model.fit(
    train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, 
    verbose=1, use_multiprocessing=True, workers=4)'''