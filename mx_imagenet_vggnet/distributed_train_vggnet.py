# import the necessary packages
from config import imagenet_vggnet_config as config
from pis.nn.mxconv.mxvggnet import MxVGGNet
import horovod.mxnet as hvd
import mxnet as mx
import argparse
import logging
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
        help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
        help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
        help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
        filename="training_{}.log".format(args["start_epoch"]),
        filemode="w")

# Horovod: initialize Horovod
hvd.init()
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()

num_classes = 1000
num_training_samples = 1281167
batch_size = CONFIG.BATCH_SIZE
# number of batches in an epoch for a worker
epoch_size = \
    int(math.ceil(int(num_training_samples // num_workers) / batch_size))

lr_sched = mx.lr_scheduler.PolyScheduler(
        80 * epoch_size,
        base_lr=(1e-2 * num_workers),
        pwr=2,
        warmup_steps=(10 * epoch_size),
        warmup_begin_lr=0
)

# Horovod: pin a GPU to be used to local rank
context = mx.gpu(local_rank)

# load the RGB means for the training set, then determine the batch
# size
means = json.loads(open(config.DATASET_MEAN).read())
#batchSize = config.BATCH_SIZE * config.NUM_DEVICES
batchSize = batch_size

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
        path_imgrec=config.TRAIN_MX_REC,
        data_shape=(3, 224, 224),
        batch_size=batchSize,
        rand_crop=True,
        rand_mirror=True,
        rotate=15,
        max_shear_ratio=0.1,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"],
        preprocess_threads=config.NUM_DEVICES * 2)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
        path_imgrec=config.VAL_MX_REC,
        data_shape=(3, 224, 224),
        batch_size=batchSize,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"])


# construct the checkpoints path, initialize the model argument and
# auxiliary parameters
checkpointsPath = os.path.sep.join([args["checkpoints"],
        args["prefix"]])
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
        # build the LeNet architecture
        print("[INFO] building network...")
        model = MxVGGNet.build(config.NUM_CLASSES)

# otherwise, a specific checkpoint was supplied
else:
        # load the checkpoint from disk
        print("[INFO] loading epoch {}...".format(args["start_epoch"]))
        model = mx.model.FeedForward.load(checkpointsPath,
                args["start_epoch"])

        # update the model and parameters
        argParams = model.arg_params
        auxParams = model.aux_params
        model = model.symbol

# Create model
model = mx.mod.Module(context=context, symbol=model)
model.bind(data_shapes=trainIter.provide_data,
        label_shapes=trainIter.provide_label)
model.init_params(arg_params=argParams, aux_params=auxParams)

# Horovod: fetch and broadcast parameters
(arg_params, aux_params) = model.get_params()
if arg_params is not None:
        hvd.broadcast_parameters(arg_params, root_rank=0)
if aux_params is not None:
        hvd.broadcast_parameters(aux_params, root_rank=0)
model.set_params(arg_params=arg_params, aux_params=aux_params)

# Create the optimizer
optimizer_params = {'wd': 0.005,
                        'momentum': 0.9,
                        'rescale_grad': 1.0 / batch_size,
                        'lr_scheduler': lr_sched}

opt = mx.optimizer.create('sgd', **optimizer_params)

# Horovod: wrap optimizer with DistributedOptimizer
opt = hvd.DistributedOptimizer(opt)

# Setup callback during training
batch_callback = None
if rank == 0:
        batchEndCBs = mx.callback.Speedometer(batch_size * num_workers,
                                                 250)

epochEndCBs = mx.callback.do_checkpoint(checkpointsPath)


# train the network
print("[INFO] training network...")
model.fit(
        X=trainIter,
        eval_data=valIter,
        num_epoch=80,
        kvstore=None,
        eval_metric=metrics,
        batch_end_callback=batchEndCBs,
        epoch_end_callback=epochEndCBs,
        optimizer=opt)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
res = model.score(valIter, [acc_top1, acc_top5])
for name, val in res:
        logging.info('Epoch[%d] Rank[%d] Validation-%s=%f',
                         80 - 1, rank, name, val)
