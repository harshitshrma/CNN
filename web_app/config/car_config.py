from os import path

BASE_PATH = "/home/ubuntu/datasets/cars"

IMAGES_PATH = path.sep.join([BASE_PATH, "car_ims"])
LABELS_PATH = path.sep.join([BASE_PATH, "complete_dataset.csv"])

MX_OUTPUT = BASE_PATH
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

LABEL_ENCODER_PATH = path.sep.join([BASE_PATH, "output/le.cpickle"])

R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

BATCH_SIZE = 32
NUM_DEVICES = 1
