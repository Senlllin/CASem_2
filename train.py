import argparse
import provider
import numpy as np
import datetime
import os
import config
import tensorflow as tf
from model_Att import DGCNN, DGCNNLoss
# from pointnet_model import PointNet, PointNetLoss

# Parameter Set
parser = argparse.ArgumentParser()
parser.add_argument('--test_area', type=str, default=config.TEST_AREA, help='The ID of test area.')
parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_steps', type=int, default=config.DECAY_STEPS,
                    help='Learning rate decay when it going around steps')
parser.add_argument('--decay_rate', type=float, default=config.DECAY_RATE, help='Learning_rate *= decay_rate')
parser.add_argument('--log_dir', type=str, default=config.LOG, help='Log dir [default: log]')
parser.add_argument('--epoch', type=int, default=config.EPOCH, help='The counts of training epoches')
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                    help='Batch Size during training for each GPU [default: 24]')
parser.add_argument('--test_frequency', type=int, default=config.TEST_FREQUENCY, help='Test frequency')
parser.add_argument('--save_frequency', type=int, default=config.SAVE_FREQUENCY, help='Save frequency')
parser.add_argument('--num_classes', type=int, default=config.NUM_CLASSES, help='number of classes')
FLAGS = parser.parse_args()

TEST_AREA = FLAGS.test_area
LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
EPOCH = FLAGS.epoch
BATCH_SIZE = FLAGS.batch_size
DECAY_STEPS = FLAGS.decay_steps
DECAY_RATE = FLAGS.decay_rate
TEST_FREQUENCY = FLAGS.test_frequency
SAVE_FREQUENCY = FLAGS.save_frequency

# Load Data
DATA_PATH = './data'
data_batch_list = []
label_batch_list = []
ALL_FILES = provider.getDataFiles(DATA_PATH + '/all_files.txt')
ROOM_LIST = provider.getDataFiles(DATA_PATH + '/rooms_name.txt')

for _ in ALL_FILES:
    data_batch, label_batch = provider.loadH5Files(DATA_PATH + '/' + _.split('/')[-1])
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)

test_area = TEST_AREA
print('Test Area: ' + test_area)
train_idx = []
test_idx = []
for room_id, room_name in enumerate(ROOM_LIST):
    if test_area in room_name:
        test_idx.append(room_id)
    else:
        train_idx.append(room_id)

train_data = data_batches[train_idx, ...].astype(np.float32)
train_label = label_batches[train_idx].astype(np.int32)
test_data = data_batches[test_idx, ...].astype(np.float32)
test_label = label_batches[test_idx].astype(np.int32)
TRAIN_BATCHES = train_data.shape[0]
print('Train Batches: {} Test Batches:{}'.format(TRAIN_BATCHES, test_data.shape[0]))
print('Train Points : {} Test Points:{}'.format(TRAIN_BATCHES * train_data.shape[1],
                                                test_data.shape[0] * test_data.shape[1]))
train_weights = np.zeros((config.NUM_CLASSES,),dtype=np.float32)
for l in train_label:
    for i in range(config.NUM_CLASSES):
        mask_array = np.full_like(l, i, dtype=np.int32)
        train_weights[i] += np.sum(np.equal(mask_array,l))
all_points_count = np.sum(train_weights)
for _ in range(len(train_weights)):
    train_weights[_] /= all_points_count
    train_weights[_] = 1./np.log(1.2+train_weights[_])
# train_weights /= np.max(train_weights)


def rotate_point_cloud_z(x):
    batch_data = x.numpy()
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0.],
                                    [-sinval, cosval, 0.],
                                    [0., 0., 1.]], dtype=np.float32)
        batch_data[k, :, :3] = np.matmul(batch_data[k, :, :3], rotation_matrix)
        batch_data[k, :, 3:6] = np.matmul(batch_data[k, :, 3:6], rotation_matrix)

    return batch_data


# Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(20 * BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
del train_data, train_label, test_data, test_label
# Model
dgcnn_net = DGCNN(k=20, num_classes=12)
# dgcnn_net = PointNet(num_point=8192,num_attribute=9,num_classes=12)

# Loss
dgcnn_loss = DGCNNLoss()
# dgcnn_loss = PointNetLoss()

# Optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True
)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)


# @tf.function
def trainOneBatch(batch_data, batch_label, sample_weight=None):
    batch_data = rotate_point_cloud_z(batch_data)
    with tf.GradientTape() as tape:
        outputs = dgcnn_net(batch_data, training=True)
        loss_value = tf.reduce_sum(dgcnn_loss(y_true=batch_label, y_pred=outputs,sample_weight=sample_weight))
    gradients = tape.gradient(loss_value, dgcnn_net.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, dgcnn_net.trainable_variables))

    correct = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(outputs, 2, output_type=tf.int32), batch_label), tf.float32)
    )
    return loss_value, correct


# @tf.function
def testOneBatch(batch_data, batch_label):
    if len(batch_data.shape) < 3:
        batch_data = tf.expand_dims(batch_data, 0)
    if len(batch_label.shape) < 2:
        batch_label = tf.expand_dims(batch_label, 0)
    outputs = dgcnn_net(batch_data, training=False)
    loss_value = tf.reduce_sum(dgcnn_loss(y_true=batch_label, y_pred=outputs))
    correct = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(outputs, 2, output_type=tf.int32), batch_label), tf.float32)
    )
    return loss_value, correct


# LOGS Tracing
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, current_time))

# Training
GLOBAL_STEP = 1
BEST_TEST_ACC_LIST = [-1 * np.inf]
for epoch in range(1, EPOCH + 1):
    train_epoch_loss = 0.
    train_correct = 0
    train_count = 0
    step = 1

    for data, label in train_dataset:
        sw = train_weights[label.numpy()]
        train_batch_loss, train_batch_correct = trainOneBatch(data, label, sw)
        train_batch_count = tf.cast(label.shape[0] * label.shape[1], tf.float32)

        if train_batch_loss < 0:
            print("Skip this step due to negative loss.")
            continue
        train_epoch_loss += train_batch_loss
        train_correct += train_batch_correct
        train_count += train_batch_count

        with summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', train_batch_loss, step=GLOBAL_STEP)
        print(
            "Epoch: {}/{}, Step: {}/{},"
            "batch_loss: {:.5f}, global_steps:{}, lr:{:.9f}.".format(
                epoch,
                EPOCH,
                step,
                TRAIN_BATCHES // BATCH_SIZE + 1,
                train_batch_loss,
                GLOBAL_STEP,
                lr_schedule.__call__(GLOBAL_STEP)
            ))
        GLOBAL_STEP += 1
        step += 1
    train_epoch_loss /= step
    train_epoch_acc = train_correct / train_count
    with summary_writer.as_default():
        tf.summary.scalar('train_epoch_loss', train_epoch_loss, step=epoch)
        tf.summary.scalar('train_epoch_accuracy', train_epoch_acc, step=epoch)
    print("Epoch: {}/{},Train Average Loss:{:.5f} Accuracy:{:.5f}.".format(
        epoch, EPOCH, train_epoch_loss, train_epoch_acc)
    )

    # Testing
    if epoch % TEST_FREQUENCY == 0 and test_area != 'None':
        test_loss = 0.
        test_correct = 0
        test_count = 0
        step = 1

        for data, label in test_dataset:
            test_loss_value, test_batch_correct = testOneBatch(data, label)
            test_batch_count = tf.cast(label.shape[-1], tf.float32)
            if test_loss_value < 0:
                print("Skip this step due to negative loss.")
                continue
            test_loss += test_loss_value
            test_correct += test_batch_correct
            test_count += test_batch_count
            step += 1
        test_loss /= step
        test_acc = test_correct / test_count
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss, step=epoch)
            tf.summary.scalar('test_accuracy', test_acc, step=epoch)
        print("Test_loss: {:.5f} Accuracy: {:.5f}".format(test_loss, test_acc))

        if test_acc > BEST_TEST_ACC_LIST[-1]:
            dgcnn_net.save_weights(os.path.join(LOG_DIR, 'dgcnn'), save_format='tf')
            print(
                "The best accuracy on test dataset has declined from {} to {}, and saving model weight to {}."
                    .format(BEST_TEST_ACC_LIST[-1], test_acc, os.path.join(LOG_DIR, 'dgcnn_arg'))
            )
            BEST_TEST_ACC_LIST.append(test_acc)

    if epoch % SAVE_FREQUENCY == 0:
        dgcnn_net.save_weights(filepath=os.path.join(LOG_DIR, 'dgcnn' + "epoch-{}".format(epoch)), save_format='tf')
        print("Save Model in {}.".format(os.path.join(LOG_DIR, 'dgcnn' + "epoch-{}".format(epoch))) + "epoch-{}".format(
            epoch))

with open('record.txt', 'a') as fo:
    fo.write('\nTest Area: ' + test_area + '\n')
    fo.write('Overall Accuracy: ' + str(max(BEST_TEST_ACC_LIST)))
