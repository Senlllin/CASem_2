import tensorflow as tf
from model import DGCNN
# from pointnet_model import  PointNet
import provider
# from ldgcnn_model import LDGCNN
import numpy as np
import warnings
import h5py
import config
from model_Att import DGCNN


def constructeModel():
    # return PointNet(num_point=8192,num_attribute=9,num_classes=config.NUM_CLASSES)
    # return LDGCNN(30,num_classes=config.NUM_CLASSES)
    return DGCNN(k=20, num_classes=config.NUM_CLASSES)


def inference_single_room(test_data, test_label,real_data,real_label, file_path,name):
    """
    :param test_data: numpy array B×N×6
    :param test_label: numpy array
    :param file_path: model weight path
    :param name: string, room name
    :return: miou, oa and iou_list
    """
    # inference
    net = constructeModel()
    net.load_weights(file_path)
    pred_tensor = []

    for data in test_data:
        data = tf.expand_dims(data, 0)  # 1×4096×6
        pred = net(data, training=False)  # 1×4096×13
        pred = tf.nn.softmax(pred)
        pred = tf.argmax(pred, -1)  # 1×4096
        pred_tensor.append(pred)
    del pred, data
    pred_tensor = tf.concat(pred_tensor, 0)

    test_label = tf.cast(test_label, dtype=tf.int32)  # B×N
    pred_label = tf.cast(pred_tensor, dtype=tf.int32)  # B×N
    del pred_tensor

    # visualization
    color_dict = config.COLOR_DICT
    original_coordinates,original_label = real_data, real_label

    if int(np.sum(original_label - test_label.numpy())) != 0:
        warnings.warn("Original coordinates is inconsistent with evaluated points")

    # IOU
    iou_list = []
    category_acc = np.zeros(shape=(config.NUM_CLASSES,))
    for i in range(config.NUM_CLASSES):
        # mask
        mask_tensor = tf.fill(test_label.shape,i)
        test_sub_label = tf.cast(tf.equal(test_label,mask_tensor),tf.int32)
        pred_sub_label = tf.cast(tf.equal(pred_label,mask_tensor),tf.int32)

        # tp
        tp = tf.reduce_sum(
            tf.multiply(test_sub_label, pred_sub_label)
        )
        category_acc[i] = tp / tf.reduce_sum(test_sub_label)
        fp_fn = tf.reduce_sum(
            tf.abs(test_sub_label - pred_sub_label)
        )
        iou = (tp / (tp + fp_fn)).numpy()
        iou_list.append(iou)
    m_iou = np.average(iou_list)
    # OA
    oa = tf.reduce_sum(
        tf.cast(tf.equal(test_label,pred_label),tf.int32)
    ).numpy() / (test_label.shape[0]*test_label.shape[1])

    with open('精度.txt','a') as fw:
        fw.write(f'\nRoom:{name}')
        fw.write(f'\nOA:{oa}')
        fw.write(f'\nmIOU:{m_iou}')
        fw.write(f'\nIOU Class:')
        for i in range(config.NUM_CLASSES):
            fw.write(f'\n{i}:' + str(iou_list[i]))
        fw.write(f'\nAcc Class:')
        for i in range(config.NUM_CLASSES):
            fw.write(f'\n{i}:' + str(category_acc[i]))

    pred_label = pred_label.numpy()
    test_label = test_label.numpy()

    with open(name + "_predicted.txt", 'w') as fo_pre:
        with open(name + "_labeled.txt", 'w') as fo_lab:
            for i_ in range(original_coordinates.shape[0]):
                for j_ in range(original_coordinates.shape[1]):
                    pred_result = int(pred_label[i_, j_])
                    label_result = int(test_label[i_, j_])
                    fo_pre.write(
                        str(original_coordinates[i_, j_, 0]) + " " +
                        str(original_coordinates[i_, j_, 1]) + " " +
                        str(original_coordinates[i_, j_, 2]) + " " +
                        str(color_dict[pred_result][0]) + " " +
                        str(color_dict[pred_result][1]) + " " +
                        str(color_dict[pred_result][2]) + " " +
                        str(pred_result) + "\n"
                    )
                    fo_lab.write(
                        str(original_coordinates[i_, j_, 0]) + " " +
                        str(original_coordinates[i_, j_, 1]) + " " +
                        str(original_coordinates[i_, j_, 2]) + " " +
                        str(color_dict[label_result][0]) + " " +
                        str(color_dict[label_result][1]) + " " +
                        str(color_dict[label_result][2]) + " " +
                        str(label_result) + "\n"
                    )


if __name__ == '__main__':
    # 记录了处理后的点坐标用于直接喂入神经网络
    test_path = './data/CABDataset.h5'
    # 记录了分块后的原始坐标带有真实标签
    room_path = './data/rooms_name.txt'
    model_path = './log/dgcnn'
    test_name = config.TEST_AREA
    real_path = f'./block/{test_name}_blocked.h5'

    test_idx = []
    with open(room_path,'r') as f:
        rooms_name = f.readlines()
        for idx, room_name in enumerate(rooms_name):
            if test_name in room_name:
                test_idx.append(idx)

    with h5py.File(test_path,'r') as f:
        test_data = f['data'][:]  # B×N×6
        test_label = f['label'][:]  # B×N
    with h5py.File(real_path, 'r') as f:
        real_data = f['data'][:]  # B×N×6
        real_label = f['label'][:]  # B×N

    test_data = test_data[test_idx,...]
    test_label = test_label[test_idx,...]

    inference_single_room(test_data=test_data,
                          test_label=test_label,
                          real_data=real_data,
                          real_label=real_label,
                          file_path=model_path,
                          name=test_name)
