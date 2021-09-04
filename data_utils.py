import config
import numpy as np
import h5py
import os


def txt2Matrix():

    # 获取数据文件路径
    with open(os.path.join(config.DATA_PATH, config.DATA_FILE), 'r') as fr:
        file_name_list = [_.strip() for _ in fr.readlines()]

    data = []
    label = []
    rooms = []
    # 依次读取文件内容
    print('读取TXT文件')
    for room_name_i in file_name_list:
        print(f'\t读取{room_name_i}')
        with open(os.path.join(config.DATA_PATH, room_name_i), 'r') as fr:
            data_label = np.asarray([_.split() for _ in fr.readlines()], dtype=np.float32)
            data.append(data_label[...,:-1])  # feats
            label.append(data_label[...,-1])  # l
            rooms.append(room_name_i.split('.')[0])
    print('读取TXT完成')

    for i in range(len(data)):
        data[i][...,:3] -= np.amin(data[i][...,:3],axis=0)

    return data, label, rooms


def randomSample(point_cloud, point_cloud_label, point_cloud_name, num_sample):
    """
    对点云进行随机采样，输入的点云为二维numpy数组，第三列为原始z坐标。
    :param point_cloud: 二维numpy数组
    :param point_cloud_label: 一维numpy数组
    :param point_cloud_name: 列表
    :param num_sample: 采样点的数量
    :return:
    """

    if isinstance(point_cloud, np.ndarray):
        if len(point_cloud.shape) != 2:
            raise ValueError('输入的采样点云必须是二维numpy数组')
        elif point_cloud.shape[-1] < 3:
            raise ValueError('输入的采样点云必须多于两列')
    else:
        raise TypeError('输入的采样点云必须是numpy.ndarray类型')

    random_index = list(range(point_cloud.shape[0]))
    # np.random.seed(1024)
    batch_index = []
    while len(random_index) > 0:
        np.random.shuffle(random_index)
        if len(random_index) >= num_sample:
            batch_index.append(random_index[:num_sample])
            del random_index[:num_sample]
        else:
            batch_index.append(
                random_index + list(np.random.choice(random_index, num_sample - len(random_index)))
            )
            del random_index[:len(random_index)]

    data = np.zeros([len(batch_index), num_sample, point_cloud.shape[1]], dtype=np.float32)  # B×N×A
    label = np.zeros([len(batch_index), num_sample], dtype=np.int64)  # B×N
    rooms = []  # B
    for i, p_idx in enumerate(batch_index):
        data[i] = point_cloud[p_idx]
        label[i] = point_cloud_label[p_idx]
        rooms.append(point_cloud_name + '_batch' + str(i))

    return data, label, rooms


def room2Blocks(point_cloud, point_cloud_label, point_cloud_name, stride, block_size, num_sample):
    """
    对点云进行规则固定分块，并对每个分块点云进行z采样
    :param point_cloud: 输入点云numpy数组，形状必须为二维
    :param point_cloud_label: 输入点云标签
    :param point_cloud_name: 输入房间名称
    :param stride: 滑动步长
    :param block_size: 正方形分块边长
    :param num_sample: 每块中每批次采样点数量
    :return:
    """

    if isinstance(point_cloud, np.ndarray):
        if len(point_cloud.shape) != 2:
            raise ValueError('输入的采样点云必须是二维numpy数组')
        elif point_cloud.shape[-1] < 3:
            raise ValueError('输入的采样点云必须多于两列')
    else:
        raise TypeError('输入的采样点云必须是numpy.ndarray类型')

    # 点云分块
    limit = np.amax(point_cloud, 0)[0:3]
    # #生成角点对
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride)
            ybeg_list.append(j * stride)

    # #块集
    block_data_list = []
    block_label_list = []
    block_name_list = []
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (point_cloud[:, 0] <= xbeg + block_size) & (point_cloud[:, 0] >= xbeg)
        ycond = (point_cloud[:, 1] <= ybeg + block_size) & (point_cloud[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # 少于100个点抛弃
            continue
        block_data = point_cloud[cond]
        block_label = point_cloud_label[cond]
        block_name = point_cloud_name + '_block' + str(idx)

        block_data, block_label, block_name = randomSample(block_data, block_label, block_name, num_sample)
        block_data_list.append(block_data)
        block_label_list.append(block_label)
        block_name_list.extend(block_name)
    data = np.concatenate(block_data_list, 0)  # B×N×A
    del block_data_list
    label = np.concatenate(block_label_list, 0)  # B×N
    del block_label_list
    rooms = block_name_list  # B
    del block_name_list

    return data, label, rooms


def room2BlocksWrapper(point_cloud, point_cloud_label, point_cloud_name, stride, block_size, num_sample, is_save=True):
    """
    对多个点云进行分块处理
    :param point_cloud: 列表
    :param point_cloud_label: 列表
    :param point_cloud_name: 列表
    :param stride:
    :param block_size:
    :param num_sample:
    :param is_save:
    :return:
    """

    if not (isinstance(point_cloud, list) & isinstance(point_cloud_label, list) & isinstance(point_cloud_name, list)):
        raise TypeError('待分块点云必须为list')
    elif not (len(point_cloud) == len(point_cloud_label) == len(point_cloud_name)):
        raise ValueError('待分块点云、标签和名称列表必须一一对应')

    data = []
    label = []
    rooms = []
    print('\n分块点云')
    for idx in range(len(point_cloud)):
        print(f'\t分块{point_cloud_name[idx]}')
        data_, label_, rooms_ = room2Blocks(point_cloud[idx],
                                            point_cloud_label[idx],
                                            point_cloud_name[idx],
                                            stride,
                                            block_size,
                                            num_sample)
        data.append(data_)
        label.append(label_)
        rooms.append(rooms_)
    print('分块完成')
    print('\n正在保存分块点云')
    if is_save:
        # 保存格式 X Y Z R G B L
        for i in range(len(point_cloud_name)):
            with h5py.File(point_cloud_name[i] + "_blocked.h5", 'w') as fo_h5:
                fo_h5['data'] = data[i]
                fo_h5['label'] = label[i]
            with open(point_cloud_name[i] + '_blocked.txt', 'w') as fw:
                for j in range(data[i].shape[0]):
                    for k in range(data[i].shape[1]):
                        fw.write(
                            ' '.join(list(map(str, data[i][j,k][...,:6])) +
                                     config.COLOR_DICT[int(label[i][j,k])] +
                                     [str(label[i][j,k])]
                                     ) + '\n'
                        )

    return data, label, rooms


def formatAttribute(point_cloud, block_size):
    """
    格式化点属性
    输入格式：绝对坐标 X Y Z R G B
    格式：中心化坐标 x y Z 标准化坐标 x' y' z' 归一化r g b 极坐标ρθφ 特征值 l1l2l3 normal rate
    :param block_size:
    :param point_cloud: 输入点云，三维numpy数组，形状为B×N×A
    :return:
    """

    if not isinstance(point_cloud, np.ndarray):
        raise TypeError('待格点云必须为numpy数组')
    elif len(point_cloud.shape) < 3:
        raise ValueError('待格点云必须为3维numpy数组')
    # elif point_cloud.shape[-1] != 6:
    #     raise ValueError('待格点云属性必须为绝对坐标 X Y Z R G B')

    max_room_x = np.max(point_cloud[:, :, 0])
    max_room_y = np.max(point_cloud[:, :, 1])
    max_room_z = np.max(point_cloud[:, :, 2])
    max_room_r = np.sqrt(max_room_x ** 2 + max_room_y ** 2 + max_room_z ** 2)

    data = np.zeros((point_cloud.shape[0], point_cloud.shape[1], 9))
    for idx in range(point_cloud.shape[0]):
        data[idx, :, 3] = point_cloud[idx, :, 0] / max_room_x
        data[idx, :, 4] = point_cloud[idx, :, 1] / max_room_y
        data[idx, :, 5] = point_cloud[idx, :, 2] / max_room_z
        data[idx, :, 6] = point_cloud[idx, :, 3] / 255.  # r
        data[idx, :, 7] = point_cloud[idx, :, 4] / 255.  # g
        data[idx, :, 8] = point_cloud[idx, :, 5] / 255.  # b
        # data[idx, :, 9] = np.sqrt((point_cloud[idx, :, 0] ** 2 + point_cloud[idx, :, 1] ** 2 + point_cloud[idx, :, 2] ** 2))
        # data[idx, :, 10] = np.arccos(point_cloud[idx, :, 2] / (data[idx, :, 9] + 1e-9))
        # data[idx, :, 11] = np.arctan(point_cloud[idx, :, 1] / (point_cloud[idx, :, 0] + 1e-9)) * (2 / np.pi)
        # data[idx, :, 9] /= max_room_r
        # data[idx, :, 12] = point_cloud[idx, :, 6] * 100
        # data[idx, :, 13] = point_cloud[idx, :, 7] * 100
        # data[idx, :, 14] = point_cloud[idx, :, 8] * 100
        # data[idx, :, 15] = point_cloud[idx, :, 9]
        minx = np.min(point_cloud[idx, :, 0])
        miny = np.min(point_cloud[idx, :, 1])
        point_cloud[idx, :, 0] -= (minx + block_size / 2)
        point_cloud[idx, :, 1] -= (miny + block_size / 2)
    data[:, :, :3] = point_cloud[:, :, :3]

    return data


def formatAttributeWrapper(point_cloud, block_size):

    if not isinstance(point_cloud, list):
        raise TypeError('待格式化点云必须为list')
    print('\n格式化点云')
    return [formatAttribute(_,block_size) for _ in point_cloud]


def generateDataset(zoom_list, zoom_factor, stride, block_size, num_sample):
    # 读文件
    cab_data, cab_label, cab_rooms = txt2Matrix()
    # 分块
    cab_data, cab_label, cab_rooms = room2BlocksWrapper(cab_data,cab_label,cab_rooms,stride,block_size,num_sample)
    # 格式化
    cab_data = formatAttributeWrapper(cab_data,block_size)
    # 保存
    cab_data = np.concatenate(cab_data,0)
    cab_label = np.concatenate(cab_label,0)
    cab_rooms = list(np.concatenate(cab_rooms,0))
    print('\n保存最终文件')
    with h5py.File("CABDataset.h5", 'w') as h5_fo:
        h5_fo['data'] = cab_data
        h5_fo['label'] = cab_label
        with open("data/rooms_name.txt", 'w') as txt_fo:
            for room_name in cab_rooms:
                txt_fo.write(room_name + '\n')


if __name__ == "__main__":
    generateDataset(zoom_list=None, zoom_factor=config.ZOOM_FACTOR, stride=1., block_size=1., num_sample=4096)
