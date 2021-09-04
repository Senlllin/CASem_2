import h5py
import os
import numpy as np
from platform import system


def getDataFiles(fpath):
    with open(fpath,'r') as fr:
        lines = [line.strip() for line in fr.readlines()]
    return lines


def loadH5Files(fpath):
    file = h5py.File(fpath,'r')
    data = file['data'][:]
    label = file['label'][:]
    file.close()
    return data, label


def loadH5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def loadDataFile(h5_filename):
    return loadH5(h5_filename)


def getPath(path1,path2):
    return os.path.join(path1,path2)


def match_postfix(postfix, path=None, iscwd=False):
    """match all pointed files in the directory
    :param path: root path
    :param postfix: postfix without “.”
    :param iscwd:
    :return: list
    """
    if iscwd:
        path = os.getcwd()
    return [
        os.path.join(roots,i)
        for roots, dirs, files in os.walk(path, followlinks=False) for i in files if(i.split('.')[-1] == postfix)
    ]


def removeFiles(file_list):
    for i in file_list:
        os.remove(i)


def getSystemOS():
    """
    acquire system OS
    :return: windows:0 linux:1
    """
    c = -1
    my_os = system()
    if my_os == 'Windows': c = 0
    if my_os == 'Linux': c = 1
    return c


def getPathSeparator():
    return "/" if getSystemOS() else "\\"


def getColorDict():
    return {
        0: [255, 215, 0],  # 屋顶
        1: [30, 144, 255],  # 斗栱
        2: [205, 0, 0],  # 柱
        3: [255, 218, 185],  # 台基
        4: [220, 220, 220]  # 其它
    }


def getOriginalCoordinates(name,file_path):
    file_list = match_postfix('h5',file_path)
    h5_coordinates = []
    h5_labels = []
    for file_ in file_list:
        if name in file_:
            with h5py.File(file_,'r') as h5_fo:
                h5_coordinates = h5_fo['data'][:]
                h5_labels = h5_fo['label'][:]
            break
    return h5_coordinates,h5_labels


def getDistribution(path,classes):
    """
    统计点云类别分布
    :param path: 点云根目录
    :param classes: 类别的数量
    :return: 字典，｛文件名称1：[类别1数量，类别2数量，...]，文件名称2：[...]，...｝
    """
    file_list = match_postfix('txt',path)
    name_list = [_.split(getPathSeparator())[-1][:-4] for _ in file_list]
    stat_dict = {name_list[_]:[0 for __ in range(classes)] for _ in range(len(name_list))}
    for file in file_list:
        with open(file,'r') as fi:
            lines = [_.strip() for _ in fi.readlines()]
            number_list = [int(_.split(" ")[-1]) for _ in lines]
            file_name = [_ for _ in name_list if _ in file][0]
            for label_number in number_list:
                stat_dict[file_name][label_number] += 1
    return stat_dict


def addLabel(fpath,save_name):
    """
    为没有标注的点云添加标签，添加值为-1，添加位置为最后一列。
    x y z -> x y z l
    :param fpath: 点云txt
    :param save_name:
    :return: None
    """
    with open(fpath,'r') as f:
        lines = [_.strip()+' -1\n' for _ in f.readlines()]
        with open(save_name,'w') as fo:
            for i in lines:
                fo.write(i)

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros_like(batch_data, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.matmul(shape_pc, rotation_matrix)
    return rotated_data
