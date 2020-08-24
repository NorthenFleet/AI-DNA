"""
author:zk
environment：win8.1 xlrd1.2.0 numpy1.18.1 pandas1.0.1
time:2020.03.20
自动获取地图长宽，读取高程数据，输出numpy类型矩阵
"""
import xlrd
import numpy as np
import pandas as pd


class MapData:

    def __init__(self, height_max, width_max, HexID="HexID", GroundID="GroundID", Cond="Cond", GridID="GridID", GridType="GridType"):
        # excel 中的ID
        self.HexID = HexID
        self.GroundID = GroundID
        self.Cond = Cond
        self.GridID = GridID
        self.GridType = GridType

        # 类存储的地图信息
        self.height_max = height_max
        self.width_max = width_max
        self.map_altitude_matrix = -1
        self.map_altitude_matrix_relative = -1
        self.map_type_matrix = -1
        self.altitude_min = -1

        # 格间信息
        self.map_ati = None
        self.map_road = None
        self.map_stream = None

    def _map_altitude_type(self, table):
        # 处理地图高程信息
        # 获取MapID和GroundID的位置
        row = table.row(0)
        HexID_pos = 0
        GroundID_pos = 0
        Cond_pos = 0
        GridID_pos = 0
        GridType_pos = 0

        for i in range(len(row)):
            if row[i].value == self.HexID:
                HexID_pos = i
            if row[i].value == self.GroundID:
                GroundID_pos = i
            if row[i].value == self.Cond:
                Cond_pos = i
            if row[i].value == self.GridID:
                GridID_pos = i
            if row[i].value == self.GridType:
                GridType_pos = i

        # 获取excel中的数据
        HexID_List = table.col(HexID_pos)
        GroundID_list = table.col(GroundID_pos)
        Cond_List = table.col(Cond_pos)
        GridID_list = table.col(GridID_pos)
        GridType_list = table.col(GridType_pos)

        altitude_matrix = np.full(shape=(self.height_max, self.width_max), fill_value=0)
        type_matrix = np.full(shape=(self.height_max, self.width_max), fill_value=-1)
        altitude_relative = np.full(shape=(self.height_max, self.width_max), fill_value=0)

        # 填入
        for i in range(1, len(HexID_List)):
            # 坐标
            height = int(HexID_List[i].value) // 100
            width = int(HexID_List[i].value) % 100
            if height >= self.height_max or width >= self.width_max:
                continue
            # 高程
            altitude_matrix[height, width] = int(GroundID_list[i].value)
            # 类型
            Co = int(Cond_List[i].value)
            GID = int(GridID_list[i].value)
            GTy = int(GridType_list[i].value)
            # 判断类型
            maptype = self._maptype(Co, GID, GTy)
            assert maptype + 0.5  # 出现没见过的地图类型
            type_matrix[height, width] = maptype

        # 生成相对高程信息
        altitude_min = altitude_matrix.min()
        for height in range(0, altitude_matrix.shape[0]):
            for width in range(altitude_matrix.shape[1]):
                if altitude_matrix[height, width] < -0.5:
                    continue
                else:
                    altitude_relative[height, width] = altitude_matrix[height, width] - altitude_min + 1

        self.map_altitude_matrix = altitude_matrix
        self.map_altitude_matrix_relative = altitude_relative
        self.map_type_matrix = type_matrix
        self.altitude_min = altitude_min

    def _maptype(self, Co, GID, GTy):
        # 判断地图上一个点的类型
        if Co == 0 and GTy == 0:  # 开阔地
            return 0
        if GTy == 3 and GID == 52:  # 丛林地
            return 1
        if GTy == 2 and Co == 8:  # 道路穿过的丛林地
            return 1
        if Co == 8 and (GTy == 0 or GTy == 2):  # 道路穿过的丛林地
            return 1
        if GTy == 2 and Co == 7:  # 道路穿过的居民地
            return 2
        if GTy == 3 and GID == 51:  # 居民地
            return 2
        if Co == 7 and (GTy == 0 or GTy == 2):  # 道路穿过的居民地
            return 2
        if GTy == 1 and GID == 25:  # 面状水系
            return 3
        if Co == 6:  # 松软地
            return 4
        return 0

    def read_mapexcel(self, xlspath):
        # 打开excel
        data = xlrd.open_workbook(xlspath)
        # 通过文件名获得工作表,获取工作表1
        table = data.sheet_by_name('查询')
        # 获取高程和类型地图
        self._map_altitude_type(table)

    def read_map(self, np_path):
        map = np.load(np_path)
        self.map_ati = map[0, :, :, :]
        self.map_road = map[1, :, :, :]
        self.map_stream = map[2, :, :, :]
        # 统一尺寸
        temp1 = np.zeros((self.height_max, self.width_max, 6), dtype="uint8")
        temp2 = np.zeros((self.height_max, self.width_max, 6), dtype="uint8")
        temp3 = np.zeros((self.height_max, self.width_max, 6), dtype="uint8")
        temp1[0:self.map_ati.shape[0], 0:self.map_ati.shape[1], :] = self.map_ati
        temp2[0:self.map_ati.shape[0], 0:self.map_ati.shape[1], :] = self.map_road
        temp3[0:self.map_ati.shape[0], 0:self.map_ati.shape[1], :] = self.map_stream
        self.map_ati = temp1
        self.map_road = temp2
        self.map_stream = temp3

    def get_map_ati(self):
        return self.map_ati

    def get_map_road(self):
        return self.map_road

    def get_map_stream(self):
        return self.map_stream

    def get_map_matrix_2D(self):
        # 合成矩阵
        map_matrix = np.zeros((2, self.height_max, self.width_max))
        map_matrix[0] = self.map_altitude_matrix_relative
        map_matrix[1] = self.map_type_matrix
        return map_matrix

    def get_altitude_matrix(self):
        # 返回高度矩阵
        return self.map_altitude_matrix

    def get_altitude_matrix_relative(self):
        # 返回相对高度矩阵
        return self.map_altitude_matrix_relative

    def get_type_matrix(self):
        # 返回地图类型矩阵
        return self.map_type_matrix

    def get_altitude_min(self):
        # 返回地图类型矩阵
        return self.altitude_min


def demo1():
    # 使用MapData读取数据
    path = "../../data/高原通道.xls"
    mapdata = MapData(70, 66)
    mapdata.read_mapexcel(path)

    # 获取类内数据
    map_matrix_2D = mapdata.get_map_matrix_2D()
    altitude_matrix = mapdata.get_altitude_matrix()
    altitude_matrix_relative = mapdata.get_altitude_matrix_relative()
    type_matrix = mapdata.get_type_matrix()
    altitude_min = mapdata.get_altitude_min()

    # 保存看结果,保存成csv
    data = pd.DataFrame(altitude_matrix_relative)  # 高程
    data.to_csv('altitude_matrix_relative.csv', header=False, index=False, )
    data = pd.DataFrame(type_matrix)  # 高程
    data.to_csv('type_matrix.csv', header=False, index=False, )
    data = pd.DataFrame(altitude_matrix)  # 高程
    data.to_csv('altitude_matrix.csv', header=False, index=False, )
    # data.to_csv('type_matrix.csv')  # 是否要xy轴坐标


if __name__ == '__main__':
    demo1()
