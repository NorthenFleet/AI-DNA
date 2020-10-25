import random
from ai_company.data_process.data_process_util import *
# import ai_company.core as core

class Stragety:
    def __init__(self, stepdata):
        self.stepdata = stepdata
        self.scenario = self.stepdata.game_data.scenario
        self.color = self.stepdata.game_data.color
        self.step_num = self.stepdata.game_data.step_num
        self.y, self.x = None, None
        self.pos = []
        self.move_final = None

        self.occupy_training_scenario = [1231, 2010131194, 2010211129, 2010431153, 2010441253]
        self.move_single_step_training_scenario = [7005]

        self.scenario_info = [
            {"scenario": 1231, "city_master_value": 80, "city_master_coord": 2030, "city_slave_value": 50, "city_slave_coord": 2030,
             "tank_pos_red": [2326, 2327], "automissile_pos_red":1823, "missile_pos_red":2018,  "trops_red": 5136,
             "tank_pos_blue":1835, "automissile_pos_blue":1823, "missile_pos_blue":2018,  "trops_blue": 5136,
             },
            {"scenario": 1531, "city_master_value": 80, "city_master_coord": 2030, "city_slave_value": 50, "city_slave_coord": 2030,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018,  "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018,  "trops_blue": 5136,
             },
            {"scenario": 1631, "city_master_value": 80, "city_master_coord": 2030, "city_slave_value": 50, "city_slave_coord": 2030,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018,  "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018,  "trops_blue": 5136,
             },
            {"scenario": 3231, "city_master_value": 80, "city_master_coord": 2030, "city_slave_value": 50, "city_slave_coord": 2030,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018,  "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018,  "trops_blue": 5136,
             },
            {"scenario": 2010131194, "city_master_value": 80, "city_master_coord": 5561, "city_slave_value": 50, "city_slave_coord": 5561,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018, "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018,  "trops_blue": 5136,
             },

            {"scenario": 2010211129, "city_master_value": 80, "city_master_coord": 4435, "city_slave_value": 50, "city_slave_coord": 4029,
             "tank_pos_red": [5235, 5134], "automissile_pos_red": [5234], "missile_pos_red": [5138], "trops_red": [5136],
             "tank_pos_blue": [4437], "automissile_pos_blue": [4437], "missile_pos_blue": [4437], "trops_blue": [5136],
             },
            {"scenario": 2010431153, "city_master_value": 80, "city_master_coord": 3636, "city_slave_value": 50, "city_slave_coord": 4039,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018,  "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018,  "trops_blue": 5136,
             },
            {"scenario": 2010441253, "city_master_value": 80, "city_master_coord": 3636, "city_slave_value": 50, "city_slave_coord": 3739,
             "tank_pos_red": [2326, 2327], "automissile_pos_red": 1823, "missile_pos_red": 2018,  "trops_red": 5136,
             "tank_pos_blue": 1835, "automissile_pos_blue": 1823, "missile_pos_blue": 2018, "trops_blue": 5136,
             },

            {"scenario": 2030331196, "city_master_value": 80, "city_master_coord": 4435, "city_slave_value": 50,
             "city_slave_coord": 4029,
             "tank_pos_red": [5235, 5134], "automissile_pos_red": [5234], "missile_pos_red": [5138],
             "trops_red": [5136],
             "tank_pos_blue": [4437], "automissile_pos_blue": [4437], "missile_pos_blue": [4437], "trops_blue": [5136],
             },
        ]
        self.scenario_info_id = 0
        for sc in self.scenario_info:
            if self.scenario == sc["scenario"]:
                break
            self.scenario_info_id += 1


        self.tank_done = False
        self.missile_done = False
        self.automissile_done = False

        # 区域移动控制
        self.move_stragety_1231 = []

        # 机动停止取消想定
        self.cancel_move_stop = []

        # 移动控制
        self.switch_move_single_step = False
        self.switch_move_multi_step = False
        self.switch_area_move = False
        self.switch_occupy_move = True

    def is_stragety(self, obj_id, operator, cur_step, move, move_type, x_prob=None, y_prob=None):
        self.obj_id = obj_id
        self.operator = operator
        self.cur_step = cur_step
        self.move = move
        self.move_type = move_type
        self.x_prob = x_prob
        self.y_prob = y_prob
        self.compute_multi_step_move()

        if self.switch_move_single_step is False and self.switch_area_move is False \
                and self.switch_occupy_move is False and self.switch_move_multi_step is False:
            return False
        else:
            return True

    def area_move(self):
        if self.scenario == 1231 and self.color == 0:
            target_pos = random.choice(self.move_stragety_1231)
            y, x = unpack_mapID(target_pos)
            a = range(10)
            k1 = random.choices(a, weights=[7, 7, 7, 6, 5, 4, 3, 2, 1, 1], k=1)
            k2 = random.choices(a, weights=[7, 7, 7, 6, 5, 4, 3, 2, 1, 1], k=1)
            print("k1:", k1, "k1:", k2)
            self.y = y + k1[0]
            self.x = x + k2[0]
            print("战术移动坐标：", self.y, self.x)
            # core.occupy_scenario_1231 = True

    # fixme
    # def designed_point(self):
    #     if self.operator["sub_type"] == 0:
    #         target_pos = random.choice(self.tank_pos_blue)
    #
    #     if self.tank_done is True and self.missile_done is True and self.automissile_done:
    #         target_pos = self.multi_step_move(self.move)
    #     else:
    #         if self.operator["sub_type"] == 1:
    #             target_pos = random.choice(self.missile_pos_blue)
    #             self.missile_done = True
    #         elif self.operator["sub_type"] == 4:
    #             target_pos = random.choice(self.automissile_pos_red)
    #             self.automissile_done = True
    #     self.y, self.x = unpack_mapID(target_pos)

    def ouccupy_move(self):
        if self.cur_step < self.stepdata.game_data.rule_period_first:
            if self.operator["sub_type"] == 0:
                target_pos = np.random.choice(self.scenario_info[self.scenario_info_id]["tank_pos_red"])
            elif self.operator["sub_type"] == 1:
                target_pos = np.random.choice(self.scenario_info[self.scenario_info_id]["missile_pos_red"])
            elif self.operator["sub_type"] == 4:
                target_pos = np.random.choice(self.scenario_info[self.scenario_info_id]["automissile_pos_red"])
        elif self.cur_step < self.stepdata.game_data.rule_period_second:

            if self.operator["sub_type"] == 0:
                target_pos = np.random.choice(self.scenario_info[self.scenario_info_id]["tank_pos_red"])
            elif self.operator["sub_type"] == 1:
                target_pos = np.random.choice(self.scenario_info[self.scenario_info_id]["missile_pos_red"])
            elif self.operator["sub_type"] == 4:
                self.multi_step_move(self.move)
                target_pos = int(self.y * 100 + self.x)
        else:
            if self.operator["sub_type"] in [0, 2]:
                target_pos = random.choice([self.scenario_info[self.scenario_info_id]["city_master_coord"],
                                            self.scenario_info[self.scenario_info_id]["city_slave_coord"]])
            else:
                for op in self.stepdata.my_observations['operators']:
                    if op['color'] == self.stepdata.color and op["sub_type"] == 0:
                        self.multi_step_move(self.move)
                        target_pos = int(self.y * 100 + self.x)
                    else:
                        target_pos = random.choice([self.scenario_info[self.scenario_info_id]["city_master_coord"],
                                                    self.scenario_info[self.scenario_info_id]["city_slave_coord"]])

        if self.operator["sub_type"] == 2:
            target_pos = random.choice([self.scenario_info[5]["city_master_coord"],
                                    self.scenario_info[5]["city_slave_coord"]])

        if self.operator["sub_type"] == 7:
            missile_pos = []
            for op in self.stepdata.my_observations['operators']:
                if op['color'] != self.stepdata.color:
                    missile_pos.append(op['cur_hex'])
            if missile_pos != []:
                pass
            else:
                missile_pos.append(self.scenario_info[5]["city_master_coord"])
                missile_pos.append(self.scenario_info[5]["city_slave_coord"])
            target_pos = random.choice(missile_pos)

        distance = 1000
        for index in range(len(self.pos)):
            if self.pos[index] == target_pos:
                self.move_final = index
                break
            else:
                target_y, target_x = unpack_mapID(target_pos)  # 目标点坐标
                pos_y, pos_x = unpack_mapID(self.pos[index])
                diff_distance = self.grid_distant_level(target_x, target_y, pos_x, pos_y)
                if diff_distance < distance:
                    distance = diff_distance
                    self.move_final = index
        self.multi_step_move(self.move_final)

    def single_step_move(self):
        # 选择位置
        y, x = unpack_mapID(self.coord)

        pos = [0] * 6
        pos[0] = (y - 1) * 100 + (x + 1 - 2**((y-1) % 2))
        pos[1] = (y - 1) * 100 + (x + 2 - 2**((y-1) % 2))
        pos[2] = y * 100 + (x - 1)
        pos[3] = y * 100 + (x + 1)
        pos[4] = (y + 1) * 100 + (x + 1 - 2**((y-1) % 2))
        pos[5] = (y + 1) * 100 + (x + 2 - 2**((y-1) % 2))

        # 提取概率
        x_prob, y_prob = self.get_single_move_pro(x, y)
        y_0 = y_prob[0] / sum(y_prob)
        y_1 = y_prob[1] / sum(y_prob)
        y_2 = y_prob[2] / sum(y_prob)
        x_0 = x_prob[0] / sum(x_prob)
        x_1 = x_prob[1] / sum(x_prob)
        x_2 = x_prob[2] / sum(x_prob)

        # fixme
        sum_prob_0 = sum([y_0 * x_0, y_0 * x_1, y_1 * x_0, y_1 * x_2,
                        y_2 * x_0, y_2 * x_1])
        sum_prob_1 = sum([y_0 * x_1, y_0 * x_2, y_1 * x_0, y_1 * x_2,
                        y_2 * x_1, y_2 * x_2])

        if y % 2 == 0:
            single_move_position = np.random.choice(a=pos, size=1, replace=False,
                                                    p=np.dot([y_0 * x_0, y_0 * x_1, y_1 * x_0,
                                                              y_1 * x_2, y_2 * x_0,
                                                              y_2 * x_1], 1 / sum_prob_0))
            print(np.dot([y_0 * x_0, y_0 * x_1, y_1 * x_0, y_1 * x_2, y_2 * x_0, y_2 * x_1], 1 / sum_prob_0))
        else:
            single_move_position = np.random.choice(a=pos, size=1, replace=False,
                                                    p=np.dot([y_0 * x_1, y_0 * x_2, y_1 * x_0,
                                                              y_1 * x_2, y_2 * x_1,
                                                              y_2 * x_2], 1 / sum_prob_1))
            print(np.dot([y_0 * x_1, y_0 * x_2, y_1 * x_0, y_1 * x_2, y_2 * x_1, y_2 * x_2], 1 / sum_prob_1))

        # fixme
        if self.move_type != -1:
            single_move_position = pos[self.move_type]
            self.y, self.x = unpack_mapID(single_move_position)
            while self.y < 0 or self.y > 92 or self.x < 0 or self.x > 77:
                single_move_position = random.choice(pos)
                self.y, self.x = unpack_mapID(single_move_position)
        else:
            self.y, self.x = unpack_mapID(single_move_position)
            while self.y < 0 or self.y > 92 or self.x < 0 or self.x > 77:
                single_move_position = random.choice(pos)
                self.y, self.x = unpack_mapID(single_move_position)
            print("纵坐标：", self.y, "横坐标：", self.x)

    def compute_multi_step_move(self):
        # 如果最大可走n步，本函数列出从yx出发，所有可能走到的坐标
        # 输出是一个位置列表，0-5是走一步，6-17是走2步，18-35是走3步，36-59是走4步，60-89是走5步
        n = self.step_num
        y, x = unpack_mapID(self.operator['cur_hex'])
        num = 3 * n * (n + 1)
        pos = [0] * num
        a = 2 ** ((y - 1) % 2)

        for h in range(n):
            pos[3 * (h + 1) - 1 + 3 * h * (h + 1)] = y * 100 + (x - h - 1)  # 当前坐标加上偏移量
            pos[3 * (h + 1) + 3 * h * (h + 1)] = y * 100 + (x + h + 1)

            for i in range(h + 1):  # h+1表示本轮的定步长
                pos_1 = 3 * (h + 1) - 3 - 2 * i + 3 * h * (h + 1)  # 位置下标
                pos_2 = 3 * (h + 1) - 2 - 2 * i + 3 * h * (h + 1)
                pos_3 = 3 * (h + 1) + 1 + 2 * i + 3 * h * (h + 1)
                pos_4 = 3 * (h + 1) + 2 + 2 * i + 3 * h * (h + 1)
                y_1 = y - i - 1
                y_2 = y + i + 1
                if i % 2 == 0:
                    x_1 = x + 2 - a - (h + 1) + i / 2
                    x_2 = x + 1 - a + (h + 1) - i / 2
                    pos[pos_1] = y_1 * 100 + x_1
                    pos[pos_2] = y_1 * 100 + x_2
                    pos[pos_3] = y_2 * 100 + x_1
                    pos[pos_4] = y_2 * 100 + x_2
                else:
                    x_1 = x - (h + 1) + (i + 1) / 2
                    x_2 = x + (h + 1) - (i + 1) / 2
                    pos[pos_1] = y_1 * 100 + x_1
                    pos[pos_2] = y_1 * 100 + x_2
                    pos[pos_3] = y_2 * 100 + x_1
                    pos[pos_4] = y_2 * 100 + x_2
                if (i != 0) and (i == h):  # 计算上下两行位置的坐标
                    x_3 = pos[h + 3 * h * (h + 1)]
                    x_4 = pos[6 * (h + 1) - 1 - h + 3 * h * (h + 1)]
                    pos[0 + 3 * h * (h + 1)] = x_3
                    pos[6 * (h + 1) - 1 + 3 * h * (h + 1)] = x_4
                    for j in range(h):
                        pos[j + 1 + 3 * h * (h + 1)] = x_3 + j + 1
                        pos[6 * (h + 1) - 2 - j + 3 * h * (h + 1)] = x_4 - j - 1
        for g in range(num):
            self.pos.append(int(pos[g]))

    # 多步移动
    def multi_step_move(self, move):
        self.y, self.x = unpack_mapID(self.pos[move])

    # 提取单步移动的位置的概率
    def get_single_move_pro(self, x, y):
        x_indices = [x-1, x, x +1]
        y_indices = [y-1, y, y +1]
        x_prob = [self.x_prob[x_indices[0]], self.x_prob[x_indices[1]], self.x_prob[x_indices[2]]]
        y_prob = [self.x_prob[y_indices[0]], self.y_prob[y_indices[1]], self.x_prob[y_indices[2]]]
        return x_prob, y_prob

    def get_pos(self):
        if self.switch_area_move is True:
            self.area_move()
        if self.switch_occupy_move is True and self.scenario in self.occupy_training_scenario:
            self.ouccupy_move()
        # if self.switch_move_single_step is True and self.scenario in self.move_single_step_training_scenario:
        if self.switch_move_single_step is True:
            self.single_step_move()
        if self.switch_move_multi_step is True:
            self.multi_step_move(())
        return int(self.y * 100 + self.x)

    def is_stop(self, result):
        # if self.scenario in self.cancel_move_stop and self.color == 0:
        if self.scenario in self.cancel_move_stop:
            return result
        else:
            return 0

    def grid_distant_level(self, x1, y1, x2, y2):
        xyz1 = self.colrow_to_xyz(x1, y1)
        xyz2 = self.colrow_to_xyz(x2, y2)
        num = int(np.sum(np.abs(xyz1 - xyz2)) / 2)
        return num

    def colrow_to_xyz(self, col, row):
        x = col
        z = row - (col - (col & 1)) / 2
        y = -x - z
        return np.array([x, y, z])

    def get_move(self):
        return self.move_final
