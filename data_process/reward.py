from ai_company.data_process.data_process_util import *

RED = 0
BLUE = 1

class Reward:
    def __init__(self, stepdata):
        self.stepdata = stepdata
        self.delta = 1
        self.max_move_reward = self.stepdata.game_data.step_num * self.delta
        self.max_operateor_reward = 8
        self.max_occupy_reward = 80
        self.switch_movement_reward = True
        self.switch_movement_penalty = False
        # self.penalties = [5201, 5401, 5602, 5603, 5504, 5405]
        self.penalties = []
        self.movement_occupy_distance = 10

        if self.stepdata.color == RED:
            self.color_str ='red'
            self.enemy_color_str = 'blue'
        else:
            self.color_str = 'blue'
            self.enemy_color_str = 'red'

    def get_total_reward(self):
        basic_reward, win, enemy = self.basic_reward()
        occupy_reward = self.occupy_reward()
        operator_reward = self.operator_reward()
        movement_reward = self.movement_reward()
        movement_penalty = self.movement_penalty()
        last_action_reward = self.last_action_reward()
        # reward = basic_reward + movement_reward + last_action_reward
        reward = basic_reward + last_action_reward + movement_penalty
        return reward, basic_reward, win, enemy

    ################################################### 基础奖励奖励 ###################################################
    def basic_reward(self):
        # 基本奖励
        if self.stepdata.game_data.first_reward is False:
            if self.stepdata.game_data.basic_reward != self.stepdata.scores["{}_win".format(self.color_str)]:
                reward = self.stepdata.scores["{}_win".format(self.color_str)] - self.stepdata.scores["{}_win".format(self.enemy_color_str)]
                self.stepdata.game_data.basic_reward = self.stepdata.scores["{}_win".format(self.color_str)]
            else:
                reward = 0
        else:
            self.stepdata.game_data.basic_reward = self.stepdata.scores["{}_occupy".format(self.color_str)]
            self.stepdata.game_data.first_reward = False
            reward = 0
        win = self.stepdata.scores["{}_win".format(self.color_str)]
        enemy_win = self.stepdata.scores["{}_win".format(self.enemy_color_str)]

        return reward, win, enemy_win
        # if self.stepdata.game_data.occupy_reward != (self.stepdata.scores["{}_occupy".format(color_str)] -
        #                                              self.stepdata.scores["{}_occupy".format(enemy_color_str)]):
        #     reward = (self.stepdata.scores["{}_occupy".format(color_str)] - self.stepdata.scores[
        #         "{}_occupy".format(enemy_color_str)]) - self.stepdata.game_data.occupy_reward
        #     self.stepdata.game_data.occupy_reward = (self.stepdata.scores["{}_occupy".format(color_str)] -
        #                                             self.stepdata.scores["{}_occupy".format(enemy_color_str)])
        #     if reward > 0:
        #         print("我方夺控")
        #     else:
        #         print("敌方夺控")
        # else:
        #     reward = 0
        # return reward/self.max_reward

    def occupy_reward(self):
        # 基本奖励
        color_str = 'red' if self.stepdata.color == RED else 'blue'
        enemy_color_str = 'blue' if color_str == RED else 'red'
        if self.stepdata.game_data.occupy_reward != self.stepdata.scores["{}_occupy".format(color_str)]:
            reward = self.stepdata.scores["{}_occupy".format(color_str)] - self.stepdata.game_data.occupy_reward
            self.stepdata.game_data.occupy_reward = self.stepdata.scores["{}_occupy".format(color_str)]
            print("我方夺控")
        else:
            reward = 0
        return reward

    ################################################### 策略奖励 ###################################################
    def operator_reward(self):
        # 算子得分奖励
        reward_damage = 0
        if len(self.stepdata.judge_info) != 0:
            for n in range(len(self.stepdata.judge_info)):
                id_judge_info = self.stepdata.judge_info[n]
                tar_obj_id = id_judge_info["target_obj_id"]
                damage_info = id_judge_info["damage"]
                # 判断被攻击方
                if tar_obj_id in self.stepdata.all_enemy_operator_ids:
                    sign = 1
                else:
                    sign = -1

                # 算子被打掉给予修正奖励
                if tar_obj_id not in self.stepdata.alive_enemy_operator_ids and tar_obj_id not in self.stepdata.alive_my_operator_ids:
                    if sign == 1:
                        print("消灭一个敌方算子")
                    else:
                        print("消灭一个我方算子")
                    # reward_damage = id_info["value"] * sign
                    reward_damage = self.stepdata.game_data.last_attack * sign
                # 算子没有被打掉正常计算
                else:
                    id_info = self.stepdata.alive_operators_map[tar_obj_id]
                    max_blood = id_info["max_blood"]
                    # 打到了按正常奖励计算
                    if damage_info > 0:
                        reward_damage = damage_info / max_blood * id_info["value"] * sign
                        if sign == 1:
                            print("我方射击伤害：", reward_damage)
                        else:
                            print("敌方射击伤害：", reward_damage)
                    # 没打到给予一定的奖励修正
                    else:
                        reward_damage = 0
                        # fixme 奖励有正有负
                        # if sign == -1:
                        #     reward_damage = id_judge_info["ori_damage"] / max_blood * id_info["value"]
                        #     print("原始伤害", id_judge_info["ori_damage"])
                        #     print("我方躲避伤害奖励：", reward_damage)
                        # else:
                        #     reward_damage = - id_judge_info["ori_damage"] / max_blood * id_info["value"]
                        #     print("原始伤害", id_judge_info["ori_damage"])
                        #     print("敌方躲避伤害奖励：", reward_damage)

        return reward_damage

    def movement_penalty(self):
        if self.switch_movement_penalty is True:
            reward_occupy_stragety = 0
            reward_occupy_list_negative = []
            reward_occupy_list_positive = []
            if self.stepdata.game_data.last_move:
                if self.stepdata.game_data.move_flag:
                    last_x, last_y, obj_id = self.stepdata.game_data.last_move
                    last_point = last_y*100 + last_x
                    if obj_id in self.stepdata.alive_operators_map.keys():
                        if self.stepdata.alive_operators_map[obj_id]["sub_type"] not in self.stepdata.unoccupy_subtype:
                            cur_y, cur_x = unpack_mapID(self.stepdata.alive_operators_map[obj_id]["cur_hex"])  # 算子当前坐标
                            for city in self.stepdata.cities:
                                occupy_y, occupy_x = unpack_mapID(city["coord"])  # 夺控点坐标
                                tar_distance = self.stepdata.grid_distant_level(last_x, last_y, occupy_x, occupy_y)
                                cur_distance = self.stepdata.grid_distant_level(cur_x, cur_y, occupy_x, occupy_y)
                                dif_distance = cur_distance - tar_distance
                                if tar_distance > self.movement_occupy_distance:
                                    if city["value"] == 80:
                                        reward_occupy_temp = dif_distance * self.delta
                                        # print("算子ID：", obj_id, "当前位置：", cur_x, cur_y, "移动目标位置：", last_x, last_y, "主夺控点距离奖励：",
                                        #       reward_occupy_temp, )
                                    else:
                                        reward_occupy_temp = dif_distance * (50 / 80) * self.delta
                                        # print("算子ID：", obj_id, "当前位置：", cur_x, cur_y, "移动目标位置：", last_x, last_y, "次夺控点距离奖励：",
                                        #       reward_occupy_temp)
                                    if reward_occupy_temp > 0:
                                        reward_occupy_list_positive.append(reward_occupy_temp)
                                    else:
                                        reward_occupy_list_negative.append(reward_occupy_temp)

                            if last_point in self.penalties:
                                reward_occupy_stragety = -5
                        else:
                            reward_occupy_stragety = 0
                    else:
                        reward_occupy_stragety = 0
                    self.stepdata.game_data.move_flag = False
                    return reward_occupy_stragety / self.max_move_reward
            return reward_occupy_stragety / self.max_move_reward
        else:
            return 0

    def movement_reward(self):
        if self.switch_movement_reward is True:
            reward_occupy_stragety = 0
            reward_occupy_list_negative = []
            reward_occupy_list_positive = []
            if self.stepdata.game_data.last_move:
                if self.stepdata.game_data.move_flag:
                    last_x, last_y, obj_id = self.stepdata.game_data.last_move
                    if obj_id in self.stepdata.alive_operators_map.keys():
                        if self.stepdata.alive_operators_map[obj_id]["sub_type"] not in self.stepdata.unoccupy_subtype:
                            cur_y, cur_x = unpack_mapID(self.stepdata.alive_operators_map[obj_id]["cur_hex"])  # 算子当前坐标
                            for city in self.stepdata.cities:
                                occupy_y, occupy_x = unpack_mapID(city["coord"])  # 夺控点坐标
                                tar_distance = self.stepdata.grid_distant_level(last_x, last_y, occupy_x, occupy_y)
                                cur_distance = self.stepdata.grid_distant_level(cur_x, cur_y, occupy_x, occupy_y)
                                dif_distance = cur_distance - tar_distance

                                if city["value"] == 80:
                                    reward_occupy_temp = dif_distance * self.delta
                                    # print("算子ID：", obj_id, "当前位置：", cur_x, cur_y, "移动目标位置：", last_x, last_y, "主夺控点距离奖励：",
                                    #       reward_occupy_temp, )
                                else:
                                    reward_occupy_temp = dif_distance * (50 / 80) * self.delta
                                    # print("算子ID：", obj_id, "当前位置：", cur_x, cur_y, "移动目标位置：", last_x, last_y, "次夺控点距离奖励：",
                                    #       reward_occupy_temp)
                                if reward_occupy_temp > 0:
                                    reward_occupy_list_positive.append(reward_occupy_temp)
                                else:
                                    reward_occupy_list_negative.append(reward_occupy_temp)
                            if reward_occupy_list_positive != []:
                                reward_occupy_stragety = max(reward_occupy_list_positive)
                            else:
                                reward_occupy_stragety = np.mean(reward_occupy_list_negative)
                                if reward_occupy_stragety == 0:
                                    reward_occupy_stragety = -self.delta
                            print("算子ID：", obj_id, "当前位置：", cur_x, cur_y, "移动目标位置：", last_x, last_y,
                                  "移动奖励：", reward_occupy_stragety / self.max_move_reward)
                        else:
                            reward_occupy_stragety = 0
                    else:
                        reward_occupy_stragety = 0
                    self.stepdata.game_data.move_flag = False
                    return reward_occupy_stragety / self.max_move_reward
            return reward_occupy_stragety / self.max_move_reward
        else:
            return 0

    # # 集火攻击策略奖励
    # print(id_judge_info["target_obj_id"], id_judge_info["cur_step"])
    # print(reward_damage)
    # if len(self.game_data.attack_strategy) > 0:
    #     for m in range(len(self.game_data.attack_strategy)):
    #         interval_step = id_judge_info["cur_step"] - self.game_data.attack_strategy[m][1]
    #         if interval_step >= self.game_data.attack_strategy_step_limit:
    #             break
    #         if id_judge_info["target_obj_id"] == self.game_data.attack_strategy[m][0]:
    #             print(self.game_data.attack_strategy)
    #             reward_attack_stragety = reward_damage * (self.game_data.attack_strategy_step_limit -
    #                                                       interval_step) * self.game_data.attack_strategy_value
    #             print('集火损伤', reward_attack_stragety)
    #             print(self.game_data.attack_strategy[m][0])
    #         else:
    #             reward_attack_stragety = 0
    #         reward += reward_attack_stragety
    # record = [id_judge_info["target_obj_id"], id_judge_info["cur_step"]]
    # self.game_data.attack_strategy.insert(0, record)

    def last_action_reward(self):
        last_action_reward = 0
        if self.stepdata.game_data.last_action is not None:
            action, obj_id, target_obj_id, target_state = self.stepdata.game_data.last_action
            if obj_id in self.stepdata.alive_operators_map.keys():
                distance_list = self.stepdata.get_distance_list(obj_id)

                if action == 0:
                    # 判断敌方算子是否存在
                    for op in self.stepdata.my_observations['operators']:
                        if op['color'] != self.stepdata.color and self.stepdata.alive_operators_map[obj_id][
                            "sub_type"] != 0:
                            last_action_reward = 1
                            break
                        else:
                            for i in range(len(distance_list)):
                                if distance_list[i] < 10:
                                    last_action_reward = 5
                                    break
                                else:
                                    last_action_reward = -5

                if action == 3:
                    if self.stepdata.alive_operators_map[obj_id]["launcher"] is None:
                        last_action_reward = 10
                    else:
                        for i in range(len(distance_list)):
                            if distance_list[i] < 10:
                                last_action_reward = -10
                            else:
                                last_action_reward = 10
                                break
                # fixme
                if action == 4:
                    # list = self.stepdata.alive_operators_map[obj_id]['passenger_ids']
                    for i in range(len(distance_list)):
                        if distance_list[i] < 10:
                            last_action_reward = 10
                            break
                        else:
                            last_action_reward = -10

                # fixme
                # if action == 6:
                #     for op in self.stepdata.my_observations:
                #         if op['color'] != self.stepdata.color:
                #             last_action_reward = 1
                #             break
                #         else:
                #             last_action_reward = 1

                if action == 9:
                    last_action_reward = 10

                # 判断敌方算子是否存在
                if action == 10:
                    for op in self.stepdata.my_observations['operators']:
                        if op['color'] != self.stepdata.color and self.stepdata.alive_operators_map[obj_id]["sub_type"] != 0:
                            last_action_reward = 10
                            break
                        else:
                            for i in range(len(distance_list)):
                                if distance_list[i] < 6:
                                    last_action_reward = 5
                                    break
                                else:
                                    last_action_reward = -5

        self.stepdata.game_data.last_action = None
        return last_action_reward
