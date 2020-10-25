from collections import defaultdict
from ai_company.data_process.data_process_util import *
from ai_company.data_process.strategy import Stragety
from ai_company.data_process.reward import Reward
import random

def extract_members(obj, exclude=[]):
    import inspect
    members = []
    for name in dir(obj):
        if name.startswith('__') and name.endswith('__'):
            continue
        attr = getattr(obj, name)
        if not inspect.ismethod(attr) and name not in exclude:
            members.append((name, attr))
    return members

RED = 0
BLUE = 1

class GameData:
    def __init__(self, step_num):
        self.step_num = step_num
        # self.reward_occupy = 0
        self.last_attack = 5
        self.attack_strategy_step_limit = 10
        self.attack_strategy_value = 0.2  # 1.2
        self.occupy_subtype = [0, 1]  # 选择参与夺控的算子类型，0表示坦克
        self.rule_period_first = 400
        self.rule_period_second = 1500

    def reset(self, reset_data):
        self.initialize(reset_data)

    def initialize(self, init_data):
        self.color = init_data['color']
        self.enemy_color = BLUE if self.color == RED else RED

        self.scenario = init_data['scenario']
        self.map_size = init_data['map_size']

        self.initial_all_operators = init_data['passengers'] + init_data['operators']

        self.initial_my_operator_ids = set(
            [op['obj_id'] for op in init_data['passengers'] + init_data['operators'] if op['color'] == self.color])
        self.initial_my_passengers_ids = set(
            [op['obj_id'] for op in init_data['passengers'] if op['color'] == self.color])
        self.initial_enemy_operator_ids = set(
            [op['obj_id'] for op in init_data['passengers'] + init_data['operators'] if
             op['color'] == self.enemy_color])
        self.initial_enemy_passengers_ids = set(
            [op['obj_id'] for op in init_data['passengers'] if op['color'] == self.enemy_color])

        self.thread_score = {}
        self.cur_thread = None
        self.first_reward = True
        self.basic_reward = 0
        self.occupy_reward = 0
        self.reward_win = None
        self.enemy_blood = None
        self.last_move = None
        self.move_flag = False
        self.last_action = None
        self.reward_remain = None
        self.reward_attack = 0
        self.attack_strategy = []

    def get_step_data(self, step_data):
        my_observations = step_data['my_observations']
        enemy_observations = step_data['enemy_observations']
        return StepData(self, my_observations, enemy_observations, False)

    def __str__(self):
        lines = []
        lines.append(f"class: {self.__class__.__name__}")
        lines.append(f"members:")
        members = extract_members(self, exclude=['initial_all_operators'])
        for name, attr in members:
            lines.append(f'- self.{name} = {attr}')
        return '\n'.join(lines)


NO_ACTION = 0
MOVE = 1
SHOOT = 2
GET_ON = 3
GET_OFF = 4
OCCUPY = 5
CHANGE_STATE = 6
REMOVE_KEEP = 7
JM_SHOOT = 8
GUIDE_SHOOT = 9
STOP_MOVE = 10
WEAPON_LOCK = 11
WEAPON_UNLOCK = 12
CANCEL_JM_PLAN = 13

MAX_OP = 34
MAX_VA = MAX_OP * 4 + 6 + 4
MAX_VA_MATRIX_LEN = MAX_VA * MAX_OP


class StepData:
    def __str__(self):
        lines = []
        lines.append(f"class: {self.__class__.__name__}")
        lines.append(f"members:")
        members = extract_members(self, ['game_data', 'alive_operators'])
        for name, attr in members:
            lines.append(f'- self.{name} = {attr}')
        return '\n'.join(lines)

    def __init__(self, game_data: GameData, my_observations, enemy_observations, game_over):
        self.game_data = game_data

        self.game_over = game_over
        self.color = game_data.color

        self.my_observations = my_observations
        self.enemy_observations = enemy_observations

        self.unoccupy_subtype = [7]  # 不能够夺控的算子类型

        self.stragety = Stragety(self)
        self.reward = Reward(self)

        # 以下代码是为了解决XZ交流群里提到的bug：
        # ”车辆算子被消灭后乘客算子仍然在passengers里面“
        alive_operator_ids_not_in_car = set()
        passengers = []

        # 返回的算子列表包含能够观测到的敌方我方所有算子！！！
        my_operators = [op for op in my_observations['operators'] if op['color'] == self.color]
        enemy_operators = [op for op in enemy_observations['operators'] if op['color'] != self.color]
        my_passengers = [op for op in my_observations['passengers'] if op['color'] == self.color]
        enemy_passengers = [op for op in enemy_observations['passengers'] if op['color'] != self.color]

        for op in my_operators + enemy_operators:
            alive_operator_ids_not_in_car.add(op['obj_id'])
        for p in my_passengers + enemy_passengers:
            if p['car'] != None and p['car'] in alive_operator_ids_not_in_car:
                passengers.append(p)

        # 当前step所有存活的算子，list
        ops = my_operators + enemy_operators + passengers

        # 剔除零血算子
        self.alive_operators = [op for op in ops if op['blood'] > 0]

        # 算子ID与算子数据对应，dict
        self.alive_operators_map = {}
        for op in self.alive_operators:
            self.alive_operators_map[op['obj_id']] = op

        self.alive_operators = list(self.alive_operators_map.values())

        self.alive_my_operator_ids = set()
        self.alive_enemy_operator_ids = set()

        self.alive_my_tank_ids = set()
        self.alive_my_car_ids = set()
        self.alive_my_autocar_ids = set()
        self.alive_my_autoflight_ids = set()
        self.alive_my_flight_ids = set()

        self.alive_my_missile_ids = set()
        self.alive_enemy_tank_ids = set()
        self.alive_enemy_car_ids = set()
        self.alive_enemy_autocar_ids = set()
        self.alive_enemy_autoflight_ids = set()
        self.alive_enemy_flight_ids = set()
        self.alive_enemy_missile_ids = set()

        for op in self.alive_operators:
            if op['color'] == self.color:
                self.alive_my_operator_ids.add(op['obj_id'])
                if op["sub_type"] == 0:
                    self.alive_my_tank_ids.add(op['obj_id'])
                elif op["sub_type"] == 1:
                    self.alive_my_car_ids.add(op['obj_id'])
                elif op["sub_type"] == 4:
                    self.alive_my_autocar_ids.add(op['obj_id'])
                elif op["sub_type"] == 5:
                    self.alive_my_autoflight_ids.add(op['obj_id'])
                elif op["sub_type"] == 6:
                    self.alive_my_flight_ids.add(op['obj_id'])
                elif op["sub_type"] == 7:
                    self.alive_my_missile_ids.add(op['obj_id'])
            else:
                self.alive_enemy_operator_ids.add(op['obj_id'])
                if op["sub_type"] == 0:
                    self.alive_enemy_tank_ids.add(op['obj_id'])
                elif op["sub_type"] == 1:
                    self.alive_enemy_car_ids.add(op['obj_id'])
                elif op["sub_type"] == 4:
                    self.alive_enemy_autocar_ids.add(op['obj_id'])
                elif op["sub_type"] == 5:
                    self.alive_enemy_autoflight_ids.add(op['obj_id'])
                elif op["sub_type"] == 6:
                    self.alive_enemy_flight_ids.add(op['obj_id'])
                elif op["sub_type"] == 7:
                    self.alive_enemy_missile_ids.add(op['obj_id'])


        # 当前step所有活着的、在车上的算子ID
        # BUG: passenger字段指向不确定，暂不使用
        # self.alive_passengers_id = set([op['obj_id'] for op in my_observations['passengers'] + enemy_observations['passengers']])

        # 所有我方算子ID，包含被摧毁的的算子
        self.all_my_operator_ids = game_data.initial_my_operator_ids

        # 所有敌方算子ID，包含被摧毁的的算子
        self.all_enemy_operator_ids = game_data.initial_enemy_operator_ids

        self.all_ids = list(self.all_my_operator_ids) + list(self.all_enemy_operator_ids)  # 确保id顺序

        self.cities = my_observations['cities']
        self.jm_points = my_observations["jm_points"]
        self.time = my_observations["time"]
        self.cur_step = my_observations["time"]["cur_step"]
        self.judge_info = my_observations["judge_info"]
        self.jm_points = my_observations["jm_points"]
        self.valid_actions = my_observations["valid_actions"]
        self.scores = my_observations["scores"]

    def is_game_over(self):
        return self.game_over

    def get_valid_action_interface(self):
        return ValidActionInterface(self)

    def choose_valid_action(self, thread_id, action_type, my_id, target_id=None, target_state=None, move=None,
                            x=None, y=None, x_prob=None, y_prob=None, move_type=None):
        self.game_data.cur_thread = thread_id
        if action_type in [0, 15] :
            return {}
        action_type = int(action_type)
        my_id = int(my_id)
        if target_id:
            target_id = int(target_id)
        if target_state:
            target_state = int(target_state)
        if x:
            x = int(x)
        if y:
            y = int(y)
        obj_id = my_id
        target_obj_id = target_id
        operator = self.alive_operators_map[my_id]
        if action_type == 1:
            if self.stragety.is_stragety(obj_id, operator, self.cur_step, move, move_type, x_prob, y_prob,):
                pass
            else:
                # target_pos = int(y * 100 + x)
                self.stragety.multi_step_move(move)
            target_pos = self.stragety.get_pos()
            y, x = unpack_mapID(target_pos)
            self.game_data.move_flag = True
            self.game_data.last_move = (x, y, obj_id)
            if operator["type"] == 2:
                if operator["move_state"] == 1:
                    mod = 1  # 车辆行军
                else:
                    mod = 0  # 车辆机动
            elif operator["type"] == 3:
                mod = 3  # 飞机机动
            else:
                mod = 2  # 人员机动
            return {"obj_id": obj_id, "target_pos": target_pos, "coord": operator["cur_hex"], "type": 1, "mod": mod}

        if action_type == 2:
            res = {"obj_id": obj_id, "type": 2}
            attack_actions = self.valid_actions[str(obj_id)]['2']
            attack_actions.sort(key=lambda ele: ele["attack_level"], reverse=True)
            for i in attack_actions:
                if i["target_obj_id"] == target_obj_id:
                    res.update(i)
                    break
            return res

        if action_type == 3:
            self.game_data.last_action = (3, obj_id, target_obj_id, target_state)
            return {"obj_id": obj_id, "target_obj_id": target_obj_id, "type": 3}

        if action_type == 4:
            self.game_data.last_action = (4, obj_id, target_obj_id, target_state)
            return {"obj_id": obj_id, "target_obj_id": target_obj_id, "type": 4}

        if action_type == 5:
            return {"obj_id": obj_id, "type": 5}

        if action_type == 6:
            self.game_data.last_action = (6, obj_id, target_obj_id, target_state)
            return {"obj_id": obj_id, "target_state": target_state, "type": 6}

        if action_type == 7:
            return {"obj_id": obj_id, "type": 7}

        if action_type == 8:
            target_pos = y * 100 + x

            jm_actions = self.valid_actions[str(obj_id)]['8']
            wp_id = random.choice(jm_actions)['weapon_id']
            return {"obj_id": obj_id, "type": 8, "jm_pos": target_pos, 'weapon_id': wp_id}

        if action_type == 9:
            res = {"obj_id": obj_id, "type": 9}
            attack_actions = self.valid_actions[str(obj_id)]['9']
            attack_actions.sort(key=lambda ele: ele["attack_level"], reverse=True)
            for i in attack_actions:
                if i["target_obj_id"] == target_obj_id:
                    res.update(i)
                    break
            return res

        if action_type == 10:
            self.game_data.last_action = (10, obj_id, target_obj_id, target_state)
            return {"obj_id": obj_id, "type": 10}

        if action_type == 11:
            return {"obj_id": obj_id, "type": 11}

        if action_type == 12:
            return {"obj_id": obj_id, "type": 12}

        if action_type == 13:
            return {"obj_id": obj_id, "type": 13}

        if action_type == 14:
            return {"obj_id": obj_id, "type": 14}

    def get_reward(self):
        return self.reward.get_total_reward()

    # 返回算子距离夺控点的距离列表
    def get_distance_list(self, obj_id):
        distance_list = []
        cur_y, cur_x = unpack_mapID(self.alive_operators_map[obj_id]["cur_hex"])  # 算子当前坐标
        for city in self.cities:
            occupy_y, occupy_x = unpack_mapID(city["coord"])  # 夺控点坐标
            cur_distance = self.grid_distant_level(cur_x, cur_y, occupy_x, occupy_y)
            distance_list.append(cur_distance)
        return distance_list

    def colrow_to_xyz(self, col, row):
        x = col
        z = row - (col - (col & 1)) / 2
        y = -x - z
        return np.array([x, y, z])

    def grid_distant_level(self, x1, y1, x2, y2):
        xyz1 = self.colrow_to_xyz(x1, y1)
        xyz2 = self.colrow_to_xyz(x2, y2)
        num = int(np.sum(np.abs(xyz1 - xyz2)) / 2)
        return num


ACTION_TYPE_NUM = 10 + 3
class ValidActionInterface:
    def __init__(self, step_data: StepData):
        self.step_data = step_data
        self.map_size_x = step_data.game_data.map_size[0]
        self.map_size_y = step_data.game_data.map_size[1]

    def get_type_vector(self):
        ops_id = self.step_data.valid_actions.keys()
        valid_type_set = set()
        for i in ops_id:
            valid_type_set.update(self.step_data.valid_actions[i].keys())
        result = [1]
        for i in range(1, ACTION_TYPE_NUM + 1):
            result.append(1 if str(i) in valid_type_set else 0)
        # result[10] = self.step_data.stragety.is_stop(result[10])
        result[11] = self.step_data.stragety.is_stop(result[11])
        result[12] = self.step_data.stragety.is_stop(result[12])
        result[13] = self.step_data.stragety.is_stop(result[13])
        if sum(result) > 1:
            result[0] = 0
        return result

    def get_operator_vector_of_specific_type(self, type_id):
        result = []
        for i in self.step_data.all_my_operator_ids:
            if str(i) in self.step_data.valid_actions.keys() and str(type_id) in self.step_data.valid_actions[str(i)].keys():
                result.append(1)
            else:
                result.append(0)
        result += [0] * (MAX_OP - len(self.step_data.all_my_operator_ids))
        return result

    def get_valid_my_operator_id_vector(self):
        result = []
        vmid = self.step_data.valid_actions.keys()
        for oid in list(self.step_data.all_my_operator_ids):
            result.append(1 if str(oid) in vmid else 0)
        result += [0] * (MAX_OP - len(self.step_data.all_my_operator_ids))
        return result

    def query_my_id(self, index):

        return list(self.step_data.all_my_operator_ids)[index]

    def query_enemy_id(self, index):
        return list(self.step_data.all_enemy_operator_ids)[index]

    def query_state(self):
        return 3

    # 长度为14，第一位为空动作
    def get_type_vector_by_selected_operator_id(self, my_id):
        if self.step_data.alive_operators_map[my_id]["sub_type"] not in [0, 7]:
            result = [1]  # 第一个元素表示不执行动作,放入记忆库
        else:
            result = [0]
        if str(my_id) not in self.step_data.valid_actions:
            va_type = [0]
        else:
            va_type = self.step_data.valid_actions[str(my_id)].keys()
        for va in range(1, ACTION_TYPE_NUM + 1):
            if str(va) in va_type:
                result.append(1)
            else:
                result.append(0)
        if self.step_data.alive_operators_map[my_id]["sub_type"] == 0:
            result[10] = 0
        else:
            if self.step_data.cur_step < self.step_data.game_data.rule_period_first:
                result[10] = 0
            elif self.step_data.cur_step > self.step_data.game_data.rule_period_second and sum(self.step_data.alive_my_tank_ids) == 0:
                result[10] = 0

        # if self.get_on_flage is True and action_type == 3:
        #     action_type = 0
        #     result['action_type'] = action_type
        if self.step_data.alive_operators_map[my_id]["sub_type"] == 1:
            distance_list = self.step_data.get_distance_list(my_id)
            for i in range(len(distance_list)):
                if distance_list[i] < 12:
                    break
                else:
                    result[4] = 0

        result[11] = self.step_data.stragety.is_stop(result[11])
        result[12] = self.step_data.stragety.is_stop(result[12])
        result[13] = self.step_data.stragety.is_stop(result[13])
        return result

    def query_action_type(self, index):
        return index

    # fixme
    def get_param_by_selected_action_type(self, action_type, my_id):
        def extract_ids_by_key(target_list, key="target_obj_id"):
            ids = set()
            for t in target_list:
                ids.add(t[key])
            return ids

        result = []
        target_list = self.step_data.valid_actions[str(my_id)][str(action_type)]
        if action_type in [2, 3, 4, 9]:
            target_ids = extract_ids_by_key(target_list)
            if action_type == 2 or action_type == 9:
                ids = self.step_data.all_enemy_operator_ids
                tp = 'enemy_ops'
            else:
                ids = self.step_data.all_my_operator_ids
                tp = 'my_ops'
            for oid in ids:
                result.append(1 if oid in target_ids else 0)
            result += [0] * (MAX_OP - len(ids))
            return {"type": tp, "value": result}
        if action_type == 6:
            target_states = extract_ids_by_key(target_list, key="target_state")
            for i in [0, 1, 2, 3, 4]:
                result.append(1 if i in target_states else 0)
            result += [0] * (MAX_OP - 5)
            return {"type": "state", "value": result}

        if action_type in [11, 12, 13]:
            pass

        return None