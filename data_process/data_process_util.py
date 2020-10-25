import numpy as np


WEAPON_ID_MAP = { 29: 0, 35: 1, 36: 2, 37: 3, 43: 4, 54: 5, 56: 6, 69: 7, 71: 8, 72: 9, 73: 10, 74: 11, 75: 12, 
                  76: 13, 83: 14, 84: 15, 88: 16, 89: 17}

# 辅助函数，生成武器one hot编码（共18种武器，所以数组长度18）
def _weapon_one_hot(weapon_ids):
    _one_hot = [0]*18
    for wid in weapon_ids:
        _one_hot[WEAPON_ID_MAP[wid]] = 1
    return _one_hot


def _one_hot(value, type_range):
    arr = []
    for i in type_range:
        if value == i:
            arr.append(1)
        else:
            arr.append(0)
    return arr

def unpack_mapID(mapID):
    mapID = int(mapID)
    row = mapID // 100
    col = mapID % 100
    return (row, col)

#==================================================================
#======================= 以下是使用的算子字段 ========================
#==================================================================
ID_FIELD = [
    "obj_id",  # 算子ID
    "launcher",  # 所属发射器ID（其实是所属载具，这个值对步兵、无人战车和巡飞弹有效，0代表没有所属载具）
    "car"  # 目前没啥用，永远为0
]
INT_AND_FLOAT = [
    "basic_speed",  # 基础速度
    "value",  # 分值
    "cur_pos",  # 当前格到下一个格的百分比
    "speed",  # 当前算子速度
    "move_to_stop_remain_time",  # 停止机动剩余时间 
    "blood",  # 算子当前生命值
    "max_blood",  # 算子最大生命值
    "tire_accumulate_time",  # 疲劳累计时间
    "keep_remain_time",  # 压制状态剩余时间 
    "get_on_remain_time",  # 上车剩余时间
    "get_off_remain_time",  # 下车剩余时间
    "change_state_remain_time",  # 切换状态剩余时间
    "weapon_cool_time",  # 武器剩余冷却时间
    "weapon_unfold_time",  # 武器剩余展开时间
    "alive_remain_time"  # 当前算子存活时间，这个字段专门针对发射出去的巡飞弹，其他算子这个值为0
]
CLASSIFICATION_FIELD = {
    "color": [0, 1],  # 算子所属阵营
    "type": [1, 2, 3],  # 算子大类型，1-步兵 2-车辆 3-飞机
    "sub_type": [0, 1, 2, 3, 4, 5, 6, 7],  # 算子细分类型，坦克 0/  战车1 / 人员2 / 炮兵3 / 无人战车4 / 无人机5 / 直升机6 / 巡飞弹7
    "armor": [0, 1, 2, 3, 4],  # 算子装甲类型 0-无装甲 1-轻型装甲 2-中型装甲 3-重型装甲 4-复合装甲
    "A1": [0, 1],  # 算子是否有行进间射击能力
    "stack": [0, 1],  # 算子是否在当前格堆叠
    "can_to_move": [0, 1],  #是否可机动标志位.只在停止转换过程中用来判断是否可以继续机动.强制停止不能继续机动,正常停止可以继续机动. 0-否 1-是
    "flag_force_stop": [0, 1],  # 是否被强制停止机动 0-否 1-是
    "guide_ability": [0, 1],  # 算子是否有引导其他算子射击的能力
    "move_state": [0, 1, 2, 3, 4],  # 算子机动状态 0-正常机动 1-行军 2-一级冲锋 3-二级冲锋 4-掩蔽
    "stop": [0, 1],  # 算子是否静止
    "tire": [0, 1, 2],  # 算子疲劳等级 0-不疲劳 1-一级疲劳 2-二级疲劳
    "keep": [0, 1],  # 算子是否被压制
    "on_board": [0, 1],  # 算子是否在车上
    "lose_control": [0, 1],  # 算子是否失去控制
    "target_state": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 切换动作的目标状态，0代表当前没有在切换状态，'1-机动', '2-射击', '3-上车', '4-下车', '5-夺控', '6-切换状态', '7-移除压制', '8-间瞄', '9-引导射击'
}
MAX_VOLUMN = 4  # 车辆最大容量

def cvt_operator(op=None):
    # op为None是生成全零数组，表示空
    result = []

    row, col = unpack_mapID(op["cur_hex"]) if op else (0, 0)
    result += [row, col]
    
    # 生成数值类型数组
    for i in INT_AND_FLOAT:
        result.append(op[i] if op else 0)
    
    # 生成由ID组成的数组
    for i in ID_FIELD:
        if op == None:
            result.append(0)
        elif op[i] == None:
            result.append(0)
        else:
            result.append(op[i])
    
    # 生成类别类型数组（one hot）
    for field, type_range in CLASSIFICATION_FIELD.items():
        result += _one_hot(op[field], type_range) if op else [0] * len(type_range)

    max_passenger_nums = []
    if op:
        for st in [0, 1, 2, 3, 4, 5, 6, 7]:
            if st in op["max_passenger_nums"].keys():  # 算子可装载的乘客类型与数量
                max_passenger_nums.append(op["max_passenger_nums"][st])
            else:
                max_passenger_nums.append(0)
        result += max_passenger_nums
    else:
        result += [0] * 8
    
    result += [op['observe_distance']['1'], op['observe_distance']['2'], op['observe_distance']['3']] if op else [0] * 3  # 算子对步兵、车辆和飞机的观测距离
    result += op["passenger_ids"] + (MAX_VOLUMN-len(op["passenger_ids"]))*[0] if op else MAX_VOLUMN * [0]  # 车上乘客算子的ID
    result += op["launch_ids"] + (MAX_VOLUMN-len(op["launch_ids"]))*[0] if op else MAX_VOLUMN * [0]  # 车上可发射算子的ID

    if op:
        if len(op["get_on_partner_id"]) != 0:  # 这个字段代表现在正在执行上车动作的算子列表，我直接转换为当前载具是否有人在上车
            result += _one_hot(1, [0, 1])
        else:
            result += _one_hot(0, [0, 1])

        if len(op["get_off_partner_id"]) != 0:  # 正在执行下车动作的算子列表
            result += _one_hot(1, [0, 1])
        else:
            result += _one_hot(0, [0, 1])
    else:
        result += [0] * 4

    # 剩余弹药数
    for k in [0, 100, 101, 102]:
        result.append(op['remain_bullet_nums'].get(k, 0))

    # 把算子包含的武器进行one hot编码
    result += _weapon_one_hot(op["carry_weapon_ids"]) if op else [0] * 18
    return result


# 输入单个城市结构体，输出一维数组
def cvt_city(city: dict):
    if city:
        row, col = unpack_mapID(city["coord"])  # 夺控点坐标
        return [row, col] + [city["value"]] + _one_hot(city["flag"], [0, 1])
    else:
        return [0] * 5


# 输入单个间瞄点数据，输出一维数组
# 间瞄点就是火炮火力覆盖点
def cvt_jm_point(jm_point: dict):
    result = []
    row, col = unpack_mapID(jm_point['pos']) if jm_point else (0, 0)  # 间瞄点位置
    result += [row, col]
    result.append(jm_point['obj_id'] if jm_point else 0)  # 该间瞄点是由哪个算子产生的
    result.append(jm_point['fly_time'] if jm_point else 0)  # 炮弹剩余飞行时间
    result.append(jm_point['boom_time'] if jm_point else 0)  # 炮火轰炸还有多长时间结束
    result += _one_hot(jm_point["state"], [0, 1, 2])  if jm_point else [0] * 3 # 间瞄点状态 0-正在飞行 1-正在爆炸 2-无效
    result += _weapon_one_hot([jm_point["weapon_id"]]) if jm_point else [0] * 18  # 该间瞄点是用什么武器打的
    return result


# 输入时间数据，输出一维数组
def cvt_time(time):
    if time == None:
        return [0, 0]
    return [time["cur_step"], time["tick"]]


def get_operator_positon_map(map_size, operators):
    arr = np.zeros(map_size)
    for op in operators:
        row, col = unpack_mapID(op['cur_pos'])
        arr[row][col] = 1
    return arr


def get_city_position_map(map_size, cities):
    arr = np.zeros(map_size)
    for city in cities:
        row, col = unpack_mapID(city['coord'])
        arr[row][col] = 2
    return arr


def extra_all_ids(operators):
    ids = set()
    for op in operators:
        ids.add(op["obj_id"])
        for passenger_id in op["passenger_ids"]:
            ids.add(passenger_id)
    return ids


def get_passenger_dict(operators):
    passenger_dict = {}
    for op in operators:
        if len(op["passenger_ids"]) > 0:
            if passenger_dict.get(op["obj_id"], None) is None:
                passenger_dict[op["obj_id"]] = set()
            passenger_dict[op["obj_id"]].update(op["passenger_ids"])
    return passenger_dict

