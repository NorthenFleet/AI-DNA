scenario_info = [
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
             "tank_pos_red": [5038, 5037], "automissile_pos_red": [5038], "missile_pos_red": [5038], "trops_red": [5136],
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
        ]

for sc in scenario_info:
    if 2010211129 == sc["scenario"]:
        scenario_info_id = index
        print(scenario_info_id)