import math
import numpy as np


class Trans:
    @staticmethod
    def llh_to_ecef(lat, lon, alt) -> {}:
        # WGS84椭球参数
        a = 6378137.0  # 地球半径
        e = 8.1819190842622e-2  # 偏心率

        # 将度转换为弧度
        # lat = math.radians(lat)
        # lon = math.radians(lon)

        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)

        X = (N + alt) * math.cos(lat) * math.cos(lon)
        Y = (N + alt) * math.cos(lat) * math.sin(lon)
        Z = ((1 - e ** 2) * N + alt) * math.sin(lat)

        return {'X': X, 'Y': Y, 'Z': Z}

    @staticmethod
    def body_to_ecef(yaw, pitch, roll, vector_body) -> {}:
        # 将角度从度转换为弧度
        # yaw = np.radians(yaw)
        # pitch = np.radians(pitch)
        # roll = np.radians(roll)
        vector_body = np.array([vector_body['X'], vector_body['Y'], vector_body['Z']])
        # 偏航旋转矩阵
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 俯仰旋转矩阵
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 翻滚旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转
        R = np.dot(R_z, np.dot(R_y, R_x))

        # 将机体系向量转换为ECEF系
        vector_ecef = np.dot(R, vector_body)

        return {'X': vector_ecef[0], 'Y': vector_ecef[1], 'Z': vector_ecef[2]}

    @staticmethod
    def body_to_ecef_mat(yaw, pitch, roll):
        # 偏航旋转矩阵
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 俯仰旋转矩阵
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 翻滚旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转
        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    @staticmethod
    def mat_mutil_vec(mat, vec):
        vector_body = np.array([vec['X'], vec['Y'], vec['Z']])

        # 将机体系向量转换为ECEF系
        vector_ecef = np.dot(mat, vector_body)

        return {'X': vector_ecef[0], 'Y': vector_ecef[1], 'Z': vector_ecef[2]}

    @staticmethod
    def ecef_to_body(yaw, pitch, roll, vector_ecef) -> {}:
        # 将角度从度转换为弧度
        # yaw = np.radians(yaw)
        # pitch = np.radians(pitch)
        # roll = np.radians(roll)
        vector_ecef = np.array([vector_ecef['X'], vector_ecef['Y'], vector_ecef['Z']])
        # 偏航旋转矩阵（逆）
        R_z_inv = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 俯仰旋转矩阵（逆）
        R_y_inv = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 翻滚旋转矩阵（逆）
        R_x_inv = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转（注意旋转顺序相反）
        R_inv = np.dot(R_x_inv, np.dot(R_y_inv, R_z_inv))

        # 将ECEF系向量转换为机体系
        vector_body = np.dot(R_inv, vector_ecef)

        return {'X': vector_body[0], 'Y': vector_body[1], 'Z': vector_body[2]}

    @staticmethod
    def ecef_to_lla(x, y, z):
        # WGS-84椭球体常数
        a = 6378137  # 赤道半径
        e = 8.1819190842622e-2  # 第一偏心率

        # 计算经度
        lon = math.atan2(y, x)

        # 迭代计算纬度和高度
        p = math.sqrt(x ** 2 + y ** 2)
        lat = math.atan2(z, p * (1 - e ** 2))
        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - N

        # 转换为度
        # lat = math.degrees(lat)
        # lon = math.degrees(lon)

        return lat, lon, alt

    @staticmethod
    def cal_four_pos():
        lat = 15
        lon = 115
        # 将度转换为弧度
        lat = math.radians(lat)
        lon = math.radians(lon)
        self_pos_ecef = Trans.llh_to_ecef(lat, lon, 8000)
        body_to_ecef_mat = Trans.body_to_ecef_mat(0, 0, 0)

        dis_ = 150000
        left = [0, -dis_, 0]
        vector_body = np.array(left)
        ecef_dir = np.dot(body_to_ecef_mat, vector_body)
        for i, key in enumerate(['X', 'Y', 'Z']):
            ecef_dir[i] += self_pos_ecef[key]
        new_pos_lla = Trans.ecef_to_lla(ecef_dir[0], ecef_dir[1], ecef_dir[2])
        # print('左', math.degrees(new_pos_lla[0]), math.degrees(new_pos_lla[1]))

        down = [-dis_, 0, 0]
        vector_body = np.array(down)
        ecef_dir = np.dot(body_to_ecef_mat, vector_body)
        for i, key in enumerate(['X', 'Y', 'Z']):
            ecef_dir[i] += self_pos_ecef[key]
        new_pos_lla = Trans.ecef_to_lla(ecef_dir[0], ecef_dir[1], ecef_dir[2])
        # print('下', math.degrees(new_pos_lla[0]), math.degrees(new_pos_lla[1]))

        right = [0, dis_, 0]
        vector_body = np.array(right)
        ecef_dir = np.dot(body_to_ecef_mat, vector_body)
        for i, key in enumerate(['X', 'Y', 'Z']):
            ecef_dir[i] += self_pos_ecef[key]
        new_pos_lla = Trans.ecef_to_lla(ecef_dir[0], ecef_dir[1], ecef_dir[2])
        # print('右', math.degrees(new_pos_lla[0]), math.degrees(new_pos_lla[1]))

        up = [dis_, 0, 0]
        vector_body = np.array(up)
        ecef_dir = np.dot(body_to_ecef_mat, vector_body)
        for i, key in enumerate(['X', 'Y', 'Z']):
            ecef_dir[i] += self_pos_ecef[key]
        new_pos_lla = Trans.ecef_to_lla(ecef_dir[0], ecef_dir[1], ecef_dir[2])
        # print('上', math.degrees(new_pos_lla[0]), math.degrees(new_pos_lla[1]))


distance = 400


class CalNineDir:
    @staticmethod
    def determine_quadrant(angle):
        # 标准化角度到 [-pi, pi] 范围
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        # 特殊位置：坐标轴
        if angle == 0:
            return 0, 1  # 正 y 轴
        elif angle == math.pi or angle == -math.pi:
            return 0, -1  # 负 y 轴
        elif angle == math.pi / 2:
            return 1, 0  # 正 x 轴
        elif angle == -math.pi / 2:
            return -1, 0  # 负 x 轴

        # 根据角度所在的范围返回 x 和 y 的符号
        if 0 < angle < math.pi / 2:
            return 1, 1  # 第一象限
        elif math.pi / 2 < angle < math.pi:
            return 1, -1  # 第四象限
        elif -math.pi / 2 < angle < 0:
            return -1, 1  # 第二象限
        elif -math.pi < angle < -math.pi / 2:
            return -1, -1  # 第三象限

    @staticmethod
    def get_forward_dir(yaw, cur_x, cur_y, cur_z) -> {}:
        x, y = CalNineDir.determine_quadrant(yaw)
        cur_x = cur_x + distance * x * abs(math.sin(yaw))
        cur_y = cur_y + distance * y * abs(math.cos(yaw))
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_backward_dir(yaw, cur_x, cur_y, cur_z) -> {}:
        yaw = -yaw
        x, y = CalNineDir.determine_quadrant(yaw)
        cur_x = cur_x + distance * x * abs(math.sin(yaw))
        cur_y = cur_y + distance * y * abs(math.cos(yaw))
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_left_dir(yaw, cur_x, cur_y, cur_z) -> {}:
        yaw -= math.pi / 2
        if yaw < -math.pi:
            yaw += math.pi * 2
        x, y = CalNineDir.determine_quadrant(yaw)
        cur_x = cur_x + distance * x * abs(math.sin(yaw))
        cur_y = cur_y + distance * y * abs(math.cos(yaw))
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_right_dir(yaw, cur_x, cur_y, cur_z) -> {}:
        yaw += math.pi / 2
        if yaw > math.pi:
            yaw -= math.pi * 2
        x, y = CalNineDir.determine_quadrant(yaw)
        cur_x = cur_x + distance * x * abs(math.sin(yaw))
        cur_y = cur_y + distance * y * abs(math.cos(yaw))
        # print(x, y)
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_lef_up_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        ret = CalNineDir.get_left_dir(yaw, cur_x, cur_y, cur_z)
        return {'X': ret['X'], 'Y': ret['Y'], 'Z': cur_z + distance}

    @staticmethod
    def get_lef_down_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        ret = CalNineDir.get_left_dir(yaw, cur_x, cur_y, cur_z)
        return {'X': cur_x + ret['X'], 'Y': cur_y + ret['Y'], 'Z': cur_z - distance}

    @staticmethod
    def get_right_up_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        ret = CalNineDir.get_right_dir(yaw, cur_x, cur_y, cur_z)
        return {'X': cur_x + ret['X'], 'Y': cur_y + ret['Y'], 'Z': cur_z + distance}

    @staticmethod
    def get_right_down_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        ret = CalNineDir.get_right_dir(yaw, cur_x, cur_y, cur_z)
        return {'X': cur_x + ret['X'], 'Y': cur_y + ret['Y'], 'Z': cur_z - distance}

    @staticmethod
    def get_up_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        cur_z = cur_z + distance
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_down_pos(yaw, cur_x, cur_y, cur_z) -> {}:
        cur_z = cur_z - distance
        return {'X': cur_x, 'Y': cur_y, 'Z': cur_z}

    @staticmethod
    def get_all_nine_dir(yaw, cur_x, cur_y, cur_z):
        # print(f'current{yaw},{cur_x},{cur_y},{cur_z}')
        ans = {}
        for k, func in all_nine_body_dir.items():
            ans[k] = func(yaw, cur_x, cur_y, cur_z)
        return ans


all_nine_body_dir = {
    '0': CalNineDir.get_forward_dir,
    '1': CalNineDir.get_lef_up_pos,
    '2': CalNineDir.get_right_up_pos,
    '3': CalNineDir.get_lef_down_pos,
    '4': CalNineDir.get_right_down_pos,
    '5': CalNineDir.get_up_pos,
    '6': CalNineDir.get_down_pos,
    '7': CalNineDir.get_left_dir,
    '8': CalNineDir.get_right_dir,
}
