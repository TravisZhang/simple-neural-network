from math import *
import math
import numpy as np
import random

invalid_dist = 100000.0


def limit_num(num, max_num, min_num):
    num = max(min(max_num, num), min_num)
    return num


def lerp_with_limit(x0, t0, x1, t1, t):
    if abs(t0 - t1) < 1e-6:
        return x0
    if t < min(t0, t1):
        if t0 < t1:
            return x0
        else:
            return x1
    if t > max(t0, t1):
        if t0 < t1:
            return x1
        else:
            return x0
    r = (t - t0) / (t1 - t0)
    return x0 + r * (x1 - x0)


def norm_2d(x, y):
    return sqrt(x * x + y * y)


def normalize(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def transform(yaw, x, y):
    m = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
    v = np.array([[x], [y]])
    return np.mat(m) * np.mat(v)


def local_to_global(x0, y0, yaw, x1, y1):
    R = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
    t = np.array([[x0], [y0]])
    p0 = np.array([[x1], [y1]])
    p1 = np.mat(R) * np.mat(p0) + t
    return float(p1[0]), float(p1[1])


def global_to_local(x0, y0, yaw, x1, y1):
    Rt = np.array([[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]])
    t = np.array([[x0], [y0]])
    p0 = np.array([[x1], [y1]])
    p1 = np.mat(Rt) * (np.mat(p0) - t)
    return float(p1[0]), float(p1[1])


def point_seg_dist(x, y, x1, y1, x2, y2):
    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if d2 < 1e-6:
        return sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1)), 1.0
    # A: x,y B1: x1,y1 B2: x2,y2
    l1_norm_threshold = 10.0
    if abs(x - x1) > l1_norm_threshold:
        return invalid_dist, 0.0
    if abs(y - y1) > l1_norm_threshold:
        return invalid_dist, 0.0
    if abs(x - x2) > l1_norm_threshold:
        return invalid_dist, 0.0
    if abs(y - y2) > l1_norm_threshold:
        return invalid_dist, 0.0
    # dot(B1A, B1B2)
    dot_prod = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    # cross(B1B2, B1A)
    cross_prod = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
    sign = 1.0 if cross_prod < 0.0 else -1.0 if cross_prod > 0.0 else 0.0
    # when A is behind B1, use dist(B1A)
    r = dot_prod / d2
    if r <= 0:
        dx = x1 - x
        dy = y1 - y
        return sqrt(dx * dx + dy * dy), sign
    # when A is in front of B2, use dist(B2A)
    if r >= 1:
        dx = x2 - x
        dy = y2 - y
        return sqrt(dx * dx + dy * dy), sign
    # when A is in btwn B1 B2, use vertical dist(A, B1B2)
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    dx = px - x
    dy = py - y
    return sqrt(dx * dx + dy * dy), sign


def round_float(x):
    a = int(x * 10.0) % 10
    if a >= 5:
        return int(x) + 1
    else:
        return int(x)


def get_rand_xy(x, y, degree):
    region = len(x) / (degree + 1)
    xvals = np.zeros(degree + 1)
    yvals = np.zeros(degree + 1)
    for j in range(degree + 1):
        min_idx = round_float(float(j) * region)
        max_idx = round_float(float(j + 1) * region - 1)
        diff = max(max_idx - min_idx + 1, 1)
        # print('diff:',diff,'min_idx:',min_idx,'max_idx:',max_idx)
        random.seed()
        idx = random.randint(0, 100000000000) % diff + min_idx
        xvals[j] = x[idx]
        yvals[j] = y[idx]
    return xvals, yvals


def get_valid_pt_num(x, y, p, max_diff):
    valid_pt_num = 0
    for i in range(len(x)):
        diff = abs(p(x[i]) - y[i])
        if diff < max_diff:
            valid_pt_num += 1
        return valid_pt_num


# p ndarray, shape (deg + 1,) or (deg + 1, K)
# p = numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
def ransac(x, y, max_degree, min_degree, max_diff, rate_threshold):
    max_iter_num = 60
    min_iter_num = 30
    default_degree = 1

    degree_type_num = max_degree - min_degree + 1
    iter_num_list = np.ones(degree_type_num) * max_iter_num
    degree_set = np.ones(degree_type_num) * default_degree
    delta_iter = (max_iter_num - min_iter_num) / max(degree_type_num - 1, 1)
    early_found = False
    for i in range(degree_type_num):
        degree_set[i] = min_degree + i
        iter_num_list[i] -= i * delta_iter

    valid_pt_num_set = np.zeros(degree_type_num)
    best_coeff_set = []
    for i in range(degree_type_num):
        degree = int(degree_set[i])
        max_valid_pt_num = 0
        best_coeff = np.zeros(degree + 1)
        for m in range(int(iter_num_list[i])):
            xvals, yvals = get_rand_xy(x, y, degree)
            coef = np.polyfit(x, y, degree)
            p = np.polynomial.Polynomial(coef[::-1])
            valid_pt_num = get_valid_pt_num(x, y, p, max_diff)
            if valid_pt_num > max_valid_pt_num:
                max_valid_pt_num = valid_pt_num
                best_coeff = coef
            rate = valid_pt_num / len(x)
            if rate > rate_threshold:
                early_found = True
                break
        valid_pt_num_set[i] = max_valid_pt_num
        best_coeff_set.append(best_coeff)
        if early_found:
            break

    max_num = max(valid_pt_num_set)
    max_indices = [i for i, x in enumerate(valid_pt_num_set) if x == max_num]
    best_coeff = best_coeff_set[max_indices[0]]
    best_degree = degree_set[max_indices[0]]
    return best_coeff, best_degree


def kd_shuffle(max_num, random_len):
    indices = np.linspace(0, max_num - 1, max_num).astype(int)
    # print('indices:',indices)
    for i in range(random_len):
        random.seed()
        idx_range = max_num - i
        rand_idx = random.randint(0, 100000000000) % idx_range + i
        # swap rand_idx with i
        # because rand idx can be next to i, when it is and after swap,
        # the next indices[i] will not be equal to i
        temp_idx = indices[i]
        indices[i] = indices[rand_idx]
        indices[rand_idx] = temp_idx
    return indices[0:random_len]


def get_random_numbers(max_num, random_len):
    random_numbers = np.zeros(random_len).astype(int)
    region = max_num / random_len
    for j in range(random_len):
        min_idx = round_float(float(j) * region)
        max_idx = round_float(float(j + 1) * region - 1)
        # print('a:',float(j)*region, 'b:',float(j+1)*region - 1)
        diff = max(max_idx - min_idx + 1, 1)
        random.seed()
        if diff == 1:
            rand_num = 0 if random.random() <= 0.5 else 1
        else:
            rand_num = random.randint(0, 100000000000) % diff
        idx = rand_num + min_idx
        # print('min_idx:',min_idx,'max_idx:',max_idx,'idx:',idx)
        random_numbers[j] = idx
    return random_numbers


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    degree = 2
    xvals, yvals = get_rand_xy(x, y, degree)
    print(xvals)
    print(yvals)

    coeff, degree = ransac(x, y, 3, 1, 0.5, 0.9)
    print('coeff:', coeff, 'degree:', degree)

    timespan = lerp_with_limit(1.0, 0.0, 3.0, -2.0, -1.0)
    print('timespan:', timespan)

    random_list = kd_shuffle(20, 10)
    print('random_list:', random_list)

    random_list = get_random_numbers(20, 10)
    print('random_list:', random_list)
