#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import random
import time
import copy

import numpy as np
# from jax import jit
# import pandas as pd
import pprint
from tqdm import tqdm
import pandas as pd

import argparse

# 引数設定
parser = argparse.ArgumentParser()
parser.add_argument('--n', help = '次数')
parser.add_argument('-sf','--semi_ms_flg', help = '半魔方陣を求めるなら True')
args = parser.parse_args()


# n次魔方陣のランダム初期化


def initMS(n: int):
    ms_inits = []
    ms_init = []
    # M の初期化
    while len(ms_init) != n * n:
        rand_id = random.randint(1, n * n)
        if rand_id not in ms_init:
            ms_init.append(rand_id)
    ms_inits.append(ms_init)
    # Δの初期化
    # ms_inits.append([3]*(n*n)) # simple EA
    ms_inits.append([n * n] * (n * n))  # improved EA
    return ms_inits

# 各列の和と定和の差をリスト化して返す


def check_sum(ms):
    n = int(len(ms[0])**(1 / 2))
    col = []
    row = []
    dia = []
    for i in range(n):
        col.append(np.abs(sum(ms[0][i * n:i * n + n]) - magic_sum))
        row.append(np.abs(sum(ms[0][i:n * n + i:n]) - magic_sum))
    dia.append(np.abs(sum(ms[0][0:n * (n + 1):n + 1]) - magic_sum))
    dia.append(np.abs(sum(ms[0][n - 1:n * (n - 1) + 1:n - 1]) - magic_sum))
    return col, row, dia

# # 各列の和と定和の差をリスト化して返す
# # 絶対値ではないバージョン


def check_sum_without_abs(ms):
    n = int(len(ms[0])**(1 / 2))
    col = []
    row = []
    dia = []
    for i in range(n):
        col.append((sum(ms[0][i * n:i * n + n]) - magic_sum))
        row.append((sum(ms[0][i:n * n + i:n]) - magic_sum))
    dia.append((sum(ms[0][0:n * (n + 1):n + 1]) - magic_sum))
    dia.append((sum(ms[0][n - 1:n * (n - 1) + 1:n - 1]) - magic_sum))
    return col, row, dia

# 各列の和と定和の差をリスト化して返す


def check_sum_dia(ms):
    n = int(len(ms[0])**(1 / 2))
    dia = []
    dia.append(np.abs(sum(ms[0][0:n * (n + 1):n + 1]) - magic_sum))
    dia.append(np.abs(sum(ms[0][n - 1:n * (n - 1) + 1:n - 1]) - magic_sum))
    return dia

# # 各列の和と定和の差をリスト化して返す
# # 絶対値ではないバージョン


def check_sum_without_abs_dia(ms):
    n = int(len(ms[0])**(1 / 2))
    dia = []
    dia.append((sum(ms[0][0:n * (n + 1):n + 1]) - magic_sum))
    dia.append((sum(ms[0][n - 1:n * (n - 1) + 1:n - 1]) - magic_sum))
    return dia

# 定和にならない(差が0でない)の列数を計算


def count_line_row(ms):
    col, row, dia = check_sum(ms)
    return sum(x != 0 for x in row)


def count_line_col(ms):
    col, row, dia = check_sum(ms)
    return sum(x != 0 for x in col)


def count_line_dia(ms):
    col, row, dia = check_sum(ms)
    return sum(x != 0 for x in dia)

# すべての列の 定和との差 の和


def fitness_all(ms):
    col, row, dia = check_sum(ms)
    return sum(col) + sum(row) + sum(dia)

# 行・列の 定和との差 の和 (半魔方陣)


def fitness_semi(ms):
    col, row, dia = check_sum(ms)
    return sum(col) + sum(row)

# 列の 定和との差 の和


def fitness_col(ms):
    col, row, dia = check_sum(ms)
    return sum(col)

# 行の 定和との差 の和


def fitness_row(ms):
    col, row, dia = check_sum(ms)
    return sum(row)

# 斜め列の 定和との差 の和


def fitness_dia(ms):
    col, row, dia = check_sum(ms)
    return sum(dia)

# indexのマスが定和になっている列・行かの判定


def is_magic_sum(index: int):
    # index = 3
    # index を [0,n-1]のi,jに分割
    i = index // n
    j = index % n
    col, row, dia = check_sum(ms)
    if int(row[i]) != 0 and int(col[j]) != 0:
        return True
    else:
        return False

# M, Δの更新
# i = 2
# j = 1


def update_s2toall(ms, i, j, s2_list):
    ms_child = copy.deepcopy(ms)
    index = n * (i - 1) + (j - 1)

    # while True:
    rand_val = random.randint(1, ms[1][index])
    a_star_p = ms[0][index] + rand_val
    a_star_n = ms[0][index] + rand_val * (-1)
    # print(ms[0][index])
    # print((a_star_p, a_star_n))

    a_star_p = trans_astar(a_star_p)
    a_star_n = trans_astar(a_star_n)
    # print((a_star_p, a_star_n))

    s2_list_p = [np.abs(x - a_star_p) for x in s2_list]
    s2_list_n = [np.abs(x - a_star_n) for x in s2_list]

    if min(s2_list_p) > min(s2_list_n):
        s2_list_ = s2_list_n
    else:
        s2_list_ = s2_list_p

    a_kl = s2_list[np.argmin(s2_list_)]
    # a_kl = min(np.abs(a_star_p - ms[0][index]), np.abs(a_star_n - ms[0][index]))

    # if a_kl != 0:
    updated_index = ms[0].index(a_kl)

    sigma_star = ms[1][index] + random.randint(-1, 1)

    # improved_EA number1
    n_row = count_line_row(ms)
    n_col = count_line_col(ms)
    if n_row + n_col != 0:
        sigma_t = fitness_semi(ms) / (n_row + n_col)
    else:
        sigma_t = fitness_dia(ms) / count_line_dia(ms)
    sigma_t = int(sigma_t + 0.99)  # 検討要 (切り上げ)

    if sigma_star < 1 or sigma_star > sigma_t:
        ms_child[1][index] = random.randint(1, sigma_t)
    else:
        ms_child[1][index] = sigma_star

    # 変異した数に対応する数を交換 (魔方陣の連続性を保つ)
    ms_child[0][updated_index] = ms[0][index]
    ms_child[0][index] = ms[0][updated_index]

    return ms_child


def trans_astar(a_star):
    if a_star < 1:
        a_star = random.randint(1, n)
    elif a_star > n * n:
        a_star = n * n - random.randint(0, n)
    return a_star

# M, Δの更新
# update_ms1 : S2 → all (np_3 ?)
# update_ms2 : S2 → S2 (np_2 ?)
# 高速版 (np1, np2)


def update_ms_tos2(ms, i, j, s2_list):
    col, row, dia = check_sum(ms)
    ms_child = copy.deepcopy(ms)
    index = n * (i - 1) + (j - 1)
    # count = 0
    flg = False

    # 交換先が 行・列ともに定和を満たさないようにする (満たさないところまで(上限100回)繰り返し実行)
    # while True:
    # count += 1
    rand_val = random.randint(1, ms[1][index])
    a_star_p = ms[0][index] + rand_val
    a_star_n = ms[0][index] + rand_val * (-1)

    a_star_p = trans_astar(a_star_p)
    a_star_n = trans_astar(a_star_n)

    s2_list_p = [np.abs(x - a_star_p) for x in s2_list]
    s2_list_n = [np.abs(x - a_star_n) for x in s2_list]

    if min(s2_list_p) > min(s2_list_n):
        s2_list_ = s2_list_n
    else:
        s2_list_ = s2_list_p

    a_kl = s2_list[np.argmin(s2_list_)]
    # a_kl = random.choice([np.abs(a_star_p - ms[0][index]), np.abs(a_star_n - ms[0][index])])

    updated_index = ms[0].index(a_kl)
    k = updated_index // n
    l = updated_index % n
    # print(a_kl)

    # 定和判定 : 積集合
    # 変異した数に対応する数を交換 (魔方陣の連続性を保つ)
    ms_child[0][updated_index] = ms[0][index]
    ms_child[0][index] = ms[0][updated_index]

    sigma_star = ms[1][index] + random.randint(-1, 1)
    # improved_EA number1
    n_row = count_line_row(ms)
    n_col = count_line_col(ms)
    if n_row + n_col != 0:
        sigma_t = fitness_semi(ms) / (n_row + n_col)
    else:
        sigma_t = fitness_dia(ms) / count_line_dia(ms)
    sigma_t = int(sigma_t + 0.99)  # 検討要 (切り上げ)

    if sigma_star < 1 or sigma_star > sigma_t:
        ms_child[1][index] = random.randint(1, sigma_t)
        return ms_child, flg
    else:
        ms_child[1][index] = sigma_star

    flg = (ms[0][index] == a_kl)
    # print('success')
    return ms_child, flg

# 数字の入れ替え(ランダム)


def swap(ms):
    rand_id = random.randint(0, n * n - 1)
    rand_id2 = random.randint(0, n * n - 1)
    swaped_ms = copy.deepcopy(ms)
    swaped_ms[0][rand_id], swaped_ms[0][rand_id2] = swaped_ms[0][rand_id2], swaped_ms[0][rand_id]
    return swaped_ms

# 列の入れ替え(ランダム)


def swap_col(ms):
    rand_id = random.randint(0, n - 1)
    rand_id2 = random.randint(0, n - 1)
    swaped_ms = copy.deepcopy(ms)
    swaped_ms[0][rand_id *
                 n:rand_id *
                 n +
                 n], swaped_ms[0][rand_id2 *
                                  n:rand_id2 *
                                  n +
                                  n] = swaped_ms[0][rand_id2 *
                                                    n:rand_id2 *
                                                    n +
                                                    n], swaped_ms[0][rand_id *
                                                                     n:rand_id *
                                                                     n +
                                                                     n]
    return swaped_ms

# 行の入れ替え(ランダム)


def swap_row(ms):
    rand_id = random.randint(0, n - 1)
    rand_id2 = random.randint(0, n - 1)
    swaped_ms = copy.deepcopy(ms)
    swaped_ms[0][rand_id:n *
                 n +
                 rand_id:n], swaped_ms[0][rand_id2:n *
                                          n +
                                          rand_id2:n] = swaped_ms[0][rand_id2:n *
                                                                     n +
                                                                     rand_id2:n], swaped_ms[0][rand_id:n *
                                                                                               n +
                                                                                               rand_id:n]
    return swaped_ms


def ij2index(i, j):
    return n * (i - 1) + (j - 1)

# ms のうち　S2に当てはまるリストを生成(昇順にsort済み)


def make_s2(ms, col, row):
    s2_lists = []
    for i in range(n):
        if col[i] != 0:
            s2_lists += ms[0][i * n:i * n + n]
        if row[i] != 0:
            s2_lists += ms[0][i:n * n + i:n]

    return list(set(s2_lists))


# mutation位置のランダム生成
def make_child(ms):
    ms_children = []
    flgs = []
    flgs1 = []
    flgs2 = []
    flgs3 = []
    no_change_count = 0
    col, row, dia = check_sum(ms)
    s2_list = make_s2(ms, col, row)

    offspring_num = 10
    for i in range(offspring_num):
        # S1 が空集合なら np2 or np3を選択
        if sum(row) == 0 or sum(col) == 0:
            num = random.randint(1, 2)
        else:
            num = random.randint(0, 2)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        # np1 S1 → S2
        if num == 0:  # 定和判定 : 和集合 (時間かかる)
            i = row.index(int(random.choice([x for x in row if x != 0])))
            j = col.index(int(random.choice([x for x in col if x != 0])))
            ms_child, flg = update_ms_tos2(ms, i, j, s2_list)
            if flg:
                no_change_count += 1
            ms_children.append(ms_child)
            flgs.append(ms != ms_child)
            flgs1.append(ms != ms_child)
            # ms_children.append(update_ms_tos2(ms, i, j))

        # np2 S2 → S2
        elif num == 1:
            while row[i] == 0 and col[j] == 0:  # 定和判定 : 積集合
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
            ms_child, flg = update_ms_tos2(ms, i, j, s2_list)
            if flg:
                no_change_count += 1
            ms_children.append(ms_child)
            flgs.append(ms != ms_child)
            flgs2.append(ms != ms_child)
            # ms_children.append(update_ms_tos2(ms, i, j))

        # np3 S2 → all
        if num == 2:
            while row[i] == 0 and col[j] == 0:  # 定和判定 : 積集合
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)

            ms_child = update_s2toall(ms, i, j, s2_list)
            ms_children.append(ms_child)
            # flgs.append(False)
            flgs.append(ms != ms_child)
            flgs3.append(ms != ms_child)
    # print(flgs)

    # print(flgs1.count(True), flgs2.count(True),flgs3.count(True))
    # print(f'no_change_count:{no_change_count}')
    return ms_children, flgs.count(True) / offspring_num

# 行・列ごとの入れ替え


def make_child_dia(ms):
    ms_children = []
    ms_child = swap_row(ms)
    ms_child = swap_col(ms)
    ms_children.append(ms_child)

    for i in range(3):
        # 行のみ交換
        ms_child = swap_row(ms)
        ms_children.append(ms_child)
        # 列のみ交換
        ms_child = swap_col(ms)
        ms_children.append(ms_child)
        # 列・行の交換
        ms_child = swap_row(ms)
        ms_child = swap_col(ms)
        ms_children.append(ms_child)
    return ms_children

# 最良の子個体の選択(phase1)


def select_best_child(ms_children):
    fitness_list = []
    for ms_child in ms_children:
        fitness_list.append(fitness_semi(ms_child))
    best_fitness_value = min(fitness_list)  # 最良個体の fitness_value
    best_child_id = fitness_list.index(min(fitness_list))  # 最良個体のid
    best_child = ms_children[best_child_id]  # 最良個体
    return best_fitness_value, best_child

# 最良の子個体の選択


def select_best_child_dia(ms_children):
    fitness_list = []
    for ms_child in ms_children:
        fitness_list.append(fitness_all(ms_child))
        # fitness_list.append(fitness_dia(ms_child))

    best_fitness_value = min(fitness_list)  # 最良個体の fitness_value
    best_child_id = fitness_list.index(min(fitness_list))  # 最良個体のid
    best_child = ms_children[best_child_id]  # 最良個体
    return best_fitness_value, best_child


def local_recti_rc(ms):
    ms = copy.deepcopy(ms)
    col, row, dia = check_sum_without_abs(ms)
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            if k != l:
                # print(k,l)
                if col[k - 1] + col[l - 1] == 0:
                    for s in range(1, n + 1):
                        if col[k - 1] == (ms[0][ij2index(k, s)] -
                                          ms[0][ij2index(l, s)]):
                            ms[0][ij2index(k, s)], ms[0][ij2index(
                                l, s)] = ms[0][ij2index(l, s)], ms[0][ij2index(k, s)]
                            col, row, dia = check_sum_without_abs(ms)

                        for t in range(1, n + 1):
                            if s != t:
                                if col[k - 1] == ((ms[0][ij2index(k, s)] + ms[0][ij2index(k, t)]) - (
                                        ms[0][ij2index(l, s)] + ms[0][ij2index(l, t)])):
                                    ms[0][ij2index(k, s)], ms[0][ij2index(
                                        l, s)] = ms[0][ij2index(l, s)], ms[0][ij2index(k, s)]
                                    ms[0][ij2index(k, t)], ms[0][ij2index(
                                        l, t)] = ms[0][ij2index(l, t)], ms[0][ij2index(k, t)]
                                    col, row, dia = check_sum_without_abs(ms)

                if row[k - 1] + row[l - 1] == 0:
                    for s in range(1, n + 1):
                        if row[k - 1] == (ms[0][ij2index(s, k)] -
                                          ms[0][ij2index(s, l)]):
                            ms[0][ij2index(s, k)], ms[0][ij2index(
                                s, l)] = ms[0][ij2index(s, l)], ms[0][ij2index(s, k)]
                            col, row, dia = check_sum_without_abs(ms)

                        for t in range(1, n + 1):
                            if s != t:
                                if row[k - 1] == ((ms[0][ij2index(s, k)] + ms[0][ij2index(t, k)]) - (
                                        ms[0][ij2index(s, l)] + ms[0][ij2index(t, l)])):
                                    ms[0][ij2index(s, k)], ms[0][ij2index(
                                        s, l)] = ms[0][ij2index(s, l)], ms[0][ij2index(s, k)]
                                    ms[0][ij2index(t, k)], ms[0][ij2index(
                                        t, l)] = ms[0][ij2index(t, l)], ms[0][ij2index(t, k)]
                                    col, row, dia = check_sum_without_abs(ms)
    return ms


def local_recti_dia(ms):
    ms = copy.deepcopy(ms)
    dia = check_sum_without_abs_dia(ms)
    for i in range(1, n + 1):
        # error 改修済
        # ValueError: attempt to assign sequence of size 3 to extended slice of size 2 (スライスがおかしい？)
        # 条件5
        if (((ms[0][ij2index(i, i)] + ms[0][ij2index((n - i + 1), (n - i + 1))]) - (ms[0][ij2index(i, (n - i + 1))] + ms[0][ij2index((n - i + 1), i)])) == dia[0]) and \
                (((ms[0][ij2index(i, i)] + ms[0][ij2index((n - i + 1), (n - i + 1))]) - (ms[0][ij2index(i, (n - i + 1))] + ms[0][ij2index((n - i + 1), i)])) == dia[1] * (-1)):
            i_ = (n - i + 1) % n
            ms[0][i_:n * n + i_:n], ms[0][(i %
                                           n):n * n + (i %
                                                       n):n] = ms[0][(i %
                                                                      n):n * n + (i %
                                                                                  n):n], ms[0][i_:n * n + i_:n]
            dia = check_sum_dia(ms)

        for j in range(1, n + 1):
            if i != j:
                # 条件1
                if (ms[0][ij2index(i, i)] + ms[0][ij2index(i, j)] == ms[0][ij2index(j, i)] + ms[0][ij2index(j, j)]) and \
                        (((ms[0][ij2index(i, i)] + ms[0][ij2index(j, j)]) - (ms[0][ij2index(i, i)] + ms[0][ij2index(j, i)])) == dia[0]):
                    ms[0][ij2index(j, i)], ms[0][ij2index(
                        i, i)] = ms[0][ij2index(i, i)], ms[0][ij2index(j, i)]
                    ms[0][ij2index(j, j)], ms[0][ij2index(
                        i, j)] = ms[0][ij2index(i, j)], ms[0][ij2index(j, j)]
                    dia = check_sum_dia(ms)

    # error 改修要
    # ValueError: attempt to assign sequence of size 3 to extended slice of size 2 (スライスがおかしい？)
                # 条件2
                if (ms[0][ij2index(i, j)] + ms[0][ij2index(i, n - i + 1)] == ms[0][ij2index(n - j + 1, j)] + ms[0][ij2index(n - j + 1, n - i + 1)]) and \
                        (((ms[0][ij2index(i, n - i + 1)] + ms[0][ij2index(n - j + 1, j)]) - (ms[0][ij2index(n - j + 1, n - i + 1)] + ms[0][ij2index(i, j)])) == dia[1]):
                    ms[0][ij2index(i, j)], ms[0][ij2index(
                        n - j + 1, j)] = ms[0][ij2index(n - j + 1, j)], ms[0][ij2index(i, j)]
                    ms[0][ij2index(i, n -
                                   i +
                                   1)], ms[0][ij2index(n -
                                                       j +
                                                       1, n -
                                                       i +
                                                       1)] = ms[0][ij2index(n -
                                                                            j +
                                                                            1, n -
                                                                            i +
                                                                            1)], ms[0][ij2index(i, n -
                                                                                                i +
                                                                                                1)]
                    dia = check_sum_dia(ms)

    # error 改修要
    # ValueError: attempt to assign sequence of size 3 to extended slice of size 2 (スライスがおかしい？)
                # 条件3
                if (((ms[0][ij2index(i, i)] + ms[0][ij2index(j, j)]) - (ms[0][ij2index(i, j)] + ms[0][ij2index(j, i)])) == dia[0]) and \
                        (((ms[0][ij2index(i, n - i + 1)] + ms[0][ij2index(j, n - j + 1)]) - (ms[0][ij2index(i, n - j + 1)] + ms[0][ij2index(j, n - i + 1)])) == dia[1]):
                    ms[0][(j %
                           n):n * n + j:n], ms[0][(i %
                                                   n):n * n + i:n] = ms[0][(i %
                                                                            n):n * n + i:n], ms[0][(j %
                                                                                                    n):n * n + j:n]
                    dia = check_sum_dia(ms)

                # 条件4
                if (((ms[0][ij2index(i, i)] + ms[0][ij2index(j, j)]) - (ms[0][ij2index(i, j)] + ms[0][ij2index(j, i)])) == dia[0]) and \
                        (((ms[0][ij2index(n - i + 1, i)] + ms[0][ij2index(n - j + 1, j)]) - (ms[0][ij2index(n - j + 1, i)] + ms[0][ij2index(n - i + 1, j)])) == dia[1]):
                    ms[0][(j %
                           n):n * n + j:n], ms[0][(i %
                                                   n):n * n + i:n] = ms[0][(i %
                                                                            n):n * n + i:n], ms[0][(j %
                                                                                                    n):n * n + j:n]
                    # ms[0][j*n:j*n+n], ms[0][i*n:i*n+n] = ms[0][i*n:i*n+n], ms[0][j*n:j*n+n]
                    dia = check_sum_dia(ms)

    # print(np.reshape(ms[0],(n,n)))
    return ms

# 学習過程の メイン


def main_loop(ms):
    flg_rate = 0
    n_row = count_line_row(ms)
    n_col = count_line_col(ms)
    if n_row + n_col > 0:
        ms_children, flg_rate = make_child(ms)
        # ms_children = make_child(ms)
        best_fitness_value, best_child = select_best_child(ms_children)
        fitness_ms = fitness_semi(ms)
        if best_fitness_value < 50 * n:
            # Local rectification of row and columns
            ms = local_recti_rc(ms)
            # 今の親個体よりも finess_valueが小さいときは 更新
            if fitness_ms >= best_fitness_value:
                ms = copy.deepcopy(best_child)

        else:
            ms = copy.deepcopy(best_child)
    else:
        ms_children = make_child_dia(ms)  # ここの子の生成方法は変わるので変更 要
        best_fitness_value, best_child = select_best_child_dia(ms_children)
        fitness_ms = fitness_dia(ms)
        if best_fitness_value < 10000:
            # Local rectification of diagonals
            ms = local_recti_dia(ms)
            if fitness_ms >= best_fitness_value:
                ms = copy.deepcopy(best_child)

        else:
            ms = copy.deepcopy(best_child)
    return ms, ms_children, flg_rate
    # return ms, ms_children


# 長期実行用
# 10個のMS を得るまで
n = args.n
magic_sum = n * (n * n + 1) / 2
semi_ms_flg = False
semi_ms_flg = args.semi_ms_flg


def main():
    df_log = pd.read_csv('./log.csv')
    df_semi_log = pd.read_csv('./log.csv')
    _ms = []

    ms_list = []
    ms_count = []
    semi_ms_list = []
    semi_ms_count = []
    min_semi = 1000000000

    count = 0
    semi_count = 0
    for num in tqdm(range(1000000)):
        flg_count = 0
        dup_count = 0  # dup: Duplicate (重複)解のカウント
        flg = False
        
        # 魔方陣を求める場合
        if semi_ms_flg == False:
            if len(ms_count) < 10:
                ms = initMS(n)
                # for i in tqdm(range(1000000)):
                for i in (range(100000)):
                    ms, ms_children, flg_rate = main_loop(ms)
                    # print(flg_rate)
                    # ms, ms_children = main_loop(ms)
                    # pprint.pprint(ms)
                    # if i%10000==0:
                    #   print(fitness_semi(ms), fitness_all(ms))
                    #   if i%1000==0:
                    #     print(np.reshape(ms[0],(n,n)))
                    if fitness_all(ms) == 0:
                        print(str(i) + "step")
                        ms_count.append(i)
                        print(np.reshape(ms[0], (n, n)))
                        ms_list.append(ms[0])
                        count += 1
                        df_log = df_log.append({'ms': ms, 'generation': i},
                                            ignore_index=True)
                        df_log.to_csv(f'./log_ms{n}.csv')
                        break
                    elif fitness_semi(ms) == 0 and flg == False:
                        print(str(i) + "step")
                        semi_ms_count.append(i)
                        print(np.reshape(ms[0], (n, n)))
                        semi_ms_list.append(ms[0])
                        semi_count += 1
                        flg = True

                    # Early Stopping (100回以上 同じ解をとったらストップ)
                    if ms == _ms:
                        # dup_count += (1 - flg_rate)
                        dup_count += 1
                        if dup_count >= 100:
                            break
                    flg_count += flg_rate
                    _ms = copy.deepcopy(ms)

        elif semi_ms_flg == True:
            if semi_count < 10:
                ms = initMS(n)
                for i in tqdm(range(1000000)):
                    ms, ms_children, flg_rate = main_loop(ms)
                    # print(flg_rate)
                    # ms, ms_children = main_loop(ms)
                    # pprint.pprint(ms)
                    # if i%10000==0:
                    #   print(fitness_semi(ms), fitness_all(ms))
                    #   if i%1000==0:
                    #     print(np.reshape(ms[0],(n,n)))
                    if fitness_all(ms) == 0:
                        print(str(i) + "step")
                        ms_count.append(i)
                        print(np.reshape(ms[0], (n, n)))
                        ms_list.append(ms[0])
                        count += 1
                        df_log = df_log.append({'ms': ms, 'generation': i},
                                            ignore_index=True)
                        df_log.to_csv(f'./log_ms{n}.csv')
                        break
                    elif fitness_semi(ms) == 0 and flg == False:
                        print(str(i) + "step")
                        semi_ms_count.append(i)
                        print(np.reshape(ms[0], (n, n)))
                        semi_ms_list.append(ms[0])
                        semi_count += 1
                        flg = True
                        if semi_ms_flg == True:
                            df_semi_log = df_semi_log.append({'ms': ms, 'generation': i},
                                                ignore_index=True)
                            df_semi_log.to_csv(f'./log_semi_ms{n}.csv')

                    # Early Stopping (100回以上 同じ解をとったらストップ)
                    if ms == _ms:
                        # dup_count += (1 - flg_rate)
                        dup_count += 1
                        if dup_count >= 100:
                            break
                    flg_count += flg_rate
                    _ms = copy.deepcopy(ms)
            print(
                fitness_semi(ms),
                fitness_all(ms),
                count,
                semi_count,
                i,
                round(
                    flg_count,
                    1))
            # if min_semi > fitness_semi(ms):
            #   min_semi =  fitness_semi(ms)
            # print(f'min : {min_semi}')
            # print(f"row:{n - count_line_row(ms)}, col:{n - count_line_col(ms)}")
        else:
            print(str(num) + " Loop")
            break


if __name__ == "__main__":
    main()
