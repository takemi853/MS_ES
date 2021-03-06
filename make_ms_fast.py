#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import random
import time
import copy

import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm

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


# M, Δの更新
def update_ms(ms, i, j):
    ms_child = copy.deepcopy(ms)
    index = n * (i - 1) + (j - 1)

    a_star = ms[0][index] + \
        random.randint(1, ms[1][index]) * (-1)**random.randint(0, 1)
    if a_star < 1:
        ms_child[0][index] = random.randint(1, n)
    elif a_star > n * n:
        ms_child[0][index] = n * n - random.randint(0, n)
    else:
        ms_child[0][index] = a_star

    sigma_star = ms[1][index] + random.randint(-1, 1)

    # improved_EA number1
    n_row = count_line_row(ms)
    n_col = count_line_col(ms)
    if n_row + n_col != 0:
        sigma_t = (fitness_row(ms) + fitness_col(ms)) / (n_row + n_col)
    else:
        sigma_t = fitness_dia(ms) / count_line_dia(ms)
    sigma_t = int(sigma_t + 0.99)  # 検討要 (切り上げ)

    if sigma_star < 1 or sigma_star > sigma_t:
        ms_child[1][index] = random.randint(1, sigma_t)
    else:
        ms_child[1][index] = sigma_star

    # 変異した数に対応する数を交換 (魔方陣の連続性を保つ)
    updated_index = ms[0].index(ms_child[0][index])
    ms_child[0][updated_index] = ms[0][index]

    return ms_child


# 高速版
def update_ms_s2tos2(ms, i, j):
    # 交換先が 行・列ともに定和を満たさないようにする (満たさないところまで(上限100回)繰り返し実行)
    col, row, dia = check_sum(ms)
    k = random.randint(0, n - 1)
    l = random.randint(0, n - 1)
    while (row[k] == 0 and col[l] == 0) and (i != k and j != l):  # 定和判定 : 積集合
        k = random.randint(0, n - 1)
        l = random.randint(0, n - 1)

    ms_child = copy.deepcopy(ms)
    index = n * (i - 1) + (j - 1)
    updated_index = n * (k - 1) + (l - 1)

    # 変異した数に対応する数を交換 (魔方陣の連続性を保つ)
    ms_child[0][updated_index] = ms[0][index]
    ms_child[0][index] = ms[0][updated_index]

    return ms_child

# mutation位置のランダム生成


def make_child(ms):
    ms_children = []
    for num in range(10):
        col, row, dia = check_sum(ms)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if num % 3 == 0:
            while row[i] == 0 and col[j] == 0:  # 定和判定 : 積集合
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
            ms_children.append(update_ms(ms, i, j))
        elif num % 3 == 1:
            while row[i] == 0 and col[j] == 0:  # 定和判定 : 積集合
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
            ms_children.append(update_ms_s2tos2(ms, i, j))
        elif num % 3 == 2:  # 定和判定 : 和集合 (時間かかる)
            if sum(row) != 0:
                i = row.index(int(random.choice([x for x in row if x != 0])))
            if sum(col) != 0:
                j = col.index(int(random.choice([x for x in col if x != 0])))
            ms_children.append(update_ms_s2tos2(ms, i, j))
    return ms_children

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


def ij2index(i, j):
    return n * (i - 1) + (j - 1)

# 学習過程の メイン


def main_loop(ms):
    n_row = count_line_row(ms)
    n_col = count_line_col(ms)
    if n_row + n_col > 0:
        ms_children = make_child(ms)
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
        if best_fitness_value < 100:
            # Local rectification of diagonals
            ms = local_recti_dia(ms)
            if fitness_ms >= best_fitness_value:
                ms = copy.deepcopy(best_child)

        else:
            ms = copy.deepcopy(best_child)
    return ms, ms_children


# 長期実行用
# 10個のMS を得るまで


def main():
    # n = 10
    # magic_sum = n * (n * n + 1) / 2
    df_log = pd.read_csv('./log.csv')
    _ms = []

    ms_list = []
    ms_count = []
    semi_ms_list = []
    semi_ms_count = []
    min_semi = 1000000000

    count = 0
    semi_count = 0
    for num in tqdm(range(1000000)):
        dup_count = 0  # dup: Duplicate (重複)解のカウント
        flg = False
        if len(ms_count) < 1000:
            # if semi_count<10:
            ms = initMS(n)
            # for i in tqdm(range(1000000)):
            for i in (range(100000)):
                ms, ms_children = main_loop(ms)
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
                    df_log.to_csv('./log.csv')
                    break
                elif fitness_semi(ms) == 0 and flg == False:
                    print(str(i) + "step")
                    semi_ms_count.append(i)
                    print(np.reshape(ms[0], (n, n)))
                    semi_ms_list.append(ms[0])
                    semi_count += 1
                    flg = True
                    print(fitness_semi(ms), fitness_all(ms), count, semi_count)

                # Early Stopping (100回以上 同じ解をとったらストップ)
                if ms == _ms:
                    dup_count += 1
                    if dup_count >= 100:
                        break

                _ms = copy.deepcopy(ms)
            # print(fitness_semi(ms), fitness_all(ms), count, semi_count)
            # if min_semi > fitness_semi(ms):
            #   min_semi =  fitness_semi(ms)
            # print(f'min : {min_semi}')
            # print(f"row:{n - count_line_row(ms)}, col:{n - count_line_col(ms)}")
            # print()
        else:
            print(str(num) + " Loop")
            break


n = 5
magic_sum = n * (n * n + 1) / 2

if __name__ == "__main__":
    main()
