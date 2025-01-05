#!/usr/bin/env python3

from aoc import answer, parse_year
from itertools import pairwise, islice

parse = parse_year(current_year=2021)


# %%
in1 = parse(1, int)


def count_depth_increases(depths):
    return sum(1 for a, b in pairwise(depths) if b > a)


test1 = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]
assert count_depth_increases(test1) == 7

answer(1.1, 1139, lambda: count_depth_increases(in1))


def count_window_increases(depths, window_size=3):
    return count_depth_increases(
        sum(window)
        for window in zip(*(islice(depths, i, None) for i in range(window_size)))
    )


assert count_window_increases(test1, window_size=1) == 7
assert count_window_increases(test1) == 5

answer(1.2, 1103, lambda: count_window_increases(in1))

# %%
