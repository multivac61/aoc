#!/usr/bin/env python3

from aoc import answer, digits, mapt, parse_year, the

parse = parse_year(2017)

# %%
in1 = mapt(int, the(parse(1, digits)))
N = len(in1)

answer(1.1, 1097, lambda: sum(in1[i] for i in range(N) if in1[i] == in1[i - 1]))
answer(1.1, 1188, lambda: sum(in1[i] for i in range(N) if in1[i] == in1[i - N // 2]))
