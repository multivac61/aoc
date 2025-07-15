#!/usr/bin/env python3

from itertools import cycle

from aoc import answer, atom, parse_year, summary

parse = parse_year(2018)

# %%

in1 = parse(1, atom)

answer(1.1, 520, lambda: sum(in1))


def find_first_repeated_freq(frequencies):
    freq, all_freq = 0, set()
    for f in cycle(frequencies):
        freq += f
        if freq in all_freq:
            return freq
        all_freq.add(freq)


answer(1.2, 394, lambda: find_first_repeated_freq(in1))

# %% Summary
summary()
