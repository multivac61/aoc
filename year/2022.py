#!/usr/bin/env python3

from aoc import answer, ints, paragraphs, parse_year, summary

parse = parse_year(current_year=2022)


# %%
in1 = parse(1, ints, paragraphs)

answer(1.1, 66487, lambda: max(sum(calories) for calories in in1))

answer(1.2, 197301, lambda: sum(sorted(sum(calories) for calories in in1)[-3:]))

# %% Summary
summary()
