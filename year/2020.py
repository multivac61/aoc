#!/usr/bin/env python3

from aoc import answer, parse_year, first, combinations, prod
import re

parse = parse_year(2020)


# %%
in1 = parse(1, int)


def find_sum_2020(expenses, repeat=2):
    return first(nums for nums in combinations(expenses, repeat) if sum(nums) == 2020)


assert find_sum_2020([1721, 979, 366, 299, 675, 1456]) == (1721, 299)
answer(1.1, 567171, lambda: prod(find_sum_2020(in1)))

answer(1.2, 212428694, lambda: prod(find_sum_2020(in1, repeat=3)))


# %%


def parse_password_line(line):
    match = re.match(r"(\d+)-(\d+)\s+([a-z]):\s+([a-z]+)", line)

    if not match:
        raise ValueError(f"Invalid line format: {line}")

    min_count, max_count, letter, password = match.groups()
    return int(min_count), int(max_count), letter, password


def validate_passwords(min_count, max_count, letter, password):
    return min_count <= password.count(letter) <= max_count


in2 = parse(2, parse_password_line)

assert validate_passwords(*parse_password_line("1-3 a: abcde")) is True
assert validate_passwords(*parse_password_line("1-3 b: cdefg")) is False
assert validate_passwords(*parse_password_line("2-9 c: ccccccccc")) is True

answer(2.1, 456, lambda: sum(1 for data in in2 if validate_passwords(*data)))


def validate_passwords2(min_count, max_count, letter, password):
    return (password[min_count - 1] == letter) != (password[max_count - 1] == letter)


assert validate_passwords2(*parse_password_line("1-3 a: abcde")) is True
assert validate_passwords2(*parse_password_line("1-3 b: cdefg")) is False
assert validate_passwords2(*parse_password_line("2-9 c: ccccccccc")) is False


answer(2.2, 308, lambda: sum(1 for data in in2 if validate_passwords2(*data)))

# %%
