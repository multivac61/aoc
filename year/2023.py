#!/usr/bin/env python3

from aoc import answer, parse_year, summary

parse = parse_year(2023)


# %%
in1 = parse(1)


def get_calibration_value(line):
    digits = [char for char in line if char.isdigit()]
    first, last = digits[0], digits[-1]
    return int(first + last)


assert get_calibration_value("1abc2") == 12
assert get_calibration_value("pqr3stu8vwx") == 38
assert get_calibration_value("a1b2c3d4e5f") == 15
assert get_calibration_value("treb7uchet") == 77

answer(1.1, 56042, lambda: sum(map(get_calibration_value, in1)))


def get_calibration_value2(line):
    number_map = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    digits = []
    for i in range(len(line)):
        if line[i].isdigit():
            digits.append(line[i])
        for word, digit in number_map.items():
            if line[i:].startswith(word):
                digits.append(digit)
    return int(digits[0] + digits[-1])


assert get_calibration_value2("two1nine") == 29
assert get_calibration_value2("eightwothree") == 83
assert get_calibration_value2("abcone2threexyz") == 13
assert get_calibration_value2("xtwone3four") == 24
assert get_calibration_value2("4nineeightseven2") == 42
assert get_calibration_value2("zoneight234") == 14
assert get_calibration_value2("7pqrstsixteen") == 76

answer(1.2, 55358, lambda: sum(map(get_calibration_value2, in1)))

# %% Summary
summary()
