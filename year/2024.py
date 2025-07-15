#!/usr/bin/env python3

from aoc import answer, ints, parse_year, flatten, summary
from itertools import combinations
import re

parse = parse_year(2024)


# %%

in1 = parse(1, ints)
left, right = zip(*in1)

answer(
    1.1, 1879048, lambda: sum(abs(a - b) for a, b in zip(sorted(left), sorted(right)))
)


def calculate_similarity_score(left, right):
    right_counts = {}
    for num in right:
        right_counts[num] = right_counts.get(num, 0) + 1
    return sum(num * right_counts.get(num, 0) for num in left)


answer(1.2, 21024792, lambda: calculate_similarity_score(left, right))

# %%


def is_safe(xs):
    return all(
        1 <= abs(b - a) <= 3 and (b > a) == (xs[1] > xs[0]) for a, b in zip(xs, xs[1:])
    )


assert is_safe([7, 6, 4, 2, 1]) is True
assert is_safe([1, 2, 7, 8, 9]) is False
assert is_safe([9, 7, 6, 2, 1]) is False
assert is_safe([1, 3, 2, 4, 5]) is False
assert is_safe([8, 6, 4, 4, 1]) is False
assert is_safe([1, 3, 6, 7, 9]) is True

in2 = parse(2, ints)
answer(2.1, 224, lambda: sum(1 for x in in2 if is_safe(x)))


def is_safe2(xs):
    return any(is_safe(ys) for ys in combinations(xs, len(xs) - 1))


answer(2.2, 293, lambda: sum(1 for x in in2 if is_safe2(x)))


# %%


def mults(text: str):
    """A tuple of all the integers in text, ignoring non-number characters."""
    pattern = r"mul\((\d+),(\d+)\)"
    return [(int(x), int(y)) for x, y in re.findall(pattern, text)]


in3 = flatten(parse(3, mults))
answer(3.1, 187194524, lambda: sum(x * y for x, y in in3))


def operations(text):
    def parse_match(m):
        if m.group().startswith("mul"):
            return ("mul", int(m.group(1)), int(m.group(2)))
        elif m.group() == "do()":
            return ("do",)
        else:
            return ("dont",)

    pattern = r"mul\((\d+),(\d+)\)|do\(\)|don't\(\)"
    return [parse_match(m) for m in re.finditer(pattern, text)]


def add_enabled(operations):
    mults, should_mult = [], True
    for operation in operations:
        if operation[0] == "dont":
            should_mult = False
        elif operation[0] == "do":
            should_mult = True
        elif should_mult:
            mults.append((operation[1], operation[2]))
    return mults


in3 = flatten(parse(3, operations))
answer(3.2, 127092535, lambda: sum(x * y for x, y in add_enabled(in3)))


# %%

in4 = parse(4, tuple)


def count_word_occurrences(grid, word):
    rows = len(grid)
    cols = len(grid[0])
    word_len = len(word)

    # Define all 8 directions: right, left, down, up, diagonal-right-down,
    # diagonal-left-up, diagonal-right-up, diagonal-left-down
    directions = [
        (0, 1),  # right
        (0, -1),  # left
        (1, 0),  # down
        (-1, 0),  # up
        (1, 1),  # diagonal right down
        (-1, -1),  # diagonal left up
        (-1, 1),  # diagonal right up
        (1, -1),  # diagonal left down
    ]

    # Function to check if a word exists starting from position (row, col) in direction (dx, dy)
    def search_direction(row, col, dx, dy):
        # First check if the word would fit in this direction
        if (
            0 <= row + (word_len - 1) * dx < rows
            and 0 <= col + (word_len - 1) * dy < cols
        ):
            # Check each character of the word
            for i in range(word_len):
                if grid[row + i * dx][col + i * dy] != word[i]:
                    return False
            return True
        return False

    # Search through each position in the grid
    return sum(
        1
        for row in range(rows)
        for col in range(cols)
        for dx, dy in directions
        if search_direction(row, col, dx, dy)
    )


test41 = """..X...
.SAMX.
.A..A.
XMAS.S
.X...."""

assert count_word_occurrences(parse(test41, tuple), "XMAS") == 4

test42 = """MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX"""

assert count_word_occurrences(parse(test42, tuple), "XMAS") == 18

answer(4.1, 2358, lambda: count_word_occurrences(in4, "XMAS"))


def count_word_occurrences2(grid, word):
    rows, cols, word_len = len(grid), len(grid[0]), len(word)

    # Function to check if a word exists starting from position (row, col) in direction (dx, dy)
    def search_direction(row, col, dx, dy):
        if not (
            0 <= row < rows
            and 0 <= col < cols
            and 0 <= row + (word_len - 1) * dx < rows
            and 0 <= col + (word_len - 1) * dy < cols
        ):
            return False

        return all(grid[row + i * dx][col + i * dy] == word[i] for i in range(word_len))

    return sum(
        1
        for row in range(rows)
        for col in range(cols)
        if (
            search_direction(row, col, 1, 1)
            and search_direction(row + word_len - 1, col, -1, 1)
        )
        or (
            search_direction(row + word_len - 1, col, -1, 1)
            and search_direction(row + word_len - 1, col + word_len - 1, -1, -1)
        )
        or (
            search_direction(row, col + word_len - 1, 1, -1)
            and search_direction(row + word_len - 1, col + word_len - 1, -1, -1)
        )
        or (
            search_direction(row, col, 1, 1)
            and search_direction(row, col + word_len - 1, 1, -1)
        )
    )


assert count_word_occurrences2(parse(test42, tuple), "MAS") == 9
answer(4.2, 1737, lambda: count_word_occurrences2(in4, "MAS"))

# %% Summary
summary()
