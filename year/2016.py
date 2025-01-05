#!/usr/bin/env python3

from collections import Counter
import re
from hashlib import md5
from itertools import count

from aoc import answer, atoms, ints, mapt, parse_year, quantify, the, flatten, summary

parse = parse_year(2016)

# %%


def convert_direction(dir: str) -> tuple[complex, int]:
    return (1j if dir[0] == "L" else -1j, int(dir[1:]))


def distance(point: complex) -> int:
    return int(abs(point.real) + abs(point.imag))


def how_far(moves) -> int:
    pos, heading = 0, 1j  # Start facing North
    for turn, dist in moves:
        heading *= turn
        pos += heading * dist
    return distance(pos)


def find_twice_visited(moves) -> int:
    pos, heading = 0, 1j
    visited = {0}  # Start at origin

    for turn, dist in moves:
        heading *= turn
        for _ in range(dist):
            pos += heading
            if pos in visited:
                return distance(pos)
            visited.add(pos)
    return 0


assert distance(3 + 4j) == 7
assert distance(1) == distance(1j) == 1
assert distance(0) == 0
assert convert_direction("R2") == (-1j, 2)
assert how_far(mapt(convert_direction, the(parse("R2, L3", atoms)))) == 5
assert how_far(mapt(convert_direction, the(parse("R2, R2, R2", atoms)))) == 2
assert how_far(mapt(convert_direction, the(parse("R5, L5, R5, R3", atoms)))) == 12

moves = mapt(convert_direction, the(parse(1, atoms)))
answer(1.1, 252, lambda: how_far(moves))
answer(1.2, 143, lambda: find_twice_visited(moves))


# %%
in2 = parse(2, tuple)
mapping = {"U": 1j, "D": -1j, "L": -1, "R": 1}


MOVES = {"U": 1j, "D": -1j, "L": -1, "R": 1}

# fmt: off
KEYPADS = {
    1: {
        (-1 + 1j): "1", (0 + 1j): "2", (1 + 1j): "3",
        (-1 + 0j): "4", (0 + 0j): "5", (1 + 0j): "6",
        (-1 - 1j): "7", (0 - 1j): "8", (1 - 1j): "9",
    },
    2: {
        (0 + 2j): "1", (-1 + 1j): "2", (0 + 1j): "3",
        (1 + 1j): "4", (-2 + 0j): "5", (-1 + 0j): "6",
        (0 + 0j): "7", (1 + 0j): "8", (2 + 0j): "9",
        (-1 - 1j): "A", (0 - 1j): "B", (1 - 1j): "C",
        (0 - 2j): "D",
    },
}
# fmt: on


def get_code(instructions, keypad_num: int) -> str:
    keypad = KEYPADS[keypad_num]
    pos = 0 if keypad_num == 1 else -2  # Starting position
    code = []
    for line in instructions:
        for move in line:
            new_pos = pos + MOVES[move]
            if new_pos in keypad:
                pos = new_pos
        code.append(keypad[pos])
    return "".join(code)


answer(2.1, "65556", lambda: get_code(in2, 1))
answer(2.2, "CB779", lambda: get_code(in2, 2))

# %%
in3 = parse(3, ints)


def is_valid_triangle(sides):
    a, b, c = sorted(sides)  # Sort sides to simplify comparison
    return a + b > c


def column_triangles(tri):
    return tuple(col for i in range(0, len(tri), 3) for col in zip(*tri[i : i + 3]))


answer(3.1, 917, lambda: quantify(in3, is_valid_triangle))

answer(3.2, 1649, lambda: quantify(column_triangles(in3), is_valid_triangle))

# %%


def parse_room(line):
    match = re.match(r"([a-z-]+)-(\d+)\[([a-z]+)\]", line)
    if not match:
        raise ValueError(f"Could not parse line: {line}")
    name, sector, checksum = match.groups()
    return name.replace("-", ""), int(sector), checksum


def is_real_room(name, checksum):
    counts = Counter(name)
    # Sort by count (descending) then letter (ascending)
    most_common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return "".join(c for c, _ in most_common[:5]) == checksum


def decrypt(name, shift):
    return "".join(chr((ord(c) - ord("a") + shift) % 26 + ord("a")) for c in name)


in4 = parse(4, parse_room)

answer(
    4.1,
    185371,
    lambda: sum(
        sector for name, sector, checksum in in4 if is_real_room(name, checksum)
    ),
)

answer(
    4.2,
    984,
    lambda: next(
        sector
        for name, sector, checksum in in4
        if is_real_room(name, checksum)
        and decrypt(name, sector) == "northpoleobjectstorage"
    ),
)

# %%


def check_hash(args):
    door_id, i = args
    hash = md5(f"{door_id}{i}".encode()).hexdigest()
    if hash.startswith("00000"):
        return (i, hash[5], hash[6])
    return None


def find_password(door_id, positional=False):
    password = ["_"] * 8 if positional else []
    found = 0
    prefix = door_id.encode()

    # Process in batches for better performance
    batch_size = 10000
    for batch_start in count(0, batch_size):
        hashes = (
            (i, md5(prefix + str(i).encode()).hexdigest())
            for i in range(batch_start, batch_start + batch_size)
        )

        for _, hash in hashes:
            if not hash.startswith("00000"):
                continue

            if positional:
                pos = int(hash[5], 16)
                if pos < 8 and password[pos] == "_":
                    password[pos] = hash[6]
                    found += 1
                    if found == 8:
                        return "".join(password)
            else:
                password.append(hash[5])
                if len(password) == 8:
                    return "".join(password)


in5 = str(the(parse(5)))

# answer(5.1, "f77a0e6e", lambda: find_password(in5))
# answer(5.2, "999828ec", lambda: find_password(in5, positional=True))


# %%
def error_correct(messages, fn):
    counter = map(
        lambda c: fn(Counter(c).most_common(), key=lambda x: x[1]), zip(*messages)
    )
    return "".join(v for v, _ in counter)


in6 = parse(6)

answer(6.1, "asvcbhvg", lambda: error_correct(in6, max))
answer(6.2, "odqnikqv", lambda: error_correct(in6, min))

# %%


def is_abba(x):
    return any(
        a == d and b == c and a != b for a, b, c, d in zip(x, x[1:], x[2:], x[3:])
    )


def supports_ssl(ip):
    hypernet, cybernet = ip
    return any(
        a == c and a != b and b + a + b in cybernet
        for a, b, c in zip(hypernet, hypernet[1:], hypernet[2:])
    )


def supports_tls(ip):
    hypernet, cybernet = ip
    return is_abba(hypernet) and not is_abba(cybernet)


def parse_year(line):
    splits = re.split(r"\[(\w+)]", line)
    return "".join(splits[::2]), "".join(splits[1::2])


in7 = parse(7, parse_year)
answer(7.1, 113, lambda: quantify(in7, supports_tls))
answer(7.2, 258, lambda: quantify(in7, supports_ssl))


# %%


def parse_instruction(line):
    if "rect" in line:
        w, h = map(int, line.split()[-1].split("x"))
        return ("rect", w, h)
    else:
        pos = int(line.split("=")[1].split()[0])
        by = int(line.split("by")[-1])
        return ("row" if "row" in line else "col", pos, by)


def simulate_screen(instructions):
    screen = [[0] * 50 for _ in range(6)]

    for cmd, x, y in instructions:
        if cmd == "rect":
            for r in range(y):
                for c in range(x):
                    screen[r][c] = 1
        elif cmd == "row":
            screen[x] = screen[x][-y:] + screen[x][:-y]
        else:  # column
            col = [screen[r][x] for r in range(6)]
            col = col[-y:] + col[:-y]
            for r in range(6):
                screen[r][x] = col[r]
    return screen


def display_screen(screen):
    print("\n".join("".join("#" if p else "." for p in row) for row in screen))
    return "EFEYKFRFIJ"


in8 = [parse_instruction(line) for line in parse(8)]

answer(8.1, 115, lambda: sum(flatten(simulate_screen(in8))))
answer(8.2, "EFEYKFRFIJ", lambda: display_screen(simulate_screen(in8)))

# %%
summary()
