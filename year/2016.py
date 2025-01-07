#!/usr/bin/env python3

import re
from collections import Counter, defaultdict, deque
from functools import lru_cache
from hashlib import md5
from itertools import count

from aoc import (
    answer,
    atoms,
    first,
    flatten,
    ints,
    mapt,
    parse_year,
    quantify,
    the,
)

parse = parse_year(2016)


# %% Day 1
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


# %% Day 2
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

# %% Day 3
in3 = parse(3, ints)


def is_valid_triangle(sides):
    a, b, c = sorted(sides)  # Sort sides to simplify comparison
    return a + b > c


def column_triangles(tri):
    return tuple(col for i in range(0, len(tri), 3) for col in zip(*tri[i : i + 3]))


answer(3.1, 917, lambda: quantify(in3, is_valid_triangle))

answer(3.2, 1649, lambda: quantify(column_triangles(in3), is_valid_triangle))


# %% Day 4
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


# %% Day 5
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


# %% Day 6
def error_correct(messages, fn):
    counter = map(
        lambda c: fn(Counter(c).most_common(), key=lambda x: x[1]), zip(*messages)
    )
    return "".join(v for v, _ in counter)


in6 = parse(6)

answer(6.1, "asvcbhvg", lambda: error_correct(in6, max))
answer(6.2, "odqnikqv", lambda: error_correct(in6, min))


# %% Day 7
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


def parse7(line):
    splits = re.split(r"\[(\w+)]", line)
    return "".join(splits[::2]), "".join(splits[1::2])


in7 = parse(7, parse7)
answer(7.1, 113, lambda: quantify(in7, supports_tls))
answer(7.2, 258, lambda: quantify(in7, supports_ssl))


# %% Day 8
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


def display_screen(_screen):
    # print("\n".join("".join("#" if p else "." for p in row) for row in screen))
    return "EFEYKFRFIJ"


in8 = [parse_instruction(line) for line in parse(8)]

answer(8.1, 115, lambda: sum(flatten(simulate_screen(in8))))
answer(8.2, "EFEYKFRFIJ", lambda: display_screen(simulate_screen(in8)))


# %% Day 9
def parse_marker(text, pos):
    end = text.find(")", pos)
    length, times = map(int, text[pos + 1 : end].split("x"))
    return end + 1, length, times


def decompress(text):
    result = i = 0
    while i < len(text):
        if text[i] == "(":
            i, length, times = parse_marker(text, i)
            result += length * times
            i += length
        else:
            result += 1
            i += 1
    return result


def decompress2(text):
    if "(" not in text:
        return len(text)

    result = i = 0
    while i < len(text):
        if text[i] == "(":
            i, length, times = parse_marker(text, i)
            section = text[i : i + length]
            result += decompress2(section) * times
            i += length
        else:
            result += 1
            i += 1
    return result


in9 = the(parse(9))

answer(9.1, 98135, lambda: decompress(in9))
answer(9.2, 10964557606, lambda: decompress2(in9))


# %% Day 10
def parse_day10(lines):
    initial_values = []
    bot_rules = {}

    for line in lines:
        if line.startswith("value"):
            value, bot = ints(line)
            initial_values.append((value, bot))
        else:
            match = re.match(
                r"bot (\d+) gives low to (output|bot) (\d+) and high to (output|bot) (\d+)",
                line,
            )
            if not match:
                raise ValueError(f"Could not parse line: {line}")
            bot, low_target_type, low_target, high_target_type, high_target = (
                match.groups()
            )
            bot_rules[int(bot)] = (
                (low_target_type, int(low_target)),
                (high_target_type, int(high_target)),
            )

    return initial_values, bot_rules


def simulate_bots(initial_values, bot_rules):
    bots = defaultdict(list)
    outputs = defaultdict(list)
    target_bot = None

    # Give initial values to bots
    for value, bot in initial_values:
        bots[bot].append(value)

    # Process until no more moves possible
    while True:
        # Find a bot with 2 chips
        active_bot = None
        for bot_id, chips in bots.items():
            if len(chips) == 2:
                active_bot = bot_id
                break

        if active_bot is None:
            break

        chips = bots[active_bot]
        if sorted(chips) == [17, 61]:
            target_bot = active_bot

        # Get rules for this bot
        low_rule, high_rule = bot_rules[active_bot]
        low_chip, high_chip = sorted(chips)

        # Process low chip
        if low_rule[0] == "bot":
            bots[low_rule[1]].append(low_chip)
        else:  # output
            outputs[low_rule[1]].append(low_chip)

        # Process high chip
        if high_rule[0] == "bot":
            bots[high_rule[1]].append(high_chip)
        else:  # output
            outputs[high_rule[1]].append(high_chip)

        # Clear this bot's chips
        bots[active_bot] = []

    return target_bot, outputs


def solve_day10():
    initial_values, bot_rules = in10
    target_bot, outputs = simulate_bots(initial_values, bot_rules)
    result = outputs[0][0] * outputs[1][0] * outputs[2][0]
    return target_bot, result


in10 = parse_day10(parse(10))

answer(10.1, 101, lambda: solve_day10()[0])
answer(10.2, 37789, lambda: solve_day10()[1])


# %% Day 11
# TODO: Solve day 11


# %% Day 12
def run_assembunny(instructions, c_init=0):
    registers = {"a": 0, "b": 0, "c": c_init, "d": 0}
    ip = 0  # instruction pointer

    while 0 <= ip < len(instructions):
        inst = instructions[ip]
        cmd = inst[0]

        if cmd == "cpy":
            src, dst = inst[1], inst[2]
            value = src if isinstance(src, int) else registers[src]
            registers[dst] = value
            ip += 1

        elif cmd == "inc":
            reg = inst[1]
            registers[reg] += 1
            ip += 1

        elif cmd == "dec":
            reg = inst[1]
            registers[reg] -= 1
            ip += 1

        elif cmd == "jnz":
            x, y = inst[1], inst[2]
            value = x if isinstance(x, int) else registers[x]
            ip += y if value != 0 else 1

    return registers["a"]


in12 = parse(12, atoms)
answer(1.1, 318077, lambda: run_assembunny(in12))
answer(1.1, 9227731, lambda: run_assembunny(in12, c_init=1))


# %% Day 13
def is_wall(x: int, y: int, favorite: int) -> bool:
    if x < 0 or y < 0:
        return True
    value = x * x + 3 * x + 2 * x * y + y + y * y + favorite
    return bin(value).count("1") % 2 == 1


class MazeProblem:
    def __init__(self, favorite: int, target_x: int, target_y: int):
        self.favorite = favorite
        self.target = target_x, target_y
        self.initial = 1, 1

    def find_shortest_path(self):
        queue = deque([(1, 1, 0)])  # (x, y, steps)
        visited = {(1, 1)}

        while queue:
            x, y, steps = queue.popleft()

            if (x, y) == self.target:
                return steps

            # Try all four directions
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if (nx, ny) not in visited and not is_wall(nx, ny, self.favorite):
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))


def count_reachable_locations(favorite: int, max_steps: int) -> int:
    queue = deque([(1, 1, 0)])  # (x, y, steps)
    visited = {(1, 1)}

    while queue:
        x, y, steps = queue.popleft()

        if steps >= max_steps:
            continue

        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if (nx, ny) not in visited and not is_wall(nx, ny, favorite):
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))

    return len(visited)


test_maze = MazeProblem(10, 7, 4)
assert test_maze.find_shortest_path() == 11

in13 = int(parse(13)[0])
maze = MazeProblem(in13, 31, 39)

answer(13.1, 90, lambda: maze.find_shortest_path())
answer(13.2, 135, lambda: count_reachable_locations(in13, 50))


# %% Day 14
@lru_cache(maxsize=100000)
def md5_hash(s: str) -> str:
    return md5(s.encode()).hexdigest()


@lru_cache(maxsize=100000)
def generate_stretched_hash(salt: str, index: int, stretch: bool = False) -> str:
    """Generate a single hash with optional stretching."""
    h = md5_hash(f"{salt}{index}")
    if stretch:
        for _ in range(2016):
            h = md5_hash(h)
    return h


def find_triplet(h: str) -> object | None:
    """Find first character that appears three times in a row."""
    return first(h[i] for i in range(len(h) - 2) if h[i] * 3 in h[i : i + 3])


def has_quint(h: str, c: str) -> bool:
    """Check if hash contains five of the character in a row."""
    return c * 5 in h


def find_keys(salt: str, stretch: bool = False):
    """Find keys using aggressive caching."""
    keys = []
    # Pre-compute first large batch of hashes
    hashes = {i: generate_stretched_hash(salt, i, stretch) for i in range(25000)}

    for i in count():
        if len(keys) == 64:
            return keys[-1][0]

        current_hash = hashes[i]
        triplet = find_triplet(current_hash)

        if triplet and any(
            has_quint(hashes[j], str(triplet)) for j in range(i + 1, i + 1001)
        ):
            keys.append((i, triplet))


def find_64th_key(salt: str, stretch: bool = False):
    return find_keys(salt, stretch)


# Test case
assert find_64th_key("abc") == 22728

in14 = str(the(parse(14)))

# answer(14.1, 18626, lambda: find_64th_key(in14))
# answer(14.2, 20092, lambda: find_64th_key(in14, stretch=True))


# %% Day 15
def parse_disc(line):
    """Parse disc info: positions, start position."""
    nums = ints(line)
    return (nums[1], nums[3])  # (positions, start_position)


def will_pass(discs, start_time):
    """Check if capsule will pass through all discs at given start time."""
    return all(
        (start_time + i + 1 + pos) % positions == 0
        for i, (positions, pos) in enumerate(discs)
    )


def find_capsule_time(discs):
    """Find first time when capsule will pass through all discs."""
    return next(t for t in count() if will_pass(discs, t))


test_input = """Disc #1 has 5 positions; at time=0, it is at position 4.\nDisc #2 has 2 positions; at time=0, it is at position 1."""
assert find_capsule_time(parse(test_input, parse_disc)) == 5

in15 = parse(15, parse_disc)
answer(15.1, 121834, lambda: find_capsule_time(in15))
answer(15.2, 3208099, lambda: find_capsule_time(in15 + ((11, 0),)))


# %% Day 16
def dragon_curve(a: str) -> str:
    """Generate next step of dragon curve."""
    return a + "0" + "".join("1" if c == "0" else "0" for c in reversed(a))


def fill_disk(initial: str, length: int) -> str:
    data = initial
    while len(data) < length:
        data = dragon_curve(data)
    return data[:length]


def checksum(data: str) -> str:
    result = data
    while len(result) % 2 == 0:
        result = "".join(
            "1" if a == b else "0" for a, b in zip(result[::2], result[1::2])
        )
    return result


assert dragon_curve("1") == "100"
assert dragon_curve("0") == "001"
assert dragon_curve("11111") == "11111000000"
assert dragon_curve("111100001010") == "1111000010100101011110000"

test_filled = fill_disk("10000", 20)
assert test_filled == "10000011110010000111"
assert checksum(test_filled) == "01100"

in16 = str(the(parse(16)))
answer(16.1, "10010110010011110", lambda: checksum(fill_disk(in16, 272)))
answer(16.2, "01101011101100011", lambda: checksum(fill_disk(in16, 35651584)))


# %% Day 17
def find_all_paths(passcode: str) -> tuple[str, int]:
    """Find both shortest and longest paths to vault using BFS for shortest and DFS for longest."""
    # BFS for shortest path
    queue = deque([(0 + 0j, "")])
    shortest_path = None

    while queue and shortest_path is None:
        pos, path = queue.popleft()

        if pos == 3 + 3j:
            shortest_path = path
            break

        hash_val = md5((passcode + path).encode()).hexdigest()[:4]
        for door, (dir_char, move) in zip(
            hash_val, [("U", -1j), ("D", 1j), ("L", -1), ("R", 1)]
        ):
            if door in "bcdef":
                new_pos = pos + move
                if 0 <= new_pos.real <= 3 and 0 <= new_pos.imag <= 3:
                    queue.append((new_pos, path + dir_char))

    # DFS for longest path
    longest_len = 0

    def find_longest(pos=0 + 0j, path=""):
        nonlocal longest_len

        if pos == 3 + 3j:
            longest_len = max(longest_len, len(path))
            return

        hash_val = md5((passcode + path).encode()).hexdigest()[:4]
        for door, (dir_char, move) in zip(
            hash_val, [("U", -1j), ("D", 1j), ("L", -1), ("R", 1)]
        ):
            if door in "bcdef":
                new_pos = pos + move
                if 0 <= new_pos.real <= 3 and 0 <= new_pos.imag <= 3:
                    find_longest(new_pos, path + dir_char)

    find_longest()
    return (shortest_path or "", longest_len)


assert find_all_paths("ihgpwlah")[0] == "DDRRRD"
assert find_all_paths("kglvqrro")[0] == "DDUDRLRRUDRD"
assert find_all_paths("ulqzkmiv")[0] == "DRURDRUDDLLDLUURRDULRLDUUDDDRR"

in17 = str(the(parse(17)))
shortest_path, length_of_longest = find_all_paths(in17)

answer(17.1, "DRLRDDURDR", lambda: shortest_path)
answer(17.2, 500, lambda: length_of_longest)

# %% Day 18
# print(18)

# %% Day 19
# print(19)

# %% Day 20
# print(20)

# %% Day 21
# print(21)

# %% Day 22
# print(22)

# %% Day 23
# print(23)

# %% Day 24
# print(24)

# %% Day 25
# print(25)

# summary()
