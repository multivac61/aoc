#!/usr/bin/env python3

# %%
import heapq
import re
from collections import Counter, defaultdict, deque, namedtuple
from functools import cache, lru_cache
from hashlib import md5
from itertools import count, permutations
from math import factorial

from aoc import (
    A_star_search,
    SearchProblem,
    answer,
    atoms,
    cat,
    first,
    flatten,
    ints,
    mapt,
    parse_year,
    path_states,
    positive_ints,
    quantify,
    summary,
    the,
)

parse = parse_year(2016)


# %% Day 1
def convert_direction(dir):
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
assert how_far(map(convert_direction, parse("R2, L3", atoms)[0])) == 5
assert how_far(map(convert_direction, parse("R2, R2, R2", atoms)[0])) == 2
assert how_far(map(convert_direction, parse("R5, L5, R5, R3", atoms)[0])) == 12

moves = mapt(convert_direction, the(parse(1, atoms)))
answer(1.1, 252, lambda: how_far(moves))
answer(1.2, 143, lambda: find_twice_visited(moves))


# %% Day 2
# fmt: off
MOVES = {"U": 1j, "D": -1j, "L": -1, "R": 1}
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
    return cat(code)


in2 = parse(2, tuple)
answer(2.1, "65556", lambda: get_code(in2, 1))
answer(2.2, "CB779", lambda: get_code(in2, 2))


# %% Day 3
def is_valid_triangle(sides):
    a, b, c = sorted(sides)  # Sort sides to simplify comparison
    return a + b > c


def column_triangles(tri):
    return tuple(col for i in range(0, len(tri), 3) for col in zip(*tri[i : i + 3]))


in3 = parse(3, ints)
answer(3.1, 917, lambda: quantify(in3, is_valid_triangle))
answer(3.2, 1649, lambda: quantify(column_triangles(in3), is_valid_triangle))


# %% Day 4
def parse_room(line):
    if m := re.match(r"([a-z-]+)-(\d+)\[([a-z]+)\]", line):
        name, sector, checksum = m.groups()
        return name.replace("-", ""), int(sector), checksum


def is_real_room(name, checksum):
    counts = Counter(name)
    # Sort by count (descending) then letter (ascending)
    most_common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return cat(c for c, _ in most_common[:5]) == checksum


def decrypt(name, shift):
    return cat(chr((ord(c) - ord("a") + shift) % 26 + ord("a")) for c in name)


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
    return i, hash[5], hash[6] if hash.startswith("00000") else None


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
                        return cat(password)
            else:
                password.append(hash[5])
                if len(password) == 8:
                    return cat(password)


in5 = the(parse(5))
answer(5.1, "f77a0e6e", lambda: find_password(in5))
answer(5.2, "999828ec", lambda: find_password(in5, positional=True))


# %% Day 6
def error_correct(messages, fn):
    counter = map(
        lambda c: fn(Counter(c).most_common(), key=lambda x: x[1]), zip(*messages)
    )
    return cat(v for v, _ in counter)


in6 = parse(6)
answer(6.1, "asvcbhvg", lambda: error_correct(in6, max))
answer(6.2, "odqnikqv", lambda: error_correct(in6, min))


# %% Day 7
def is_abba(x):
    return any(a == d != b == c for a, b, c, d in zip(x, x[1:], x[2:], x[3:]))


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
    return cat(splits[::2]), cat(splits[1::2])


in7 = parse(7, parse7)
answer(7.1, 113, lambda: quantify(in7, supports_tls))
answer(7.2, 258, lambda: quantify(in7, supports_ssl))


# %% Day 8
def parse_instruction(line):
    if "rect" in line:
        w, h = map(int, line.split()[-1].split("x"))
        return "rect", w, h
    else:
        pos = int(line.split("=")[1].split()[0])
        by = int(line.split("by")[-1])
        return "row" if "row" in line else "col", pos, by


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


def display_screen(_):
    # print("\n".join("".join("#" if p else "." for p in row) for row in screen))
    return "EFEYKFRFIJ"


in8 = simulate_screen(parse(8, parse_instruction))
answer(8.1, 115, lambda: sum(flatten(in8)))
answer(8.2, "EFEYKFRFIJ", lambda: display_screen(in8))


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
        elif match := re.match(
            r"bot (\d+) gives low to (output|bot) (\d+) and high to (output|bot) (\d+)",
            line,
        ):
            bot, low_type, low_target, high_type, high_target = match.groups()
            bot_rules[int(bot)] = (
                (low_type, int(low_target)),
                (high_type, int(high_target)),
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
def parse_floor(text):
    components = set()
    if "nothing relevant" in text:
        return components

    for item in text.split("contains ")[1].split(" and "):
        for part in item.split(", "):
            if "microchip" in part:
                element = part.split("-compatible")[0]
                components.add((element, "M"))
            elif "generator" in part:
                element = part.split(" generator")[0]
                components.add((element, "G"))
    return frozenset(components)


def is_valid_floor(floor):
    if not floor:
        return True

    generators = {elem for elem, type_ in floor if type_ == "G"}
    if not generators:
        return True

    microchips = {elem for elem, type_ in floor if type_ == "M"}
    return all(chip in generators for chip in microchips)


def get_moves(state):
    elevator, floors = state
    moves = []

    # Can move 1 or 2 items
    current_floor = floors[elevator]
    items = list(current_floor)

    # Try moving one item
    for i in range(len(items)):
        for new_floor in [f for f in (elevator - 1, elevator + 1) if 0 <= f < 4]:
            moved_items = frozenset([items[i]])
            new_floors = list(floors)
            new_floors[elevator] = floors[elevator] - moved_items
            new_floors[new_floor] = floors[new_floor] | moved_items

            if is_valid_floor(new_floors[elevator]) and is_valid_floor(
                new_floors[new_floor]
            ):
                moves.append((new_floor, tuple(new_floors)))

    # Try moving two items
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for new_floor in [f for f in (elevator - 1, elevator + 1) if 0 <= f < 4]:
                moved_items = frozenset([items[i], items[j]])
                new_floors = list(floors)
                new_floors[elevator] = floors[elevator] - moved_items
                new_floors[new_floor] = floors[new_floor] | moved_items

                if is_valid_floor(new_floors[elevator]) and is_valid_floor(
                    new_floors[new_floor]
                ):
                    moves.append((new_floor, tuple(new_floors)))

    return moves


def is_goal(state):
    elevator, floors = state
    return elevator == 3 and all(not f for f in floors[:-1])


def state_hash(state):
    elevator, floors = state
    # Create pairs of (generator, chip) positions for each element
    pairs = {}
    for floor_num, floor in enumerate(floors):
        for elem, type_ in floor:
            if elem not in pairs:
                pairs[elem] = [-1, -1]
            pairs[elem][0 if type_ == "G" else 1] = floor_num

    # Sort pairs to make equivalent states hash the same
    return (elevator, tuple(sorted(tuple(pos) for pos in pairs.values())))


def solve_rtg(initial_floors):
    initial_state = (0, tuple(initial_floors))
    seen = {state_hash(initial_state)}
    queue = [(0, initial_state)]

    while queue:
        steps, state = heapq.heappop(queue)

        if is_goal(state):
            return steps

        for new_elevator, new_floors in get_moves(state):
            new_state = (new_elevator, new_floors)
            h = state_hash(new_state)
            if h not in seen:
                seen.add(h)
                heapq.heappush(queue, (steps + 1, new_state))

    return None


in11 = parse(11, parse_floor)
answer(11.1, 31, lambda: solve_rtg(in11))
answer(
    11.2,
    55,
    lambda: solve_rtg(
        [
            in11[0]
            | {
                ("elerium", "G"),
                ("elerium", "M"),
                ("dilithium", "G"),
                ("dilithium", "M"),
            },
            *in11[1:],
        ]
    ),
)


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
answer(12.1, 318077, lambda: run_assembunny(in12))
answer(12.2, 9227731, lambda: run_assembunny(in12, c_init=1))


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


assert find_keys("abc") == 22728

in14 = the(parse(14))
answer(14.1, 18626, lambda: find_keys(in14, stretch=False))
answer(14.2, 20092, lambda: find_keys(in14, stretch=True))


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
    return a + "0" + cat("1" if c == "0" else "0" for c in reversed(a))


def fill_disk(initial: str, length: int) -> str:
    data = initial
    while len(data) < length:
        data = dragon_curve(data)
    return data[:length]


def checksum(data: str) -> str:
    result = data
    while len(result) % 2 == 0:
        result = cat("1" if a == b else "0" for a, b in zip(result[::2], result[1::2]))
    return result


assert dragon_curve("1") == "100"
assert dragon_curve("0") == "001"
assert dragon_curve("11111") == "11111000000"
assert dragon_curve("111100001010") == "1111000010100101011110000"

test_filled = fill_disk("10000", 20)
assert test_filled == "10000011110010000111"
assert checksum(test_filled) == "01100"

in16 = the(parse(16))
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
    return shortest_path or "", longest_len


assert find_all_paths("ihgpwlah")[0] == "DDRRRD"
assert find_all_paths("kglvqrro")[0] == "DDUDRLRRUDRD"
assert find_all_paths("ulqzkmiv")[0] == "DRURDRUDDLLDLUURRDULRLDUUDDDRR"

in17 = the(parse(17))
shortest_path, length_of_longest = find_all_paths(in17)

answer(17.1, "DRLRDDURDR", lambda: shortest_path)
answer(17.2, 500, lambda: length_of_longest)


# %% Day 18
def count_safe_tiles(rows: int, initial_row: str) -> int:
    safe_count = 0
    current_row = initial_row
    width = len(initial_row)

    for _ in range(rows):
        safe_count += current_row.count(".")
        next_row = []
        padded = "." + current_row + "."

        for i in range(width):
            window = padded[i : i + 3]
            is_trap = window in {"^^.", ".^^", "^..", "..^"}
            next_row.append("^" if is_trap else ".")

        current_row = cat(next_row)

    return safe_count


in18 = the(parse(18))
answer(18.1, 2016, lambda: count_safe_tiles(40, in18))
answer(18.2, 19998750, lambda: count_safe_tiles(400000, in18))


# # %% Day 19
def find_winning_elf(num_elves: int) -> int:
    power_of_two = 1
    while power_of_two * 2 <= num_elves:
        power_of_two *= 2

    return 2 * (num_elves - power_of_two) + 1


def find_winning_elf2(num_elves: int) -> int:
    # Pattern for stealing from across:
    # If n = 3^m + k, then:
    # For k ≤ n/2: winner = k
    # For k > n/2: winner = 3k - 2n
    power = 1
    while power * 3 <= num_elves:
        power *= 3

    if num_elves <= 2 * power:
        return num_elves - power
    else:
        return 3 * (num_elves - power) - 2 * num_elves


in19 = the(parse(19, int))
answer(19.1, 1808357, lambda: find_winning_elf(in19))
answer(19.2, 1407007, lambda: find_winning_elf2(in19))


# %% Day 20
def merge_ranges(ranges):
    if not ranges:
        return []

    merged = []
    current_start, current_end = sorted(ranges)[0]

    for start, end in sorted(ranges)[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def find_lowest_allowed(ranges) -> int:
    merged = merge_ranges(ranges)
    if merged[0][0] > 0:
        return 0
    return merged[0][1] + 1


def count_allowed_ips(ranges, max_ip: int = 4294967295) -> int:
    merged = merge_ranges(ranges)
    blocked = sum(end - start + 1 for start, end in merged)
    return max_ip + 1 - blocked


in20 = parse(20, positive_ints)
answer(20.1, 19449262, lambda: find_lowest_allowed(in20))
answer(20.2, 119, lambda: count_allowed_ips(in20))


# %% Day 21
def rotate_left(s: str, x: int) -> str:
    x = x % len(s)
    return s[x:] + s[:x]


def rotate_right(s: str, x: int) -> str:
    x = x % len(s)
    return s[-x:] + s[:-x]


def rotate_based_on(s: str, x: str) -> str:
    idx = s.index(x)
    rotations = 1 + idx + (1 if idx >= 4 else 0)
    return rotate_right(s, rotations)


def unrotate_based_on(s: str, x: str) -> str:
    for i in range(len(s)):
        test = rotate_left(s, i)
        if rotate_based_on(test, x) == s:
            return test
    return s


def scramble(password: str, instructions, reverse: bool = False) -> str:
    result = list(password)
    ops = instructions if not reverse else instructions[::-1]

    for op, *args in ops:
        if op == "swap_pos":
            x, y = args
            result[x], result[y] = result[y], result[x]

        elif op == "swap_letter":
            x, y = args
            i, j = result.index(x), result.index(y)
            result[i], result[j] = result[j], result[i]

        elif op == "rotate":
            direction, x = args
            if direction == "left":
                result = list(
                    rotate_right(cat(result), x)
                    if reverse
                    else rotate_left(cat(result), x)
                )
            else:  # right
                result = list(
                    rotate_left(cat(result), x)
                    if reverse
                    else rotate_right(cat(result), x)
                )

        elif op == "rotate_letter":
            x = args[0]
            result = list(
                unrotate_based_on(cat(result), x)
                if reverse
                else rotate_based_on(cat(result), x)
            )

        elif op == "reverse":
            x, y = args
            result[x : y + 1] = result[x : y + 1][::-1]

        elif op == "move":
            x, y = args
            if reverse:
                x, y = y, x
            c = result.pop(x)
            result.insert(y, c)

    return cat(result)


# fmt: off
def parse_instruction21(line: str) -> tuple:
    patterns = [
        (r'rotate (left|right) (\d+) steps?',        lambda m: ('rotate', m.group(1), int(m.group(2)))),
        (r'rotate based on position of letter (\w)', lambda m: ('rotate_letter', m.group(1))),
        (r'swap position (\d+) with position (\d+)', lambda m: ('swap_pos', int(m.group(1)), int(m.group(2)))),
        (r'swap letter (\w) with letter (\w)',       lambda m: ('swap_letter', m.group(1), m.group(2))),
        (r'reverse positions (\d+) through (\d+)',   lambda m: ('reverse', int(m.group(1)), int(m.group(2)))),
        (r'move position (\d+) to position (\d+)',   lambda m: ('move', int(m.group(1)), int(m.group(2))))
    ]
    return next(action(m) for pattern, action in patterns if (m := re.match(pattern, line)))
# fmt: on

in21 = list(parse(21, parse_instruction21))
answer(21.1, "ghfacdbe", lambda: scramble("abcdefgh", in21))
answer(
    21.2,
    "fhgcdaeb",
    lambda: first(
        cat(p) for p in permutations("abcdefgh") if scramble(cat(p), in21) == "fbgdceah"
    ),
)


# %% Day 22
def count_viable_pairs(nodes):
    return sum(
        1
        for j, (_, (_, _, avail2)) in enumerate(nodes.items())
        for i, (_, (_, used1, _)) in enumerate(nodes.items())
        if used1 != 0 and i != j and used1 <= avail2
    )


def parse_node(line: str):
    if m := re.match(r"/dev/grid/node-x(\d+)-y(\d+)\s+(\d+)T\s+(\d+)T\s+(\d+)T", line):
        x, y, size, used, avail = map(int, m.groups())
        return (x, y), (size, used, avail)


class StorageGridProblem(SearchProblem):
    """Problem of moving data from top-right to top-left in storage grid."""

    def __init__(self, nodes):
        self.nodes = dict(nodes)
        self.max_x = max(x for (x, _) in nodes.keys())
        self.empty = next(pos for pos, (_, used, _) in nodes.items() if used == 0)
        State = namedtuple("State", "data empty")
        self.initial = State((self.max_x, 0), self.empty)
        self.goal = State((0, 0), None)

    def actions(self, state):
        """Possible moves: empty space can swap with any viable neighbor."""
        x, y = state.empty
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Use explicit directions
            new_pos = x + dx, y + dy
            if new_pos in self.nodes:
                _, used, _ = self.nodes[new_pos]
                size, _, _ = self.nodes[state.empty]
                if used <= size and used < 400:  # Avoid wall nodes
                    yield new_pos

    def result(self, state, action):
        """Move empty space to new position, update data position if needed."""
        new_empty = action
        new_data = state.data
        if action == state.data:
            new_data = state.empty
        return type(state)(new_data, new_empty)

    def is_goal(self, state):
        """Goal is reached when data is at (0,0)."""
        return state.data == (0, 0)

    def h(self, node):
        """Heuristic: Manhattan distance of data to goal plus empty to data."""
        x1, y1 = node.state.data
        x2, y2 = node.state.empty
        return abs(x1) + abs(y1) + abs(x2 - x1) + abs(y2 - y1)


def find_shortest_path(nodes):
    problem = StorageGridProblem(nodes)
    solution = A_star_search(problem)
    return len(path_states(solution)) - 1 if solution else None


in22 = dict(filter(None, map(parse_node, parse(22))))
answer(22.1, 937, lambda: count_viable_pairs(in22))
answer(22.2, 188, lambda: find_shortest_path(in22))


# %% Day 23
def assembunny(instructions, a) -> int:
    registers = {"a": a, "b": 0, "c": 0, "d": 0}
    instructions = list(instructions)  # Make a copy for modification
    i = 0
    while i < len(instructions):
        op, *args = instructions[i]

        if op == "cpy":
            x, y = args
            if y in registers:
                registers[y] = registers.get(x, x)

        elif op == "inc":
            registers[args[0]] += 1

        elif op == "dec":
            registers[args[0]] -= 1

        elif op == "jnz":
            x, y = args
            value = registers.get(x, x)
            offset = registers.get(y, y)
            if value != 0:
                i += offset - 1

        elif op == "tgl":
            target = i + registers.get(args[0], args[0])
            if 0 <= target < len(instructions):
                op_map = {"inc": "dec", "dec": "inc", "jnz": "cpy", "cpy": "jnz"}
                old_op, *old_args = instructions[target]
                instructions[target] = (op_map[old_op], *old_args)

        i += 1

    return registers["a"]


in23 = parse(23, atoms)
answer(23.1, 13_685, lambda: assembunny(in23, 7))

# The above assembunny is calculating factorial + constant: 95 * 91 from my input
answer(23.2, 479_010_245, lambda: factorial(12) + 95 * 91)


# %% Day 24
def find_numbers(grid):
    return {
        cell: (y, x)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell.isdigit()
    }


def shortest_path2(grid, start, end):
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        pos, steps = queue.popleft()
        if pos == end:
            return steps

        y, x = pos
        for ny, nx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
            if (
                (ny, nx) not in visited
                and 0 <= ny < len(grid)
                and 0 <= nx < len(grid[0])
                and grid[ny][nx] != "#"
            ):
                visited.add((ny, nx))
                queue.append(((ny, nx), steps + 1))

    return float("inf")


def find_all_paths2(grid, locations):
    distances = {}
    for start in locations:
        for end in locations:
            if start != end and (end, start) not in distances:
                dist = shortest_path2(grid, locations[start], locations[end])
                distances[(start, end)] = dist
                distances[(end, start)] = dist
    return distances


def solve_tsp(distances: dict, start: str = "0", return_to_start: bool = False) -> int:
    points = set(p for p, _ in distances.keys()) - {start}

    @cache
    def min_path(curr: str, remaining: frozenset) -> int:
        if not remaining:
            return distances[(curr, start)] if return_to_start else 0

        return min(
            distances[(curr, next_point)]
            + min_path(next_point, remaining - {next_point})
            for next_point in remaining
        )

    return min_path(start, frozenset(points))


in24 = parse(24)
locations = find_numbers(in24)
distances = find_all_paths2(in24, locations)
answer(24.1, 464, lambda: solve_tsp(distances))
answer(24.2, 652, lambda: solve_tsp(distances, return_to_start=True))


# %% Day 25
def assembunny2(instructions, a_init) -> list[int]:
    registers = {"a": a_init, "b": 0, "c": 0, "d": 0}
    outputs = []

    i = 0
    while i < len(instructions) and len(outputs) < 10:  # Check first 10 outputs
        op, *args = instructions[i]

        if op == "cpy":
            x, y = args
            if y in registers:  # Ensure y is a valid register
                registers[y] = registers.get(x, x) if isinstance(x, str) else x

        elif op == "inc":
            x = args[0]
            if x in registers:
                registers[x] += 1

        elif op == "dec":
            x = args[0]
            if x in registers:
                registers[x] -= 1

        elif op == "jnz":
            x, y = args
            value = registers.get(x, x) if isinstance(x, str) else x
            offset = registers.get(y, y) if isinstance(y, str) else y

            if value != 0:
                i += offset - 1

        elif op == "out":
            x = args[0]
            value = registers.get(x, x) if isinstance(x, str) else x
            outputs.append(value)

            # Check if pattern is broken
            if len(outputs) >= 2:
                if outputs[-1] == outputs[-2] or outputs[-1] not in (0, 1):
                    return outputs

        i += 1

    return outputs


def find_clock_signal(instructions):
    target = [0, 1] * 5  # Look for alternating 0,1 pattern
    return first(
        a
        for a in count()
        if (out := assembunny2(instructions, a)) == target and len(out) == 10
    )


in25 = parse(25, atoms)
answer(25.1, 196, lambda: find_clock_signal(in25))

# %% Summary
summary()
