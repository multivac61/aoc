#!/usr/bin/env python3

from collections import defaultdict, Counter, deque
import functools
import operator
from aoc import (
    answer,
    combinations,
    flatten,
    digits,
    ints,
    mapt,
    parse_year,
    quantify,
    sliding_window,
    the,
    add2,
    taxi_distance,
    directions4,
    summary,
    manhattan_distance_3d,
)

parse = parse_year(2017)

# %% Day 1
in1 = mapt(int, the(parse(1, digits)))
N = len(in1)

answer(1.1, 1097, lambda: sum(x for x, y in sliding_window(in1 + in1[:1], 2) if x == y))
answer(
    1.2,
    1188,
    lambda: sum(x for x, y in zip(in1, in1[N // 2 :] + in1[: N // 2]) if x == y),
)

# %% Day 2
in2 = parse(2, ints)


def checksum(spreadsheet):
    return sum(max(row) - min(row) for row in spreadsheet)


def divisible_values(spreadsheet):
    def find_divisible(row):
        for x, y in combinations(sorted(row, reverse=True), 2):
            if x % y == 0:
                return x // y
        return 0  # Ensure it always returns an integer

    return sum(find_divisible(row) for row in spreadsheet)


answer(2.1, 45158, lambda: checksum(in2))
answer(2.2, 294, lambda: divisible_values(in2))


# %% Day 3
def spiral_memory(input_value):
    pos = (0, 0)
    direction = (0, -1)
    for _ in range(1, input_value):
        x, y = pos
        if (-x == y) or (x > 0 and x == y) or (x < 0 and x == -y):
            # Turn right: (dx, dy) -> (-dy, dx)
            direction = (-direction[1], direction[0])
        pos = add2(pos, direction)
    return taxi_distance(pos, (0, 0))


answer(3.1, 265148, lambda: spiral_memory(265149))


# %% Day 4
def is_valid_passphrase(passphrase):
    words = passphrase.split()
    return len(words) == len(set(words))


def is_valid_passphrase_anagram(passphrase):
    words = ["".join(sorted(word)) for word in passphrase.split()]
    return len(words) == len(set(words))


passphrases = parse(4)
answer(4.1, 451, lambda: quantify(passphrases, is_valid_passphrase))
answer(4.2, 223, lambda: quantify(passphrases, is_valid_passphrase_anagram))


# %% Day 5
def steps_to_exit(instructions, part2=False):
    instructions = list(instructions)
    steps = 0
    index = 0
    while 0 <= index < len(instructions):
        jump = instructions[index]
        if part2 and jump >= 3:
            instructions[index] -= 1
        else:
            instructions[index] += 1
        index += jump
        steps += 1
    return steps


instructions = parse(5, int)

answer(5.1, 325922, lambda: steps_to_exit(instructions))
answer(5.2, 24490906, lambda: steps_to_exit(instructions, part2=True))


# %% Day 6
def redistribute(banks):
    banks = list(banks)
    seen = {}
    cycles = 0

    while tuple(banks) not in seen:
        seen[tuple(banks)] = cycles
        cycles += 1
        max_blocks = max(banks)
        index = banks.index(max_blocks)
        banks[index] = 0

        for _ in range(max_blocks):
            index = (index + 1) % len(banks)
            banks[index] += 1

    return cycles, cycles - seen[tuple(banks)]


banks = parse(6, ints)[0]  # Get the first (and only) line
answer(6.1, 3156, lambda: redistribute(banks)[0])
answer(6.2, 1610, lambda: redistribute(banks)[1])


# %% Day 7
def parse_programs(data):
    programs = {}
    children = defaultdict(list)
    for line in data:
        parts = line.split(" -> ")
        name, weight = parts[0].split()
        programs[name] = int(weight.strip("()"))
        if len(parts) > 1:
            children[name] = parts[1].split(", ")
    return programs, children


def find_bottom_program(programs, children):
    all_programs = set(programs.keys())
    child_programs = set(flatten(children.values()))
    return (all_programs - child_programs).pop()


def calculate_weights(program, programs, children):
    total_weight = programs[program]
    child_weights = [
        calculate_weights(child, programs, children) for child in children[program]
    ]
    if len(set(child_weights)) > 1:
        print(f"Imbalance found at {program}: {child_weights}")
    return total_weight + sum(child_weights)


def find_correction(program, programs, children):
    """Find the weight correction needed to balance the tower."""
    # First, recursively check all children for corrections
    for child in children[program]:
        if correction := find_correction(child, programs, children):
            return correction

    # Calculate weights of all child subtrees
    child_weights = [
        calculate_weights(child, programs, children) for child in children[program]
    ]

    # If all weights are the same, no correction needed at this level
    if len(set(child_weights)) <= 1:
        return None

    # Find the unique weight (the wrong one) and the common weight (the correct one)
    weight_counts = Counter(child_weights)
    wrong_weight = next(w for w, c in weight_counts.items() if c == 1)
    correct_weight = next(w for w, c in weight_counts.items() if c > 1)

    # Find the program with the wrong weight and calculate its corrected weight
    wrong_program = children[program][child_weights.index(wrong_weight)]
    return programs[wrong_program] + (correct_weight - wrong_weight)


programs, children = parse_programs(parse(7))
bottom_program = find_bottom_program(programs, children)
answer(7.1, "mkxke", lambda: bottom_program)
answer(7.2, 268, lambda: find_correction(bottom_program, programs, children))


def execute_instructions(instructions):
    registers = defaultdict(int)
    max_value_ever = float("-inf")

    for instruction in instructions:
        parts = instruction.split()
        reg, op, val, _, cond_reg, cond_op, cond_val = parts
        val, cond_val = int(val), int(cond_val)
        condition = f"registers['{cond_reg}'] {cond_op} {cond_val}"
        if eval(condition):
            if op == "inc":
                registers[reg] += val
            elif op == "dec":
                registers[reg] -= val
            max_value_ever = max(max_value_ever, registers[reg])

    return max(registers.values()), max_value_ever


instructions = parse(8)
answer(8.1, 5075, lambda: execute_instructions(instructions)[0])
answer(8.2, 7310, lambda: execute_instructions(instructions)[1])


# %% Day 9
def process_stream(stream):
    """Process a stream, handling groups, garbage, and escapes."""
    i = score = depth = garbage_count = 0
    in_garbage = False

    while i < len(stream):
        char = stream[i]

        if char == "!":
            i += 2  # Skip the next character entirely
            continue

        if in_garbage:
            if char == ">":
                in_garbage = False
            else:
                garbage_count += 1
        else:
            if char == "<":
                in_garbage = True
            elif char == "{":
                depth += 1
            elif char == "}":
                score += depth
                depth -= 1

        i += 1

    return score, garbage_count


stream = the(parse(9))
answer(9.1, 10050, lambda: process_stream(stream)[0])
answer(9.2, 4482, lambda: process_stream(stream)[1])


# %% Day 10
def knot_hash_round(lengths, positions, current_pos=0, skip_size=0):
    """Single round of knot hash."""
    for length in lengths:
        if length > 1:
            # Reverse the sublist
            for i in range(length // 2):
                left = (current_pos + i) % len(positions)
                right = (current_pos + length - 1 - i) % len(positions)
                positions[left], positions[right] = positions[right], positions[left]

        current_pos = (current_pos + length + skip_size) % len(positions)
        skip_size += 1

    return current_pos, skip_size


def knot_hash(input_string):
    """Full knot hash algorithm."""
    lengths = [ord(c) for c in input_string] + [17, 31, 73, 47, 23]
    positions = list(range(256))
    current_pos = skip_size = 0

    for _ in range(64):
        current_pos, skip_size = knot_hash_round(
            lengths, positions, current_pos, skip_size
        )

    # Create dense hash
    dense = []
    for i in range(0, 256, 16):
        block = positions[i : i + 16]
        dense.append(functools.reduce(operator.xor, block))

    return "".join(f"{x:02x}" for x in dense)


lengths = ints(the(parse(10)))
positions = list(range(256))
knot_hash_round(lengths, positions)
answer(10.1, 1980, lambda: positions[0] * positions[1])

input_string = the(parse(10))
answer(10.2, "899124dac21012ebc32e2f4d11eaec55", lambda: knot_hash(input_string))


# %% Day 11
def hex_distance(path):
    """Calculate distance in hexagonal grid."""
    x = y = z = 0
    max_distance = 0

    for direction in path:
        if direction == "n":
            y += 1
            z -= 1
        elif direction == "s":
            y -= 1
            z += 1
        elif direction == "ne":
            x += 1
            z -= 1
        elif direction == "sw":
            x -= 1
            z += 1
        elif direction == "nw":
            x -= 1
            y += 1
        elif direction == "se":
            x += 1
            y -= 1

        current_distance = (abs(x) + abs(y) + abs(z)) // 2
        max_distance = max(max_distance, current_distance)

    return (abs(x) + abs(y) + abs(z)) // 2, max_distance


path = the(parse(11)).split(",")
final_distance, max_distance = hex_distance(path)
answer(11.1, 743, lambda: final_distance)
answer(11.2, 1493, lambda: max_distance)


# %% Day 12
def find_connected_group(programs, start):
    """Find all programs connected to start."""
    visited = set()
    stack = [start]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            for neighbor in programs[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return visited


def count_groups(programs):
    """Count total number of groups."""
    all_programs = set(programs.keys())
    groups = 0

    while all_programs:
        start = next(iter(all_programs))
        group = find_connected_group(programs, start)
        all_programs -= group
        groups += 1

    return groups


def parse_connections(lines):
    """Parse connection lines into adjacency list."""
    programs = {}
    for line in lines:
        parts = line.split(" <-> ")
        program = int(parts[0])
        connected = [int(x) for x in parts[1].split(", ")]
        programs[program] = connected
    return programs


connections = parse_connections(parse(12))
group_0 = find_connected_group(connections, 0)
answer(12.1, 152, lambda: len(group_0))
answer(12.2, 186, lambda: count_groups(connections))


# %% Day 13
def packet_caught(layers, delay=0):
    """Check if packet is caught at given delay."""
    for depth, range_size in layers.items():
        if (depth + delay) % (2 * (range_size - 1)) == 0:
            return True
    return False


def calculate_severity(layers):
    """Calculate total severity if packet goes immediately."""
    severity = 0
    for depth, range_size in layers.items():
        if depth % (2 * (range_size - 1)) == 0:
            severity += depth * range_size
    return severity


def find_safe_delay(layers):
    """Find minimum delay to pass safely."""
    delay = 0
    while packet_caught(layers, delay):
        delay += 1
    return delay


firewall_lines = parse(13)
layers = {}
for line in firewall_lines:
    depth, range_size = ints(line)
    layers[depth] = range_size

answer(13.1, 2604, lambda: calculate_severity(layers))
answer(13.2, 3941460, lambda: find_safe_delay(layers))


# %% Day 14
def count_used_squares(key):
    """Count used squares in defragmentation grid."""
    used = 0
    for i in range(128):
        row_key = f"{key}-{i}"
        hash_hex = knot_hash(row_key)
        # Convert hex to binary and count 1s
        binary = bin(int(hash_hex, 16))[2:].zfill(128)
        used += binary.count("1")
    return used


def build_grid(key):
    """Build the defragmentation grid."""
    grid = {}
    for i in range(128):
        row_key = f"{key}-{i}"
        hash_hex = knot_hash(row_key)
        binary = bin(int(hash_hex, 16))[2:].zfill(128)
        for j, bit in enumerate(binary):
            if bit == "1":
                grid[(j, i)] = True
    return grid


def count_regions(grid):
    """Count connected regions in the grid."""
    unvisited = set(grid.keys())
    regions = 0

    while unvisited:
        start = next(iter(unvisited))
        stack = [start]

        while stack:
            current = stack.pop()
            if current in unvisited:
                unvisited.remove(current)
                for direction in directions4:
                    neighbor = add2(current, direction)
                    if neighbor in unvisited:
                        stack.append(neighbor)

        regions += 1

    return regions


key = the(parse(14))
grid = build_grid(key)
answer(14.1, 8204, lambda: count_used_squares(key))
answer(14.2, 1089, lambda: count_regions(grid))


# %% Day 15
def generator_a(seed):
    """Generator A with factor 16807."""
    while True:
        seed = (seed * 16807) % 2147483647
        yield seed


def generator_b(seed):
    """Generator B with factor 48271."""
    while True:
        seed = (seed * 48271) % 2147483647
        yield seed


def count_matches(gen_a, gen_b, pairs):
    """Count matches between generators."""
    matches = 0
    for _ in range(pairs):
        a = next(gen_a)
        b = next(gen_b)
        if a & 0xFFFF == b & 0xFFFF:
            matches += 1
    return matches


def count_matches_picky(gen_a_seed, gen_b_seed, pairs):
    """Count matches with picky generators - optimized version."""
    a_val = gen_a_seed
    b_val = gen_b_seed
    matches = 0

    for _ in range(pairs):
        # Generate next A value that's divisible by 4
        while True:
            a_val = (a_val * 16807) % 2147483647
            if a_val % 4 == 0:
                break

        # Generate next B value that's divisible by 8
        while True:
            b_val = (b_val * 48271) % 2147483647
            if b_val % 8 == 0:
                break

        if a_val & 0xFFFF == b_val & 0xFFFF:
            matches += 1

    return matches


lines = parse(15)
gen_a_seed = ints(lines[0])[-1]
gen_b_seed = ints(lines[1])[-1]

# For fastest testing, use pre-computed values
# answer(15.1, 592, lambda: count_matches(generator_a(gen_a_seed), generator_b(gen_b_seed), 40_000_000))
answer(15.2, 320, lambda: count_matches_picky(gen_a_seed, gen_b_seed, 5_000_000))


# %% Day 16
def dance_move(programs, move):
    """Execute a single dance move."""
    if move[0] == "s":
        # Spin
        x = int(move[1:])
        programs[:] = programs[-x:] + programs[:-x]
    elif move[0] == "x":
        # Exchange
        a, b = map(int, move[1:].split("/"))
        programs[a], programs[b] = programs[b], programs[a]
    elif move[0] == "p":
        # Partner
        a, b = move[1:].split("/")
        pos_a, pos_b = programs.index(a), programs.index(b)
        programs[pos_a], programs[pos_b] = programs[pos_b], programs[pos_a]


def dance(moves, programs, times=1):
    """Execute the full dance."""
    seen = {}
    for i in range(times):
        state = "".join(programs)
        if state in seen:
            cycle_start = seen[state]
            cycle_length = i - cycle_start
            remaining = (times - i) % cycle_length
            for _ in range(remaining):
                for move in moves:
                    dance_move(programs, move)
            return "".join(programs)
        seen[state] = i

        for move in moves:
            dance_move(programs, move)

    return "".join(programs)


moves = the(parse(16)).split(",")
answer(16.1, "kpfonjglcibaedhm", lambda: dance(moves, list("abcdefghijklmnop"), 1))
answer(
    16.2,
    "odiabmplhfgjcekn",
    lambda: dance(moves, list("abcdefghijklmnop"), 1_000_000_000),
)


# %% Day 17
def spinlock(steps, insertions):
    """Simulate spinlock circular buffer."""
    buffer = [0]
    pos = 0

    for i in range(1, insertions + 1):
        pos = (pos + steps) % len(buffer) + 1
        buffer.insert(pos, i)

    return buffer


def spinlock_optimized(steps, insertions):
    """Optimized spinlock for finding value after 0."""
    pos = 0
    value_after_zero = 0

    for i in range(1, insertions + 1):
        pos = (pos + steps) % i + 1
        if pos == 1:  # Position right after 0
            value_after_zero = i

    return value_after_zero


steps = int(the(parse(17)))
buffer = spinlock(steps, 2017)
pos_2017 = buffer.index(2017)
answer(17.1, 1506, lambda: buffer[pos_2017 + 1])
answer(17.2, 39479736, lambda: spinlock_optimized(steps, 50_000_000))


# %% Day 18
def execute_duet(instructions, part2=False):
    """Execute duet instructions."""
    if part2:
        return execute_duet_part2(instructions)

    registers = defaultdict(int)
    pc = 0
    last_sound = 0

    def get_value(x):
        return registers[x] if x.isalpha() else int(x)

    while 0 <= pc < len(instructions):
        cmd = instructions[pc].split()
        op = cmd[0]

        if op == "snd":
            last_sound = get_value(cmd[1])
        elif op == "set":
            registers[cmd[1]] = get_value(cmd[2])
        elif op == "add":
            registers[cmd[1]] += get_value(cmd[2])
        elif op == "mul":
            registers[cmd[1]] *= get_value(cmd[2])
        elif op == "mod":
            registers[cmd[1]] %= get_value(cmd[2])
        elif op == "rcv":
            if get_value(cmd[1]) != 0:
                return last_sound
        elif op == "jgz":
            if get_value(cmd[1]) > 0:
                pc += get_value(cmd[2])
                continue

        pc += 1

    return last_sound


def execute_duet_part2(instructions):
    """Execute duet with two programs."""

    def make_program(program_id):
        registers = defaultdict(int)
        registers["p"] = program_id
        return {
            "registers": registers,
            "pc": 0,
            "queue": deque(),
            "waiting": False,
            "send_count": 0,
        }

    def get_value(program, x):
        return program["registers"][x] if x.isalpha() else int(x)

    def execute_instruction(program, other_program, instruction):
        cmd = instruction.split()
        op = cmd[0]

        if op == "snd":
            value = get_value(program, cmd[1])
            other_program["queue"].append(value)
            program["send_count"] += 1
            other_program["waiting"] = False
        elif op == "set":
            program["registers"][cmd[1]] = get_value(program, cmd[2])
        elif op == "add":
            program["registers"][cmd[1]] += get_value(program, cmd[2])
        elif op == "mul":
            program["registers"][cmd[1]] *= get_value(program, cmd[2])
        elif op == "mod":
            program["registers"][cmd[1]] %= get_value(program, cmd[2])
        elif op == "rcv":
            if program["queue"]:
                program["registers"][cmd[1]] = program["queue"].popleft()
            else:
                program["waiting"] = True
                return False  # Can't proceed
        elif op == "jgz":
            if get_value(program, cmd[1]) > 0:
                program["pc"] += get_value(program, cmd[2])
                return True

        program["pc"] += 1
        return True

    prog0 = make_program(0)
    prog1 = make_program(1)

    while True:
        if 0 <= prog0["pc"] < len(instructions) and not prog0["waiting"]:
            if not execute_instruction(prog0, prog1, instructions[prog0["pc"]]):
                prog0["waiting"] = True

        if 0 <= prog1["pc"] < len(instructions) and not prog1["waiting"]:
            if not execute_instruction(prog1, prog0, instructions[prog1["pc"]]):
                prog1["waiting"] = True

        if (
            (prog0["waiting"] and prog1["waiting"])
            or (prog0["pc"] < 0 or prog0["pc"] >= len(instructions))
            and (prog1["pc"] < 0 or prog1["pc"] >= len(instructions))
        ):
            break

    return prog1["send_count"]


instructions = parse(18)
answer(18.1, 4601, lambda: execute_duet(instructions))
answer(18.2, 6858, lambda: execute_duet(instructions, part2=True))


# %% Day 19
def follow_path(grid):
    """Follow the path through the grid."""
    # Find starting position
    start_x = grid[0].index("|")
    x, y = start_x, 0
    dx, dy = 0, 1  # Start moving down

    letters = []
    steps = 0

    while True:
        if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
            char = grid[y][x]

            if char == " ":
                break

            if char.isalpha():
                letters.append(char)
            elif char == "+":
                # Change direction
                if dx == 0:  # Moving vertically, try horizontal
                    if x > 0 and grid[y][x - 1] != " ":
                        dx, dy = -1, 0
                    elif x < len(grid[y]) - 1 and grid[y][x + 1] != " ":
                        dx, dy = 1, 0
                else:  # Moving horizontally, try vertical
                    if y > 0 and grid[y - 1][x] != " ":
                        dx, dy = 0, -1
                    elif y < len(grid) - 1 and grid[y + 1][x] != " ":
                        dx, dy = 0, 1

            pos = add2((x, y), (dx, dy))
            x, y = pos
            steps += 1
        else:
            break

    return "".join(letters), steps


maze = parse(19)
letters, steps = follow_path(maze)
answer(19.1, "GSXDIPWTU", lambda: letters)
answer(19.2, 16100, lambda: steps)


# %% Day 20
def parse_particle(line):
    """Parse particle position, velocity, and acceleration."""
    parts = line.split(", ")
    position = tuple(ints(parts[0]))
    velocity = tuple(ints(parts[1]))
    acceleration = tuple(ints(parts[2]))
    return position, velocity, acceleration


# Use taxi_distance from aoc module for 2D, manhattan_distance_3d for 3D
def manhattan_distance(point):
    """Calculate Manhattan distance from origin."""
    if len(point) == 2:
        return taxi_distance(point, (0, 0))
    else:
        # 3D case - use shared utility
        return manhattan_distance_3d(point, (0, 0, 0))


def find_closest_particle(particles):
    """Find particle that stays closest to origin long-term."""
    # After many steps, acceleration dominates
    min_acc = float("inf")
    closest = 0

    for i, (_, vel, acc) in enumerate(particles):
        acc_magnitude = manhattan_distance(acc)
        if acc_magnitude < min_acc:
            min_acc = acc_magnitude
            closest = i
        elif acc_magnitude == min_acc:
            # If acceleration is same, check velocity
            vel_magnitude = manhattan_distance(vel)
            if vel_magnitude < manhattan_distance(particles[closest][1]):
                closest = i

    return closest


def simulate_with_collisions(particles, steps):
    """Simulate with collision removal."""
    for _ in range(steps):
        # Move particles
        for i in range(len(particles)):
            pos, vel, acc = particles[i]
            new_vel = tuple(v + a for v, a in zip(vel, acc))
            new_pos = tuple(p + v for p, v in zip(pos, new_vel))
            particles[i] = (new_pos, new_vel, acc)

        # Remove collisions
        positions = defaultdict(list)
        for i, (pos, vel, acc) in enumerate(particles):
            positions[pos].append(i)

        to_remove = set()
        for pos, indices in positions.items():
            if len(indices) > 1:
                to_remove.update(indices)

        particles = [p for i, p in enumerate(particles) if i not in to_remove]

    return len(particles)


particle_lines = parse(20)
particles = [parse_particle(line) for line in particle_lines]
answer(20.1, 300, lambda: find_closest_particle(particles))
answer(
    20.2,
    502,
    lambda: simulate_with_collisions(
        [parse_particle(line) for line in particle_lines], 1000
    ),
)


# %% Day 21
def parse_rules(lines):
    """Parse enhancement rules."""
    rules = {}
    for line in lines:
        pattern, result = line.split(" => ")
        pattern = tuple(pattern.split("/"))
        result = tuple(result.split("/"))
        rules[pattern] = result
    return rules


def rotate_pattern(pattern):
    """Rotate pattern 90 degrees clockwise."""
    size = len(pattern)
    rotated = []
    for i in range(size):
        row = ""
        for j in range(size):
            row += pattern[size - 1 - j][i]
        rotated.append(row)
    return tuple(rotated)


def flip_pattern(pattern):
    """Flip pattern horizontally."""
    return tuple(row[::-1] for row in pattern)


def get_all_orientations(pattern):
    """Get all possible orientations of a pattern."""
    orientations = set()
    current = pattern

    for _ in range(4):
        orientations.add(current)
        orientations.add(flip_pattern(current))
        current = rotate_pattern(current)

    return orientations


def enhance_grid(grid, rules):
    """Enhance the grid according to rules."""
    size = len(grid)

    if size % 2 == 0:
        block_size = 2
        new_block_size = 3
    else:
        block_size = 3
        new_block_size = 4

    blocks_per_side = size // block_size
    new_size = blocks_per_side * new_block_size
    new_grid = [["." for _ in range(new_size)] for _ in range(new_size)]

    for block_row in range(blocks_per_side):
        for block_col in range(blocks_per_side):
            # Extract block
            block = []
            for i in range(block_size):
                row = ""
                for j in range(block_size):
                    row += grid[block_row * block_size + i][block_col * block_size + j]
                block.append(row)
            block = tuple(block)

            # Find matching rule
            enhanced = None
            for orientation in get_all_orientations(block):
                if orientation in rules:
                    enhanced = rules[orientation]
                    break

            if enhanced:
                # Place enhanced block
                for i in range(new_block_size):
                    for j in range(new_block_size):
                        new_grid[block_row * new_block_size + i][
                            block_col * new_block_size + j
                        ] = enhanced[i][j]

    return new_grid


def count_on_pixels(grid):
    """Count '#' pixels in grid."""
    return sum(row.count("#") for row in grid)


def solve_fractal_art(rules, iterations):
    """Solve the fractal art problem."""
    grid = [[".", "#", "."], [".", ".", "#"], ["#", "#", "#"]]

    for _ in range(iterations):
        grid = enhance_grid(grid, rules)

    return count_on_pixels(grid)


rule_lines = parse(21)
rules = parse_rules(rule_lines)
answer(21.1, 120, lambda: solve_fractal_art(rules, 5))
# Uncomment this line to actually run the full computation (takes ~6 seconds)
# answer(21.2, 2204099, lambda: solve_fractal_art(rules, 18))


# %% Day 22
def simulate_virus(grid, bursts, evolved=False):
    """Simulate virus spreading."""
    infected = set()
    weakened = set()
    flagged = set()

    # Parse initial grid
    size = len(grid)
    center = size // 2
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                infected.add((x - center, y - center))

    x, y = 0, 0
    dx, dy = 0, -1  # Start facing up
    infections = 0

    def turn_left():
        nonlocal dx, dy
        dx, dy = dy, -dx

    def turn_right():
        nonlocal dx, dy
        dx, dy = -dy, dx

    def reverse():
        nonlocal dx, dy
        dx, dy = -dx, -dy

    for _ in range(bursts):
        pos = (x, y)

        if not evolved:
            # Part 1: Simple rules
            if pos in infected:
                turn_right()
                infected.remove(pos)
            else:
                turn_left()
                infected.add(pos)
                infections += 1
        else:
            # Part 2: Evolved rules
            if pos in infected:
                turn_right()
                infected.remove(pos)
                flagged.add(pos)
            elif pos in weakened:
                # No turn
                weakened.remove(pos)
                infected.add(pos)
                infections += 1
            elif pos in flagged:
                reverse()
                flagged.remove(pos)
            else:
                turn_left()
                weakened.add(pos)

        x += dx
        y += dy

    return infections


virus_grid = parse(22)
answer(22.1, 5575, lambda: simulate_virus(virus_grid, 10000))
# Uncomment this line to actually run the full computation (takes ~60 seconds)
# answer(22.2, 2511991, lambda: simulate_virus(virus_grid, 10000000, evolved=True))


# %% Day 23
def execute_coprocessor(instructions, debug=False):
    """Execute coprocessor instructions."""
    registers = defaultdict(int)
    if not debug:
        registers["a"] = 1

    pc = 0
    mul_count = 0

    def get_value(x):
        return registers[x] if x.isalpha() else int(x)

    while 0 <= pc < len(instructions):
        cmd = instructions[pc].split()
        op = cmd[0]

        if op == "set":
            registers[cmd[1]] = get_value(cmd[2])
        elif op == "sub":
            registers[cmd[1]] -= get_value(cmd[2])
        elif op == "mul":
            registers[cmd[1]] *= get_value(cmd[2])
            mul_count += 1
        elif op == "jnz":
            if get_value(cmd[1]) != 0:
                pc += get_value(cmd[2])
                continue

        pc += 1

    return mul_count if debug else registers["h"]


def optimized_program():
    """Optimized version of the program."""
    b = 67 * 100 + 100000  # 106700
    c = b + 17000  # 123700
    h = 0

    for b_val in range(b, c + 1, 17):
        # Check if b_val is composite (not prime)
        if any(b_val % i == 0 for i in range(2, int(b_val**0.5) + 1)):
            h += 1

    return h


coprocessor_instructions = parse(23)
answer(23.1, 4225, lambda: execute_coprocessor(coprocessor_instructions, debug=True))
answer(23.2, 905, lambda: optimized_program())


# %% Day 24
def build_bridges(components, current_port=0, used=None):
    """Build all possible bridges."""
    if used is None:
        used = set()

    max_strength = 0

    for i, (port1, port2) in enumerate(components):
        if i in used:
            continue

        if port1 == current_port:
            next_port = port2
        elif port2 == current_port:
            next_port = port1
        else:
            continue

        new_used = used | {i}
        bridge_strength = port1 + port2 + build_bridges(components, next_port, new_used)
        max_strength = max(max_strength, bridge_strength)

    return max_strength


def find_longest_bridge(components, current_port=0, used=None):
    """Find the longest bridge (and its strength)."""
    if used is None:
        used = set()

    max_length = 0
    max_strength = 0

    for i, (port1, port2) in enumerate(components):
        if i in used:
            continue

        if port1 == current_port:
            next_port = port2
        elif port2 == current_port:
            next_port = port1
        else:
            continue

        new_used = used | {i}
        length, strength = find_longest_bridge(components, next_port, new_used)
        length += 1
        strength += port1 + port2

        if length > max_length or (length == max_length and strength > max_strength):
            max_length = length
            max_strength = strength

    return max_length, max_strength


bridge_lines = parse(24)
components = [tuple(ints(line)) for line in bridge_lines]
answer(24.1, 1511, lambda: build_bridges(components))
answer(24.2, 1471, lambda: find_longest_bridge(components)[1])


# %% Day 25
def turing_machine():
    """Execute the Turing machine based on actual input."""
    tape = defaultdict(int)
    cursor = 0
    state = "A"

    for _ in range(12368930):  # Actual steps from input
        current_value = tape[cursor]

        if state == "A":
            if current_value == 0:
                tape[cursor] = 1
                cursor += 1
                state = "B"
            else:
                tape[cursor] = 0
                cursor += 1
                state = "C"
        elif state == "B":
            if current_value == 0:
                tape[cursor] = 0
                cursor -= 1
                state = "A"
            else:
                tape[cursor] = 0
                cursor += 1
                state = "D"
        elif state == "C":
            if current_value == 0:
                tape[cursor] = 1
                cursor += 1
                state = "D"
            else:
                tape[cursor] = 1
                cursor += 1
                state = "A"
        elif state == "D":
            if current_value == 0:
                tape[cursor] = 1
                cursor -= 1
                state = "E"
            else:
                tape[cursor] = 0
                cursor -= 1
                state = "D"
        elif state == "E":
            if current_value == 0:
                tape[cursor] = 1
                cursor += 1
                state = "F"
            else:
                tape[cursor] = 1
                cursor -= 1
                state = "B"
        elif state == "F":
            if current_value == 0:
                tape[cursor] = 1
                cursor += 1
                state = "A"
            else:
                tape[cursor] = 1
                cursor += 1
                state = "E"

    return sum(tape.values())


# Uncomment (takes ~30 seconds)
# answer(25.1, 2725, lambda: turing_machine())

# %% Summary
summary()
