#!/usr/bin/env python3

from aoc import answer, paragraphs, parse_year, first, combinations, prod, summary
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

in3 = parse(3, str)


def count_trees(grid, right, down):
    trees = 0
    x = 0
    for y in range(0, len(grid), down):
        if grid[y][x % len(grid[y])] == "#":
            trees += 1
        x += right
    return trees


test_grid = [
    "..##.......",
    "#...#...#..",
    ".#....#..#.",
    "..#.#...#.#",
    ".#...##..#.",
    "..#.##.....",
    ".#.#.#....#",
    ".#........#",
    "#.##...#...",
    "#...##....#",
    ".#..#...#.#",
]

assert count_trees(test_grid, 3, 1) == 7

answer(3.1, 156, lambda: count_trees(in3, 3, 1))

slopes = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
answer(
    3.2,
    3521829480,
    lambda: prod(count_trees(in3, right, down) for right, down in slopes),
)

# %%


def parse_passport(text):
    passport = {}
    for line in text.strip().split("\n"):
        for field in line.split():
            key, value = field.split(":")
            passport[key] = value
    return passport


def validate_passport_fields(passport):
    required = {"byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"}
    return required.issubset(passport.keys())


def validate_passport_values(passport):
    if not validate_passport_fields(passport):
        return False

    try:
        byr = int(passport["byr"])
        if not (1920 <= byr <= 2002):
            return False

        iyr = int(passport["iyr"])
        if not (2010 <= iyr <= 2020):
            return False

        eyr = int(passport["eyr"])
        if not (2020 <= eyr <= 2030):
            return False

        hgt = passport["hgt"]
        if hgt.endswith("cm"):
            height = int(hgt[:-2])
            if not (150 <= height <= 193):
                return False
        elif hgt.endswith("in"):
            height = int(hgt[:-2])
            if not (59 <= height <= 76):
                return False
        else:
            return False

        hcl = passport["hcl"]
        if not (
            len(hcl) == 7
            and hcl[0] == "#"
            and all(c in "0123456789abcdef" for c in hcl[1:])
        ):
            return False

        ecl = passport["ecl"]
        if ecl not in {"amb", "blu", "brn", "gry", "grn", "hzl", "oth"}:
            return False

        pid = passport["pid"]
        if not (len(pid) == 9 and pid.isdigit()):
            return False

        return True
    except (ValueError, KeyError):
        return False


in4 = parse(4, parse_passport, sections=paragraphs)

answer(4.1, 196, lambda: sum(1 for p in in4 if validate_passport_fields(p)))
answer(4.2, 114, lambda: sum(1 for p in in4 if validate_passport_values(p)))

# %%


def seat_id(boarding_pass):
    row = int(boarding_pass[:7].replace("F", "0").replace("B", "1"), 2)
    col = int(boarding_pass[7:].replace("L", "0").replace("R", "1"), 2)
    return row * 8 + col


in5 = parse(5, str)

assert seat_id("FBFBBFFRLR") == 357
assert seat_id("BFFFBBFRRR") == 567
assert seat_id("FFFBBBFRRR") == 119
assert seat_id("BBFFBBFRLL") == 820

answer(5.1, 901, lambda: max(seat_id(bp) for bp in in5))

all_seats = {seat_id(bp) for bp in in5}
answer(
    5.2,
    661,
    lambda: next(
        s for s in range(min(all_seats), max(all_seats)) if s not in all_seats
    ),
)

# %%


def count_anyone_yes(group):
    return len(set("".join(group)))


def count_everyone_yes(group):
    if not group:
        return 0
    return len(set(group[0]).intersection(*group[1:]))


in6 = parse(6, lambda x: x.split("\n"), sections=paragraphs)

answer(6.1, 6947, lambda: sum(count_anyone_yes(group) for group in in6))
answer(6.2, 3398, lambda: sum(count_everyone_yes(group) for group in in6))

# %%


def parse_bag_rule(line):
    parts = line.split(" contain ")
    container = parts[0].replace(" bags", "")

    if "no other bags" in parts[1]:
        return container, []

    contents = []
    for item in parts[1].split(", "):
        item = item.replace(".", "").replace(" bags", "").replace(" bag", "")
        count = int(item.split()[0])
        color = " ".join(item.split()[1:])
        contents.append((count, color))

    return container, contents


def can_contain_gold(rules, bag):
    if bag == "shiny gold":
        return True

    for _, inner_bag in rules.get(bag, []):
        if can_contain_gold(rules, inner_bag):
            return True

    return False


def count_bags_inside(rules, bag):
    total = 0
    for count, inner_bag in rules.get(bag, []):
        total += count * (1 + count_bags_inside(rules, inner_bag))
    return total


in7 = parse(7, parse_bag_rule)
rules = dict(in7)

answer(
    7.1,
    370,
    lambda: sum(
        1 for bag in rules if bag != "shiny gold" and can_contain_gold(rules, bag)
    ),
)
answer(7.2, 29547, lambda: count_bags_inside(rules, "shiny gold"))

# %%


def execute_program(instructions):
    acc = 0
    pc = 0
    visited = set()

    while pc < len(instructions):
        if pc in visited:
            return acc, False

        visited.add(pc)
        op, arg = instructions[pc]

        if op == "acc":
            acc += arg
        elif op == "jmp":
            pc += arg
            continue

        pc += 1

    return acc, True


def parse_instruction(line):
    op, arg = line.split()
    return op, int(arg)


def fix_program(instructions):
    for i, (op, arg) in enumerate(instructions):
        if op == "nop":
            test_instructions = [
                ("jmp", arg) if j == i else instr
                for j, instr in enumerate(instructions)
            ]
            acc, terminated = execute_program(test_instructions)
            if terminated:
                return acc
        elif op == "jmp":
            test_instructions = [
                ("nop", arg) if j == i else instr
                for j, instr in enumerate(instructions)
            ]
            acc, terminated = execute_program(test_instructions)
            if terminated:
                return acc

    return None


in8 = parse(8, parse_instruction)

answer(8.1, 1262, lambda: execute_program(in8)[0])
answer(8.2, 1643, lambda: fix_program(in8))

# %%


def find_invalid_number(numbers, preamble_size):
    for i in range(preamble_size, len(numbers)):
        target = numbers[i]
        preamble = numbers[i - preamble_size : i]

        valid = False
        for j in range(len(preamble)):
            for k in range(j + 1, len(preamble)):
                if preamble[j] + preamble[k] == target:
                    valid = True
                    break
            if valid:
                break

        if not valid:
            return target

    return None


def find_encryption_weakness(numbers, invalid_number):
    for i in range(len(numbers)):
        total = numbers[i]
        for j in range(i + 1, len(numbers)):
            total += numbers[j]
            if total == invalid_number:
                contiguous = numbers[i : j + 1]
                return min(contiguous) + max(contiguous)
            elif total > invalid_number:
                break

    return None


in9 = parse(9, int)

invalid = find_invalid_number(in9, 25)
answer(9.1, 88311122, lambda: invalid)
answer(9.2, 13549369, lambda: find_encryption_weakness(in9, invalid))

# %%


def count_adapter_arrangements(adapters):
    adapters = sorted(adapters)
    adapters = [0] + adapters + [adapters[-1] + 3]

    dp = [0] * len(adapters)
    dp[0] = 1

    for i in range(1, len(adapters)):
        for j in range(i):
            if adapters[i] - adapters[j] <= 3:
                dp[i] += dp[j]

    return dp[-1]


def adapter_differences(adapters):
    adapters = sorted(adapters)
    adapters = [0] + adapters + [adapters[-1] + 3]

    diffs = {1: 0, 2: 0, 3: 0}
    for i in range(1, len(adapters)):
        diff = adapters[i] - adapters[i - 1]
        diffs[diff] += 1

    return diffs[1] * diffs[3]


in10 = parse(10, int)

answer(10.1, 1885, lambda: adapter_differences(in10))
answer(10.2, 2024782584832, lambda: count_adapter_arrangements(in10))

# %%


def simulate_seating(grid, tolerance, use_line_of_sight=False):
    def get_neighbors(current_grid, row, col):
        if use_line_of_sight:
            neighbors = []
            directions = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < len(current_grid) and 0 <= c < len(current_grid[0]):
                    if current_grid[r][c] != ".":
                        neighbors.append(current_grid[r][c])
                        break
                    r, c = r + dr, c + dc
            return neighbors
        else:
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < len(current_grid) and 0 <= c < len(current_grid[0]):
                        neighbors.append(current_grid[r][c])
            return neighbors

    current_grid = [list(row) for row in grid]

    changed = True
    while changed:
        changed = False
        new_grid = [row[:] for row in current_grid]

        for row in range(len(current_grid)):
            for col in range(len(current_grid[0])):
                if current_grid[row][col] == "L":
                    neighbors = get_neighbors(current_grid, row, col)
                    if neighbors.count("#") == 0:
                        new_grid[row][col] = "#"
                        changed = True
                elif current_grid[row][col] == "#":
                    neighbors = get_neighbors(current_grid, row, col)
                    if neighbors.count("#") >= tolerance:
                        new_grid[row][col] = "L"
                        changed = True

        current_grid = new_grid

    return sum(row.count("#") for row in current_grid)


in11 = parse(11, str)

answer(11.1, 2418, lambda: simulate_seating(in11, 4))
answer(11.2, 2144, lambda: simulate_seating(in11, 5, use_line_of_sight=True))

# %%


def execute_navigation(instructions, use_waypoint=False):
    if use_waypoint:
        ship_x, ship_y = 0, 0
        waypoint_x, waypoint_y = 10, 1

        for action, value in instructions:
            if action == "N":
                waypoint_y += value
            elif action == "S":
                waypoint_y -= value
            elif action == "E":
                waypoint_x += value
            elif action == "W":
                waypoint_x -= value
            elif action == "L":
                for _ in range(value // 90):
                    waypoint_x, waypoint_y = -waypoint_y, waypoint_x
            elif action == "R":
                for _ in range(value // 90):
                    waypoint_x, waypoint_y = waypoint_y, -waypoint_x
            elif action == "F":
                ship_x += value * waypoint_x
                ship_y += value * waypoint_y

        return abs(ship_x) + abs(ship_y)
    else:
        x, y = 0, 0
        direction = 0  # 0=E, 1=S, 2=W, 3=N

        for action, value in instructions:
            if action == "N":
                y += value
            elif action == "S":
                y -= value
            elif action == "E":
                x += value
            elif action == "W":
                x -= value
            elif action == "L":
                direction = (direction - value // 90) % 4
            elif action == "R":
                direction = (direction + value // 90) % 4
            elif action == "F":
                if direction == 0:
                    x += value
                elif direction == 1:
                    y -= value
                elif direction == 2:
                    x -= value
                elif direction == 3:
                    y += value

        return abs(x) + abs(y)


def parse_navigation(line):
    return line[0], int(line[1:])


in12 = parse(12, parse_navigation)

answer(12.1, 1631, lambda: execute_navigation(in12))
answer(12.2, 58606, lambda: execute_navigation(in12, use_waypoint=True))

# %%


def find_earliest_bus(timestamp, buses):
    earliest_time = float("inf")
    earliest_bus = None

    for bus in buses:
        if bus == "x":
            continue
        bus_id = int(bus)
        wait_time = bus_id - (timestamp % bus_id)
        if wait_time < earliest_time:
            earliest_time = wait_time
            earliest_bus = bus_id

    return earliest_bus * earliest_time


def find_contest_timestamp(buses):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def chinese_remainder_theorem(remainders, moduli):
        total = 0
        prod = 1
        for m in moduli:
            prod *= m

        for r, m in zip(remainders, moduli):
            p = prod // m
            _, inv, _ = extended_gcd(p, m)
            total += r * inv * p

        return total % prod

    remainders = []
    moduli = []

    for i, bus in enumerate(buses):
        if bus != "x":
            bus = int(bus)
            remainders.append((-i) % bus)
            moduli.append(bus)

    return chinese_remainder_theorem(remainders, moduli)


def parse_bus_schedule(lines):
    timestamp = int(lines[0])
    buses = lines[1].split(",")
    return timestamp, buses


in13 = parse(13, str)
timestamp, buses = parse_bus_schedule(in13)

answer(13.1, 2305, lambda: find_earliest_bus(timestamp, buses))
answer(13.2, 552612234243498, lambda: find_contest_timestamp(buses))

# %%


def apply_mask_v1(mask, value):
    result = value
    for i, bit in enumerate(reversed(mask)):
        if bit == "0":
            result &= ~(1 << i)
        elif bit == "1":
            result |= 1 << i
    return result


def apply_mask_v2(mask, address):
    addresses = []
    floating_bits = []

    for i, bit in enumerate(reversed(mask)):
        if bit == "1":
            address |= 1 << i
        elif bit == "X":
            floating_bits.append(i)

    for i in range(1 << len(floating_bits)):
        addr = address
        for j, bit_pos in enumerate(floating_bits):
            if i & (1 << j):
                addr |= 1 << bit_pos
            else:
                addr &= ~(1 << bit_pos)
        addresses.append(addr)

    return addresses


def run_docking_program(instructions, version=1):
    memory = {}
    mask = None

    for instruction in instructions:
        if instruction.startswith("mask"):
            mask = instruction.split(" = ")[1]
        else:
            parts = instruction.split(" = ")
            address = int(parts[0][4:-1])
            value = int(parts[1])

            if version == 1:
                memory[address] = apply_mask_v1(mask, value)
            else:
                for addr in apply_mask_v2(mask, address):
                    memory[addr] = value

    return sum(memory.values())


in14 = parse(14, str)

answer(14.1, 8471403462063, lambda: run_docking_program(in14))
answer(14.2, 2667858637669, lambda: run_docking_program(in14, version=2))

# %%


def memory_game(starting_numbers, target_turn):
    spoken = {}

    for turn, num in enumerate(starting_numbers[:-1]):
        spoken[num] = turn + 1

    last_spoken = starting_numbers[-1]

    for turn in range(len(starting_numbers) + 1, target_turn + 1):
        if last_spoken in spoken:
            next_number = (turn - 1) - spoken[last_spoken]
        else:
            next_number = 0

        spoken[last_spoken] = turn - 1
        last_spoken = next_number

    return last_spoken


in15 = [11, 18, 0, 20, 1, 7, 16]

answer(15.1, 639, lambda: memory_game(in15, 2020))
answer(15.2, 266, lambda: memory_game(in15, 30000000))

# %%


def parse_ticket_rules(lines):
    rules = {}
    for line in lines:
        name, ranges = line.split(": ")
        range_parts = ranges.split(" or ")
        valid_ranges = []
        for range_part in range_parts:
            start, end = map(int, range_part.split("-"))
            valid_ranges.append((start, end))
        rules[name] = valid_ranges
    return rules


def is_valid_for_any_field(value, rules):
    for field_ranges in rules.values():
        for start, end in field_ranges:
            if start <= value <= end:
                return True
    return False


def find_invalid_values(ticket, rules):
    invalid = []
    for value in ticket:
        if not is_valid_for_any_field(value, rules):
            invalid.append(value)
    return invalid


def is_valid_ticket(ticket, rules):
    return len(find_invalid_values(ticket, rules)) == 0


def solve_field_positions(valid_tickets, rules):
    num_fields = len(valid_tickets[0])
    possible_fields = {}

    for field_name, field_ranges in rules.items():
        possible_fields[field_name] = set()
        for position in range(num_fields):
            valid_for_position = True
            for ticket in valid_tickets:
                value = ticket[position]
                valid_for_field = False
                for start, end in field_ranges:
                    if start <= value <= end:
                        valid_for_field = True
                        break
                if not valid_for_field:
                    valid_for_position = False
                    break
            if valid_for_position:
                possible_fields[field_name].add(position)

    field_positions = {}
    while len(field_positions) < num_fields:
        for field_name, positions in possible_fields.items():
            if field_name not in field_positions and len(positions) == 1:
                position = positions.pop()
                field_positions[field_name] = position
                for other_field in possible_fields:
                    if other_field != field_name:
                        possible_fields[other_field].discard(position)

    return field_positions


def parse_tickets_input(sections):
    rules_lines = sections[0].split("\n")
    rules = parse_ticket_rules(rules_lines)

    my_ticket = list(map(int, sections[1].split("\n")[1].split(",")))

    nearby_tickets = []
    for line in sections[2].split("\n")[1:]:
        if line.strip():
            nearby_tickets.append(list(map(int, line.split(","))))

    return rules, my_ticket, nearby_tickets


in16 = parse(16, str, sections=paragraphs)
rules, my_ticket, nearby_tickets = parse_tickets_input(in16)

error_rate = sum(sum(find_invalid_values(ticket, rules)) for ticket in nearby_tickets)
answer(16.1, 21081, lambda: error_rate)

valid_tickets = [ticket for ticket in nearby_tickets if is_valid_ticket(ticket, rules)]
valid_tickets.append(my_ticket)
field_positions = solve_field_positions(valid_tickets, rules)

departure_product = 1
for field_name, position in field_positions.items():
    if field_name.startswith("departure"):
        departure_product *= my_ticket[position]

answer(16.2, 314360510573, lambda: departure_product)

# %%


def simulate_conway_cubes(initial_state, dimensions=3, cycles=6):
    active = set()

    for y, row in enumerate(initial_state):
        for x, cell in enumerate(row):
            if cell == "#":
                if dimensions == 3:
                    active.add((x, y, 0))
                else:
                    active.add((x, y, 0, 0))

    for _ in range(cycles):
        new_active = set()

        if dimensions == 3:
            candidates = set()
            for x, y, z in active:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            candidates.add((x + dx, y + dy, z + dz))

            for x, y, z in candidates:
                neighbors = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            if (x + dx, y + dy, z + dz) in active:
                                neighbors += 1

                if (x, y, z) in active:
                    if neighbors == 2 or neighbors == 3:
                        new_active.add((x, y, z))
                else:
                    if neighbors == 3:
                        new_active.add((x, y, z))
        else:
            candidates = set()
            for x, y, z, w in active:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            for dw in [-1, 0, 1]:
                                candidates.add((x + dx, y + dy, z + dz, w + dw))

            for x, y, z, w in candidates:
                neighbors = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            for dw in [-1, 0, 1]:
                                if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                                    continue
                                if (x + dx, y + dy, z + dz, w + dw) in active:
                                    neighbors += 1

                if (x, y, z, w) in active:
                    if neighbors == 2 or neighbors == 3:
                        new_active.add((x, y, z, w))
                else:
                    if neighbors == 3:
                        new_active.add((x, y, z, w))

        active = new_active

    return len(active)


in17 = parse(17, str)

answer(17.1, 280, lambda: simulate_conway_cubes(in17))
answer(17.2, 1696, lambda: simulate_conway_cubes(in17, dimensions=4))

# %%


def tokenize_expression(expr):
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(int(expr[i:j]))
            i = j
        elif expr[i] in "+*()":
            tokens.append(expr[i])
            i += 1
        else:
            i += 1
    return tokens


def evaluate_expression_v1(tokens):
    def parse_expression(pos):
        left, pos = parse_term(pos)
        while pos < len(tokens) and tokens[pos] in ["+", "*"]:
            op = tokens[pos]
            pos += 1
            right, pos = parse_term(pos)
            if op == "+":
                left = left + right
            else:
                left = left * right
        return left, pos

    def parse_term(pos):
        if tokens[pos] == "(":
            pos += 1
            result, pos = parse_expression(pos)
            pos += 1  # skip ')'
            return result, pos
        else:
            return tokens[pos], pos + 1

    result, _ = parse_expression(0)
    return result


def evaluate_expression_v2(tokens):
    def parse_expression(pos):
        left, pos = parse_addition(pos)
        while pos < len(tokens) and tokens[pos] == "*":
            pos += 1
            right, pos = parse_addition(pos)
            left = left * right
        return left, pos

    def parse_addition(pos):
        left, pos = parse_term(pos)
        while pos < len(tokens) and tokens[pos] == "+":
            pos += 1
            right, pos = parse_term(pos)
            left = left + right
        return left, pos

    def parse_term(pos):
        if tokens[pos] == "(":
            pos += 1
            result, pos = parse_expression(pos)
            pos += 1  # skip ')'
            return result, pos
        else:
            return tokens[pos], pos + 1

    result, _ = parse_expression(0)
    return result


def solve_math_homework(expressions, version=1):
    total = 0
    for expr in expressions:
        tokens = tokenize_expression(expr)
        if version == 1:
            result = evaluate_expression_v1(tokens)
        else:
            result = evaluate_expression_v2(tokens)
        total += result
    return total


in18 = parse(18, str)

answer(18.1, 7293529867931, lambda: solve_math_homework(in18))
answer(18.2, 60807587180737, lambda: solve_math_homework(in18, version=2))

# %%


def parse_grammar_rules(lines):
    rules = {}
    for line in lines:
        rule_id, rule_def = line.split(": ")
        rule_id = int(rule_id)
        if '"' in rule_def:
            rules[rule_id] = rule_def.strip('"')
        else:
            alternatives = []
            for alt in rule_def.split(" | "):
                alternatives.append([int(x) for x in alt.split()])
            rules[rule_id] = alternatives
    return rules


def generate_regex_pattern(rules, rule_id, memo=None):
    if memo is None:
        memo = {}

    if rule_id in memo:
        return memo[rule_id]

    rule = rules[rule_id]

    if isinstance(rule, str):
        pattern = rule
    else:
        alternatives = []
        for alt in rule:
            parts = []
            for sub_rule in alt:
                parts.append(generate_regex_pattern(rules, sub_rule, memo))
            alternatives.append("".join(parts))

        if len(alternatives) == 1:
            pattern = alternatives[0]
        else:
            pattern = "(" + "|".join(alternatives) + ")"

    memo[rule_id] = pattern
    return pattern


def count_matching_messages(messages, rules):
    import re

    pattern = "^" + generate_regex_pattern(rules, 0) + "$"
    regex = re.compile(pattern)
    return sum(1 for msg in messages if regex.match(msg))


def count_matching_messages_v2(messages, rules):
    import re

    rules_copy = rules.copy()
    rules_copy[8] = [[42], [42, 8]]
    rules_copy[11] = [[42, 31], [42, 11, 31]]

    pattern_42 = generate_regex_pattern(rules, 42)
    pattern_31 = generate_regex_pattern(rules, 31)

    count = 0
    for msg in messages:
        for n in range(1, 10):
            pattern = f"^({pattern_42})+({pattern_42}){{{n}}}({pattern_31}){{{n}}}$"
            if re.match(pattern, msg):
                count += 1
                break

    return count


def parse_messages_input(sections):
    rules_lines = sections[0].split("\n")
    rules = parse_grammar_rules(rules_lines)

    messages = sections[1].split("\n")

    return rules, messages


in19 = parse(19, str, sections=paragraphs)
rules, messages = parse_messages_input(in19)

answer(19.1, 120, lambda: count_matching_messages(messages, rules))
answer(19.2, 350, lambda: count_matching_messages_v2(messages, rules))

# %%


def parse_tile(tile_text):
    lines = tile_text.split("\n")
    tile_id = int(lines[0].split()[1][:-1])
    grid = [list(line) for line in lines[1:]]
    return tile_id, grid


def get_edges(grid):
    top = "".join(grid[0])
    bottom = "".join(grid[-1])
    left = "".join(row[0] for row in grid)
    right = "".join(row[-1] for row in grid)
    return [top, right, bottom, left]


def get_all_orientations(grid):
    orientations = []
    current = grid

    for _ in range(4):
        orientations.append([row[:] for row in current])
        current = [list(row) for row in zip(*current[::-1])]

    flipped = [row[::-1] for row in grid]
    current = flipped

    for _ in range(4):
        orientations.append([row[:] for row in current])
        current = [list(row) for row in zip(*current[::-1])]

    return orientations


def solve_jigsaw(tiles):
    tile_dict = {}
    for tile_id, grid in tiles:
        tile_dict[tile_id] = grid

    # Find all unique edges and their reverse
    all_edges = set()
    tile_edges = {}

    for tile_id, grid in tiles:
        edges = get_edges(grid)
        tile_edges[tile_id] = edges
        for edge in edges:
            all_edges.add(edge)
            all_edges.add(edge[::-1])  # Add reverse

    # Count how many tiles each edge appears in
    edge_counts = {}
    for tile_id, grid in tiles:
        for orientation in get_all_orientations(grid):
            edges = get_edges(orientation)
            for edge in edges:
                if edge not in edge_counts:
                    edge_counts[edge] = set()
                edge_counts[edge].add(tile_id)

    # Corner tiles have exactly 2 edges that don't match any other tile
    corner_tiles = []
    for tile_id, grid in tiles:
        edges = get_edges(grid)
        unmatched_edges = 0
        for edge in edges:
            # Check if this edge matches any other tile
            matched = False
            for other_tile_id, other_grid in tiles:
                if other_tile_id == tile_id:
                    continue
                for other_orientation in get_all_orientations(other_grid):
                    other_edges = get_edges(other_orientation)
                    if edge in other_edges or edge[::-1] in other_edges:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                unmatched_edges += 1

        if unmatched_edges == 2:
            corner_tiles.append(tile_id)

    return corner_tiles


def assemble_jigsaw(tiles):
    import math

    tile_dict = {}
    for tile_id, grid in tiles:
        tile_dict[tile_id] = grid

    grid_size = int(math.sqrt(len(tiles)))

    # Find all possible edge matches
    def edges_match(edge1, edge2):
        return edge1 == edge2 or edge1 == edge2[::-1]

    # Find corners first
    corner_tiles = solve_jigsaw(tiles)

    # Start with a corner tile and find its correct orientation
    start_tile = corner_tiles[0]
    start_orientation = None

    for orientation in get_all_orientations(tile_dict[start_tile]):
        edges = get_edges(orientation)

        # Count how many edges have matches
        matches = [0, 0, 0, 0]
        for i, edge in enumerate(edges):
            for tile_id, grid in tiles:
                if tile_id == start_tile:
                    continue
                for other_orientation in get_all_orientations(grid):
                    other_edges = get_edges(other_orientation)
                    for other_edge in other_edges:
                        if edges_match(edge, other_edge):
                            matches[i] += 1
                            break

        # Corner should have exactly 2 matching edges
        if matches[0] == 0 and matches[3] == 0:  # Top and left are outer
            start_orientation = orientation
            break

    # Build the puzzle using backtracking
    puzzle = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    used_tiles = set()

    def place_tile(row, col):
        if row == grid_size:
            return True

        next_row, next_col = (row, col + 1) if col + 1 < grid_size else (row + 1, 0)

        if row == 0 and col == 0:
            # Place the starting tile
            puzzle[row][col] = (start_tile, start_orientation)
            used_tiles.add(start_tile)
            return place_tile(next_row, next_col)

        # Try each unused tile
        for tile_id, grid in tiles:
            if tile_id in used_tiles:
                continue

            for orientation in get_all_orientations(grid):
                edges = get_edges(orientation)

                # Check constraints
                valid = True

                # Top constraint
                if row > 0:
                    _, top_orientation = puzzle[row - 1][col]
                    top_edges = get_edges(top_orientation)
                    if not edges_match(edges[0], top_edges[2]):
                        valid = False

                # Left constraint
                if col > 0 and valid:
                    _, left_orientation = puzzle[row][col - 1]
                    left_edges = get_edges(left_orientation)
                    if not edges_match(edges[3], left_edges[1]):
                        valid = False

                if valid:
                    puzzle[row][col] = (tile_id, orientation)
                    used_tiles.add(tile_id)

                    if place_tile(next_row, next_col):
                        return True

                    # Backtrack
                    puzzle[row][col] = None
                    used_tiles.remove(tile_id)

        return False

    # Solve the puzzle
    if not place_tile(0, 0):
        raise ValueError("Could not solve the puzzle")

    # Assemble the final image
    tile_size = len(tile_dict[start_tile]) - 2  # Remove borders
    image_size = grid_size * tile_size
    final_image = [["."] * image_size for _ in range(image_size)]

    for row in range(grid_size):
        for col in range(grid_size):
            tile_entry = puzzle[row][col]
            if tile_entry is not None:
                tile_id, orientation = tile_entry
                borderless_tile = remove_borders(orientation)

                for tr in range(tile_size):
                    for tc in range(tile_size):
                        final_image[row * tile_size + tr][col * tile_size + tc] = (
                            borderless_tile[tr][tc]
                        )

    return final_image


def remove_borders(grid):
    return [row[1:-1] for row in grid[1:-1]]


def find_sea_monsters(image):
    monster_pattern = [
        "                  # ",
        "#    ##    ##    ###",
        " #  #  #  #  #  #   ",
    ]

    monster_positions = []
    for row in range(len(image) - 2):
        for col in range(len(image[0]) - 19):
            is_monster = True
            for mr, monster_row in enumerate(monster_pattern):
                for mc, char in enumerate(monster_row):
                    if char == "#" and image[row + mr][col + mc] != "#":
                        is_monster = False
                        break
                if not is_monster:
                    break
            if is_monster:
                monster_positions.append((row, col))

    return monster_positions


def count_water_roughness(image):
    # Convert to list of strings for easier manipulation
    image_strings = ["".join(row) for row in image]

    for orientation in get_all_orientations(image_strings):
        monsters = find_sea_monsters(orientation)
        if monsters:
            total_hash = sum(row.count("#") for row in orientation)
            monster_hash = len(monsters) * 15  # Each monster has 15 # symbols
            return total_hash - monster_hash

    return sum(row.count("#") for row in image_strings)


in20 = parse(20, parse_tile, sections=paragraphs)

corner_tiles = solve_jigsaw(in20)
corner_product = 1
for tile_id in corner_tiles:
    corner_product *= tile_id

answer(20.1, 5966506063747, lambda: corner_product)

# Part 2: Assemble the image and find sea monsters
assembled_image = assemble_jigsaw(in20)
water_roughness = count_water_roughness(assembled_image)
answer(20.2, 1714, lambda: water_roughness)

# %%


def parse_food(line):
    ingredients, allergens = line.split(" (contains ")
    return set(ingredients.split()), set(allergens.rstrip(")").split(", "))


def solve_allergens(foods):
    possible = {}
    for ingredients, allergens in foods:
        for allergen in allergens:
            if allergen in possible:
                possible[allergen] &= ingredients
            else:
                possible[allergen] = ingredients.copy()

    # Constraint satisfaction
    assigned = {}
    while possible:
        for allergen, ingredients in possible.items():
            if len(ingredients) == 1:
                ingredient = ingredients.pop()
                assigned[allergen] = ingredient
                # Remove from all other possibilities
                for other_allergen in possible:
                    if other_allergen != allergen:
                        possible[other_allergen].discard(ingredient)
                break
        else:
            break
        del possible[allergen]

    return assigned


in21 = parse(21, parse_food)
allergen_map = solve_allergens(in21)
dangerous = set(allergen_map.values())
safe_count = sum(len(ingredients - dangerous) for ingredients, _ in in21)

answer(21.1, 2595, lambda: safe_count)

canonical = ",".join(allergen_map[allergen] for allergen in sorted(allergen_map))
answer(21.2, "thvm,jmdg,qrsczjv,hlmvqh,zmb,mrfxh,ckqq,zrgzf", lambda: canonical)

# %%


def play_combat(deck1, deck2):
    while deck1 and deck2:
        card1, card2 = deck1.pop(0), deck2.pop(0)
        if card1 > card2:
            deck1.extend([card1, card2])
        else:
            deck2.extend([card2, card1])
    return deck1 or deck2


def play_recursive_combat(deck1, deck2):
    seen = set()
    while deck1 and deck2:
        state = (tuple(deck1), tuple(deck2))
        if state in seen:
            return deck1, True  # Player 1 wins
        seen.add(state)

        card1, card2 = deck1.pop(0), deck2.pop(0)

        if len(deck1) >= card1 and len(deck2) >= card2:
            # Recursive game
            _, p1_wins = play_recursive_combat(deck1[:card1], deck2[:card2])
            if p1_wins:
                deck1.extend([card1, card2])
            else:
                deck2.extend([card2, card1])
        else:
            # Regular round
            if card1 > card2:
                deck1.extend([card1, card2])
            else:
                deck2.extend([card2, card1])

    return (deck1, True) if deck1 else (deck2, False)


def score_deck(deck):
    return sum(i * card for i, card in enumerate(reversed(deck), 1))


def parse_decks(sections):
    return [list(map(int, section.split("\n")[1:])) for section in sections]


in22 = parse(22, str, sections=paragraphs)
deck1, deck2 = parse_decks(in22)

winner = play_combat(deck1[:], deck2[:])
answer(22.1, 31754, lambda: score_deck(winner))

winner, _ = play_recursive_combat(deck1[:], deck2[:])
answer(22.2, 35436, lambda: score_deck(winner))

# %%


def play_cups(cups, moves):
    n = len(cups)
    # Convert to circular linked list representation
    next_cup = [0] * (n + 1)
    for i in range(n):
        next_cup[cups[i]] = cups[(i + 1) % n]

    current = cups[0]
    for _ in range(moves):
        # Pick up three cups
        pick1 = next_cup[current]
        pick2 = next_cup[pick1]
        pick3 = next_cup[pick2]

        # Remove them from the circle
        next_cup[current] = next_cup[pick3]

        # Find destination
        dest = current - 1
        if dest < 1:
            dest = n
        while dest in (pick1, pick2, pick3):
            dest -= 1
            if dest < 1:
                dest = n

        # Insert picked cups after destination
        next_cup[pick3] = next_cup[dest]
        next_cup[dest] = pick1

        current = next_cup[current]

    return next_cup


def cups_after_1(next_cup):
    result = []
    cup = next_cup[1]
    while cup != 1:
        result.append(str(cup))
        cup = next_cup[cup]
    return "".join(result)


cups = [3, 6, 4, 2, 9, 7, 5, 8, 1]
next_cup = play_cups(cups, 100)
answer(23.1, "47382659", lambda: cups_after_1(next_cup))

# Part 2: Million cups
million_cups = cups + list(range(10, 1000001))
next_cup = play_cups(million_cups, 10000000)
cup1 = next_cup[1]
cup2 = next_cup[cup1]
answer(23.2, 42271866720, lambda: cup1 * cup2)

# %%


def parse_hex_dir(line):
    """Parse hex directions: e, se, sw, w, nw, ne"""
    directions = []
    i = 0
    while i < len(line):
        if line[i] in "ew":
            directions.append(line[i])
            i += 1
        else:
            directions.append(line[i : i + 2])
            i += 2
    return directions


# Hex grid coordinates (using axial coordinates)
hex_dirs = {
    "e": (1, 0),
    "w": (-1, 0),
    "ne": (0, 1),
    "sw": (0, -1),
    "nw": (-1, 1),
    "se": (1, -1),
}


def hex_neighbors(pos):
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in hex_dirs.values()]


def follow_path(directions):
    x, y = 0, 0
    for direction in directions:
        dx, dy = hex_dirs[direction]
        x, y = x + dx, y + dy
    return x, y


def flip_tiles(paths):
    black = set()
    for path in paths:
        tile = follow_path(path)
        if tile in black:
            black.remove(tile)
        else:
            black.add(tile)
    return black


def daily_flip(black_tiles):
    """Game of Life for hex tiles"""
    candidates = set(black_tiles)
    for tile in black_tiles:
        candidates.update(hex_neighbors(tile))

    new_black = set()
    for tile in candidates:
        black_neighbors = sum(
            1 for neighbor in hex_neighbors(tile) if neighbor in black_tiles
        )

        if tile in black_tiles:
            if black_neighbors == 1 or black_neighbors == 2:
                new_black.add(tile)
        else:
            if black_neighbors == 2:
                new_black.add(tile)

    return new_black


in24 = parse(24, parse_hex_dir)
black_tiles = flip_tiles(in24)
answer(24.1, 293, lambda: len(black_tiles))

# Simulate 100 days
for _ in range(100):
    black_tiles = daily_flip(black_tiles)
answer(24.2, 3967, lambda: len(black_tiles))

# %%


def transform_subject(subject, loop_size):
    value = 1
    for _ in range(loop_size):
        value = (value * subject) % 20201227
    return value


def find_loop_size(target, subject=7):
    value = 1
    loop_size = 0
    while value != target:
        value = (value * subject) % 20201227
        loop_size += 1
    return loop_size


# Input: card public key and door public key
card_pub, door_pub = 335121, 363891

card_loop = find_loop_size(card_pub)
encryption_key = transform_subject(door_pub, card_loop)

answer(25.1, 9420461, lambda: encryption_key)
answer(25.2, "Merry Christmas!", lambda: "Merry Christmas!")

# %%

# %% Summary
summary()
