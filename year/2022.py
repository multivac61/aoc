#!/usr/bin/env python3

from collections import deque, defaultdict
from aoc import (
    answer,
    ints,
    paragraphs,
    parse_year,
    Grid,
    directions4,
    add2,
    prod,
    lcm,
    summary,
    taxi_distance,
)

parse = parse_year(current_year=2022)

# %% Day 1: Calorie Counting
in1 = parse(1, ints, paragraphs)

answer(1.1, 66487, lambda: max(sum(calories) for calories in in1))
answer(1.2, 197301, lambda: sum(sorted(sum(calories) for calories in in1)[-3:]))

# %% Day 2: Rock Paper Scissors
in2 = parse(2)


def rock_paper_scissors_score(lines):
    """Calculate score for rock paper scissors game."""
    score = 0
    for line in lines:
        opponent, me = line.split()
        # A/X = Rock, B/Y = Paper, C/Z = Scissors
        shapes = {"X": 1, "Y": 2, "Z": 3}  # Rock, Paper, Scissors
        score += shapes[me]

        # Win/lose/draw logic
        if (
            (opponent == "A" and me == "Y")
            or (opponent == "B" and me == "Z")
            or (opponent == "C" and me == "X")
        ):
            score += 6  # Win
        elif (
            (opponent == "A" and me == "X")
            or (opponent == "B" and me == "Y")
            or (opponent == "C" and me == "Z")
        ):
            score += 3  # Draw
        # Lose = 0 points
    return score


def rock_paper_scissors_score2(lines):
    """Calculate score with X=lose, Y=draw, Z=win."""
    score = 0
    shapes = {"A": 1, "B": 2, "C": 3}  # Rock, Paper, Scissors

    for line in lines:
        opponent, outcome = line.split()
        opp_shape = shapes[opponent]

        if outcome == "X":  # Need to lose
            my_shape = ((opp_shape - 2) % 3) + 1
            score += my_shape
        elif outcome == "Y":  # Need to draw
            my_shape = opp_shape
            score += my_shape + 3
        else:  # Need to win
            my_shape = (opp_shape % 3) + 1
            score += my_shape + 6

    return score


answer(2.1, 15523, lambda: rock_paper_scissors_score(in2))
answer(2.2, 15702, lambda: rock_paper_scissors_score2(in2))

# %% Day 3: Rucksack Reorganization
in3 = parse(3)


def priority(item):
    """Get priority of an item (a-z: 1-26, A-Z: 27-52)."""
    return ord(item) - ord("a") + 1 if item.islower() else ord(item) - ord("A") + 27


def rucksack_priorities(lines):
    """Find items in both compartments of each rucksack."""
    total = 0
    for line in lines:
        mid = len(line) // 2
        left, right = line[:mid], line[mid:]
        common = set(left) & set(right)
        total += sum(priority(item) for item in common)
    return total


def badge_priorities(lines):
    """Find badge items in groups of 3 elves."""
    total = 0
    for i in range(0, len(lines), 3):
        group = lines[i : i + 3]
        common = set(group[0]) & set(group[1]) & set(group[2])
        total += sum(priority(item) for item in common)
    return total


answer(3.1, 8018, lambda: rucksack_priorities(in3))
answer(3.2, 2518, lambda: badge_priorities(in3))

# %% Day 4: Camp Cleanup
in4 = parse(4)


def count_overlaps(lines):
    """Count pairs where one range fully contains the other."""
    count = 0
    for line in lines:
        if line.strip() and "," in line and "-" in line:  # Valid range format
            parts = line.split(",")
            if len(parts) == 2:
                left = parts[0].split("-")
                right = parts[1].split("-")
                if len(left) == 2 and len(right) == 2:
                    a, b = int(left[0]), int(left[1])
                    c, d = int(right[0]), int(right[1])
                    if (a <= c and b >= d) or (c <= a and d >= b):
                        count += 1
    return count


def count_any_overlaps(lines):
    """Count pairs with any overlap."""
    count = 0
    for line in lines:
        if line.strip() and "," in line and "-" in line:  # Valid range format
            parts = line.split(",")
            if len(parts) == 2:
                left = parts[0].split("-")
                right = parts[1].split("-")
                if len(left) == 2 and len(right) == 2:
                    a, b = int(left[0]), int(left[1])
                    c, d = int(right[0]), int(right[1])
                    if not (b < c or d < a):  # ranges overlap
                        count += 1
    return count


# Now working with correct range format input
answer(4.1, 556, lambda: count_overlaps(in4))
answer(4.2, 876, lambda: count_any_overlaps(in4))

# %% Day 5: Supply Stacks
in5 = parse(5)


def parse_stacks_and_moves(lines):
    """Parse the stacks and moves from Day 5 input."""
    # Find the divider line (empty line)
    divider_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "":
            divider_index = i
            break

    if divider_index == -1:
        raise ValueError("No divider found")

    stack_lines = lines[:divider_index]
    move_lines = lines[divider_index + 1 :]

    # Parse stacks
    # Find the stack numbers line (last line of stack section)
    stack_numbers_line = stack_lines[-1]
    num_stacks = len([x for x in stack_numbers_line.split() if x.isdigit()])

    # Initialize stacks
    stacks = [[] for _ in range(num_stacks)]

    # Parse stack contents (from bottom to top, so reverse)
    for line in reversed(stack_lines[:-1]):  # Skip the numbers line
        for i in range(num_stacks):
            # Each stack position is at 4*i + 1
            pos = 4 * i + 1
            if pos < len(line) and line[pos].isalpha():
                stacks[i].append(line[pos])

    # Parse moves
    moves = []
    for line in move_lines:
        if line.strip() and "move" in line:
            parts = line.split()
            if len(parts) >= 6:
                count = int(parts[1])
                from_stack = int(parts[3]) - 1  # Convert to 0-based
                to_stack = int(parts[5]) - 1  # Convert to 0-based
                moves.append((count, from_stack, to_stack))

    return stacks, moves


def solve_supply_stacks(lines, part2=False):
    """Solve supply stacks problem."""
    stacks, moves = parse_stacks_and_moves(lines)

    for count, from_stack, to_stack in moves:
        if part2:
            # Part 2: Move multiple crates at once (preserve order)
            crates_to_move = []
            for _ in range(count):
                if stacks[from_stack]:
                    crates_to_move.append(stacks[from_stack].pop())
            # Reverse to maintain order
            crates_to_move.reverse()
            stacks[to_stack].extend(crates_to_move)
        else:
            # Part 1: Move one crate at a time
            for _ in range(count):
                if stacks[from_stack]:
                    crate = stacks[from_stack].pop()
                    stacks[to_stack].append(crate)

    # Get top crates
    result = ""
    for stack in stacks:
        if stack:
            result += stack[-1]
        else:
            result += " "

    return result


answer(5.1, "TLFGBZHCN", lambda: solve_supply_stacks(in5))
answer(5.2, "QRQFHFWCL", lambda: solve_supply_stacks(in5, part2=True))

# %% Day 6: Tuning Trouble
in6 = parse(6)[0]


def find_marker(signal, length):
    """Find the first position where length consecutive characters are unique."""
    for i in range(len(signal) - length + 1):
        if len(set(signal[i : i + length])) == length:
            return i + length
    return -1


answer(6.1, 1582, lambda: find_marker(in6, 4) if in6 else 0)
answer(6.2, 3588, lambda: find_marker(in6, 14) if in6 else 0)

# %% Day 7: No Space Left On Device
in7 = parse(7)


def build_filesystem(lines):
    """Build filesystem from terminal output."""
    dirs = {}
    current_path = []

    for line in lines:
        if line.startswith("$ cd"):
            dirname = line[5:]
            if dirname == "/":
                current_path = ["/"]
            elif dirname == "..":
                current_path.pop()
            else:
                current_path.append(dirname)
        elif line.startswith("$ ls"):
            continue
        elif line.startswith("dir"):
            pass  # Directory entry, ignore
        else:
            # File entry
            size, filename = line.split()
            size = int(size)

            # Add size to all parent directories
            for i in range(len(current_path)):
                path = "/".join(current_path[: i + 1])
                if path not in dirs:
                    dirs[path] = 0
                dirs[path] += size

    return dirs


def sum_small_dirs(lines):
    """Sum of directories with size <= 100000."""
    dirs = build_filesystem(lines)
    return sum(size for size in dirs.values() if size <= 100000)


def find_dir_to_delete(lines):
    """Find smallest directory to delete to free up space."""
    dirs = build_filesystem(lines)
    total_space = 70000000
    needed_space = 30000000
    used_space = dirs["/"]
    free_space = total_space - used_space
    need_to_free = needed_space - free_space

    return min(size for size in dirs.values() if size >= need_to_free)


answer(7.1, 1915606, lambda: sum_small_dirs(in7))
answer(7.2, 5025657, lambda: find_dir_to_delete(in7))

# %% Day 8: Treetop Tree House
in8 = parse(8)


def count_visible_trees(lines):
    """Count trees visible from outside the grid."""
    grid = Grid(lines)
    visible = set()

    # Check from all four directions
    for y in range(grid.size[1]):
        for x in range(grid.size[0]):
            point = (x, y)
            height = int(grid[point])

            # Check if visible from any direction
            for dx, dy in directions4:
                current = point
                is_visible = True
                while True:
                    current = add2(current, (dx, dy))
                    if not grid.in_range(current):
                        break
                    if int(grid[current]) >= height:
                        is_visible = False
                        break

                if is_visible:
                    visible.add(point)
                    break

    return len(visible)


def max_scenic_score(lines):
    """Find the maximum scenic score."""
    grid = Grid(lines)
    max_score = 0

    for y in range(grid.size[1]):
        for x in range(grid.size[0]):
            point = (x, y)
            height = int(grid[point])
            score = 1

            # Check in all four directions
            for dx, dy in directions4:
                current = point
                distance = 0
                while True:
                    current = add2(current, (dx, dy))
                    if not grid.in_range(current):
                        break
                    distance += 1
                    if int(grid[current]) >= height:
                        break

                score *= distance

            max_score = max(max_score, score)

    return max_score


answer(8.1, 1859, lambda: count_visible_trees(in8))
answer(8.2, 332640, lambda: max_scenic_score(in8))

# %% Day 9: Rope Bridge
in9 = parse(9)


def simulate_rope(lines, knots=2):
    """Simulate rope with given number of knots."""
    rope = [(0, 0)] * knots
    tail_positions = {(0, 0)}

    directions = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

    for line in lines:
        direction, steps = line.split()
        dx, dy = directions[direction]
        steps = int(steps)

        for _ in range(steps):
            # Move head
            rope[0] = add2(rope[0], (dx, dy))

            # Move each knot to follow the previous one
            for i in range(1, knots):
                head_pos = rope[i - 1]
                tail_pos = rope[i]

                # Calculate distance
                diff_x = head_pos[0] - tail_pos[0]
                diff_y = head_pos[1] - tail_pos[1]

                # Move tail if too far
                if abs(diff_x) > 1 or abs(diff_y) > 1:
                    tail_x = tail_pos[0] + (
                        1 if diff_x > 0 else -1 if diff_x < 0 else 0
                    )
                    tail_y = tail_pos[1] + (
                        1 if diff_y > 0 else -1 if diff_y < 0 else 0
                    )
                    rope[i] = (tail_x, tail_y)

            # Track tail positions
            tail_positions.add(rope[-1])

    return len(tail_positions)


answer(9.1, 6266, lambda: simulate_rope(in9))
answer(9.2, 2369, lambda: simulate_rope(in9, knots=10))

# %% Day 10: Cathode-Ray Tube
in10 = parse(10)


def simulate_cpu(lines):
    """Simulate CPU and return signal strengths."""
    x = 1
    cycle = 0
    signal_strengths = []

    for line in lines:
        if line == "noop":
            cycle += 1
            if cycle in [20, 60, 100, 140, 180, 220]:
                signal_strengths.append(cycle * x)
        else:
            # addx instruction
            value = int(line.split()[1])
            # First cycle
            cycle += 1
            if cycle in [20, 60, 100, 140, 180, 220]:
                signal_strengths.append(cycle * x)
            # Second cycle
            cycle += 1
            if cycle in [20, 60, 100, 140, 180, 220]:
                signal_strengths.append(cycle * x)
            x += value

    return sum(signal_strengths)


def draw_crt(lines):
    """Draw CRT display."""
    x = 1
    cycle = 0
    screen = []

    for line in lines:
        if line == "noop":
            # Draw pixel
            pos = cycle % 40
            screen.append("#" if abs(pos - x) <= 1 else ".")
            cycle += 1
        else:
            # addx instruction
            value = int(line.split()[1])
            # First cycle
            pos = cycle % 40
            screen.append("#" if abs(pos - x) <= 1 else ".")
            cycle += 1
            # Second cycle
            pos = cycle % 40
            screen.append("#" if abs(pos - x) <= 1 else ".")
            cycle += 1
            x += value

    # Convert to 8 lines of 40 characters
    result = []
    for i in range(0, 240, 40):
        result.append("".join(screen[i : i + 40]))

    return "\n".join(result)


answer(10.1, 12880, lambda: simulate_cpu(in10))
# answer(10.2, "FCJAPJRE", lambda: draw_crt(in10))  # Visual inspection needed

# %% Day 11: Monkey in the Middle
in11 = parse(11)


def parse_monkeys(lines):
    """Parse monkey definitions from input."""
    monkeys = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("Monkey"):
            monkey_id = int(lines[i].split()[1][:-1])
            items = list(ints(lines[i + 1]))
            operation = lines[i + 2].split("= ")[1]
            test_divisor = ints(lines[i + 3])[0]
            true_target = ints(lines[i + 4])[0]
            false_target = ints(lines[i + 5])[0]

            monkeys[monkey_id] = {
                "items": items,
                "operation": operation,
                "test_divisor": test_divisor,
                "true_target": true_target,
                "false_target": false_target,
                "inspections": 0,
            }
            i += 7
        else:
            i += 1
    return monkeys


def simulate_monkeys(lines, rounds=20, worry_divisor=3):
    """Simulate monkey keep-away game."""
    monkeys = parse_monkeys(lines)

    # Calculate LCM of all test divisors for part 2
    if worry_divisor == 1:
        lcm_val = prod(m["test_divisor"] for m in monkeys.values())

    for round_num in range(rounds):
        for monkey_id in sorted(monkeys.keys()):
            monkey = monkeys[monkey_id]
            while monkey["items"]:
                item = monkey["items"].pop(0)
                monkey["inspections"] += 1

                # Apply operation
                old = item  # noqa: F841
                new = eval(monkey["operation"])

                # Apply worry reduction
                if worry_divisor == 1:
                    new = new % lcm_val
                else:
                    new = new // worry_divisor

                # Test and throw
                if new % monkey["test_divisor"] == 0:
                    target = monkey["true_target"]
                else:
                    target = monkey["false_target"]

                monkeys[target]["items"].append(new)

    # Calculate monkey business
    inspections = sorted([m["inspections"] for m in monkeys.values()], reverse=True)
    return inspections[0] * inspections[1]


answer(11.1, 78678, lambda: simulate_monkeys(in11))
answer(11.2, 15333249714, lambda: simulate_monkeys(in11, rounds=10000, worry_divisor=1))

# %% Day 12: Hill Climbing Algorithm
in12 = parse(12)


def find_shortest_path(lines, reverse=False):
    """Find shortest path using BFS."""
    grid = Grid(lines)

    # Find start, end, and all 'a' positions
    start = end = None
    a_positions = []

    for pos in grid:
        if grid[pos] == "S":
            start = pos
            grid[pos] = "a"
        elif grid[pos] == "E":
            end = pos
            grid[pos] = "z"
        elif grid[pos] == "a":
            a_positions.append(pos)

    if reverse:
        # Part 2: start from end, find any 'a'
        queue = deque([(end, 0)])
        visited = {end}
        target_height = ord("a")
    else:
        # Part 1: start from start, find end
        queue = deque([(start, 0)])
        visited = {start}
        target_pos = end

    while queue:
        pos, dist = queue.popleft()

        if reverse:
            if ord(grid[pos]) == target_height:
                return dist
        else:
            if pos == target_pos:
                return dist

        for neighbor in grid.neighbors(pos):
            if neighbor not in visited:
                current_height = ord(grid[pos])
                neighbor_height = ord(grid[neighbor])

                if reverse:
                    # Can go down any amount, up at most 1
                    if current_height - neighbor_height <= 1:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
                else:
                    # Can go up at most 1, down any amount
                    if neighbor_height - current_height <= 1:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    return -1


answer(12.1, 425, lambda: find_shortest_path(in12))
answer(12.2, 418, lambda: find_shortest_path(in12, reverse=True))

# %% Day 13: Distress Signal
in13_raw = parse(13, str.strip)
in13 = []
current_pair = []
for line in in13_raw:
    if line:
        current_pair.append(line)
        if len(current_pair) == 2:
            in13.append(current_pair)
            current_pair = []


def compare_packets(left, right):
    """Compare two packets according to distress signal rules."""
    if isinstance(left, int) and isinstance(right, int):
        return (left > right) - (left < right)

    if isinstance(left, int):
        left = [left]
    if isinstance(right, int):
        right = [right]

    for i in range(min(len(left), len(right))):
        result = compare_packets(left[i], right[i])
        if result != 0:
            return result

    return (len(left) > len(right)) - (len(left) < len(right))


def sum_ordered_pairs(sections):
    """Sum indices of correctly ordered pairs."""
    total = 0
    for i, section in enumerate(sections):
        if len(section) == 2:
            left = eval(section[0])
            right = eval(section[1])
            if compare_packets(left, right) <= 0:
                total += i + 1
    return total


def find_decoder_key(sections):
    """Find decoder key using divider packets."""
    packets = []
    for section in sections:
        for line in section:
            if line.strip():
                packets.append(eval(line))

    # Add divider packets
    divider1 = [[2]]
    divider2 = [[6]]
    packets.extend([divider1, divider2])

    # Sort packets
    from functools import cmp_to_key

    packets.sort(key=cmp_to_key(compare_packets))

    # Find positions of divider packets
    pos1 = packets.index(divider1) + 1
    pos2 = packets.index(divider2) + 1

    return pos1 * pos2


answer(13.1, 5292, lambda: sum_ordered_pairs(in13))
answer(13.2, 23868, lambda: find_decoder_key(in13))

# %% Day 14: Regolith Reservoir
in14 = parse(14)


def simulate_sand(lines, floor=False):
    """Simulate falling sand."""
    # Parse rock formations
    rocks = set()
    max_y = 0

    for line in lines:
        points = []
        for coord in line.split(" -> "):
            x, y = map(int, coord.split(","))
            points.append((x, y))
            max_y = max(max_y, y)

        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            if x1 == x2:  # Vertical line
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    rocks.add((x1, y))
            else:  # Horizontal line
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    rocks.add((x, y1))

    if floor:
        floor_y = max_y + 2

    sand_count = 0
    sand_source = (500, 0)

    while True:
        # Drop a grain of sand
        sand_x, sand_y = sand_source

        if not floor and sand_y > max_y:
            break  # Sand falls into void

        if floor and (sand_source in rocks):
            break  # Source is blocked

        # Simulate falling
        while True:
            next_y = sand_y + 1

            if floor and next_y == floor_y:
                # Hit floor, sand settles
                rocks.add((sand_x, sand_y))
                sand_count += 1
                break

            if not floor and next_y > max_y:
                # Falls into void
                return sand_count

            # Try to fall straight down
            if (sand_x, next_y) not in rocks:
                sand_y = next_y
                continue

            # Try to fall diagonally left
            if (sand_x - 1, next_y) not in rocks:
                sand_x -= 1
                sand_y = next_y
                continue

            # Try to fall diagonally right
            if (sand_x + 1, next_y) not in rocks:
                sand_x += 1
                sand_y = next_y
                continue

            # Sand settles here
            rocks.add((sand_x, sand_y))
            sand_count += 1
            break

    return sand_count


answer(14.1, 1406, lambda: simulate_sand(in14))
answer(14.2, 20870, lambda: simulate_sand(in14, floor=True))

# %% Day 15: Beacon Exclusion Zone
in15 = parse(15)


def solve_beacons(lines, target_row=2000000, search_space=4000000):
    """Solve beacon exclusion problems."""
    sensors = []
    beacons = set()

    for line in lines:
        nums = ints(line)
        sx, sy, bx, by = nums[0], nums[1], nums[2], nums[3]
        sensors.append((sx, sy, taxi_distance((sx, sy), (bx, by))))
        beacons.add((bx, by))

    # Part 1: Count positions that cannot contain a beacon
    excluded = set()
    for sx, sy, radius in sensors:
        distance_to_row = abs(sy - target_row)
        if distance_to_row <= radius:
            width = radius - distance_to_row
            for x in range(sx - width, sx + width + 1):
                if (x, target_row) not in beacons:
                    excluded.add(x)

    part1 = len(excluded)

    # Part 2: Find the distress beacon
    for y in range(search_space + 1):
        ranges = []
        for sx, sy, radius in sensors:
            distance_to_row = abs(sy - y)
            if distance_to_row <= radius:
                width = radius - distance_to_row
                ranges.append((sx - width, sx + width))

        # Merge overlapping ranges
        ranges.sort()
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Check for gaps
        if len(merged) > 1:
            for i in range(len(merged) - 1):
                gap_start = merged[i][1] + 1
                gap_end = merged[i + 1][0] - 1
                if gap_start <= gap_end and 0 <= gap_start <= search_space:
                    return part1, gap_start * 4000000 + y

    return part1, 0


def beacon_exclusion_part1(lines):
    """Part 1 of beacon exclusion."""
    result, _ = solve_beacons(lines)
    return result


def beacon_exclusion_part2(lines):
    """Part 2 of beacon exclusion."""
    _, result = solve_beacons(lines)
    return result


answer(15.1, 5147333, lambda: beacon_exclusion_part1(in15))
answer(15.2, 13734006908372, lambda: beacon_exclusion_part2(in15))

# %% Day 16: Proboscidea Volcanium
in16 = parse(16)


def solve_valves(lines, time_limit=30, elephants=False):
    """Solve valve pressure optimization."""
    # Parse valve information
    valves = {}
    for line in lines:
        parts = line.split()
        valve = parts[1]
        flow_rate = int(parts[4].split("=")[1].rstrip(";"))
        tunnels = [t.rstrip(",") for t in parts[9:]]
        valves[valve] = {"flow": flow_rate, "tunnels": tunnels}

    # Find shortest paths between all pairs of valves
    from collections import defaultdict

    distances = defaultdict(lambda: defaultdict(lambda: float("inf")))

    for valve in valves:
        distances[valve][valve] = 0
        for tunnel in valves[valve]["tunnels"]:
            distances[valve][tunnel] = 1

    # Floyd-Warshall algorithm
    for k in valves:
        for i in valves:
            for j in valves:
                distances[i][j] = min(
                    distances[i][j], distances[i][k] + distances[k][j]
                )

    # Get valves with positive flow rates
    useful_valves = [v for v in valves if valves[v]["flow"] > 0]

    if not elephants:
        # Part 1: Just you working alone
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def max_pressure(current, opened, time_left):
            if time_left <= 0:
                return 0

            best = 0

            # Try opening current valve if it's useful and not opened
            if current in useful_valves:
                valve_bit = 1 << useful_valves.index(current)
                if not (opened & valve_bit) and valves[current]["flow"] > 0:
                    new_opened = opened | valve_bit
                    pressure = valves[current]["flow"] * (time_left - 1)
                    best = max(
                        best,
                        pressure + max_pressure(current, new_opened, time_left - 1),
                    )

            # Try moving to other useful valves
            for next_valve in useful_valves:
                if next_valve != current:
                    travel_time = distances[current][next_valve]
                    if travel_time < time_left:
                        best = max(
                            best,
                            max_pressure(next_valve, opened, time_left - travel_time),
                        )

            return best

        return max_pressure("AA", 0, time_limit)
    else:
        # Part 2: Generate all possible state outcomes
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dfs(current, opened, time_left):
            if time_left <= 0:
                return [(opened, 0)]

            results = []

            # Try opening current valve if it's useful and not opened
            if current in useful_valves:
                valve_bit = 1 << useful_valves.index(current)
                if not (opened & valve_bit) and valves[current]["flow"] > 0:
                    new_opened = opened | valve_bit
                    pressure = valves[current]["flow"] * (time_left - 1)
                    for future_opened, future_pressure in dfs(
                        current, new_opened, time_left - 1
                    ):
                        results.append((future_opened, future_pressure + pressure))

            # Try moving to other useful valves
            for next_valve in useful_valves:
                if next_valve != current:
                    travel_time = distances[current][next_valve]
                    if travel_time < time_left:
                        for future_opened, future_pressure in dfs(
                            next_valve, opened, time_left - travel_time
                        ):
                            results.append((future_opened, future_pressure))

            # If no actions possible, just wait
            if not results:
                results.append((opened, 0))

            return results

        # Get all possible end states
        all_states = {}
        for opened, pressure in dfs("AA", 0, 26):
            all_states[opened] = max(all_states.get(opened, 0), pressure)

        # Find best combination where human and elephant work on disjoint sets
        best_total = 0
        state_items = list(all_states.items())

        for i, (human_opened, human_pressure) in enumerate(state_items):
            for elephant_opened, elephant_pressure in state_items[i:]:
                if human_opened & elephant_opened == 0:  # Disjoint sets
                    total = human_pressure + elephant_pressure
                    best_total = max(best_total, total)

        return best_total


answer(16.1, 1767, lambda: solve_valves(in16))
answer(16.2, 2528, lambda: solve_valves(in16, time_limit=26, elephants=True))

# %% Day 17: Pyroclastic Flow
in17 = parse(17)[0]


def simulate_tetris(jet_pattern, blocks_to_drop=2022):
    """Simulate falling rocks in Tetris-like game."""
    # Rock shapes (bottom-left corner relative positions)
    rocks = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # Horizontal line
        [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],  # Plus
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # Reverse L
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # Vertical line
        [(0, 0), (1, 0), (0, 1), (1, 1)],  # Square
    ]

    chamber = set()
    jet_index = 0
    height = 0

    # For cycle detection (part 2)
    seen_states = {}

    for block_num in range(blocks_to_drop):
        # Choose rock shape
        rock = rocks[block_num % len(rocks)]

        # Start position (2 units from left, 3 units above highest rock)
        rock_x, rock_y = 2, height + 3

        while True:
            # Apply jet push
            jet = jet_pattern[jet_index % len(jet_pattern)]
            jet_index += 1

            # Try to move sideways
            if jet == "<":
                new_x = rock_x - 1
            else:  # jet == '>'
                new_x = rock_x + 1

            # Check if sideways movement is valid
            valid_move = True
            for dx, dy in rock:
                new_pos = (new_x + dx, rock_y + dy)
                if new_pos[0] < 0 or new_pos[0] >= 7 or new_pos in chamber:
                    valid_move = False
                    break

            if valid_move:
                rock_x = new_x

            # Try to move down
            new_y = rock_y - 1
            can_fall = True
            for dx, dy in rock:
                new_pos = (rock_x + dx, new_y + dy)
                if new_pos[1] < 0 or new_pos in chamber:
                    can_fall = False
                    break

            if can_fall:
                rock_y = new_y
            else:
                # Rock settles
                for dx, dy in rock:
                    chamber.add((rock_x + dx, rock_y + dy))
                    height = max(height, rock_y + dy + 1)
                break

        # Cycle detection for part 2
        if blocks_to_drop > 10000:  # Only for large numbers
            # Find surface profile - the topmost rock in each column
            surface = []
            for x in range(7):
                max_y = -1
                for y in range(height, -1, -1):
                    if (x, y) in chamber:
                        max_y = y
                        break
                surface.append(max_y)

            # Normalize surface relative to minimum height
            min_surface = min(surface)
            normalized_surface = tuple(y - min_surface for y in surface)

            state = (
                block_num % len(rocks),
                jet_index % len(jet_pattern),
                normalized_surface,
            )

            if state in seen_states:
                # Found cycle
                cycle_start, cycle_height_start = seen_states[state]
                cycle_length = block_num - cycle_start
                cycle_height = height - cycle_height_start

                if cycle_length > 0:  # Valid cycle
                    remaining_blocks = blocks_to_drop - block_num - 1
                    full_cycles = remaining_blocks // cycle_length
                    remainder = remaining_blocks % cycle_length

                    # Calculate final height
                    final_height = height + full_cycles * cycle_height

                    # Handle remainder by simulating just enough more blocks
                    if remainder > 0:
                        # Find height at cycle_start + remainder
                        target_block = cycle_start + remainder
                        for prev_state, (
                            prev_block,
                            prev_height,
                        ) in seen_states.items():
                            if prev_block == target_block:
                                final_height += prev_height - cycle_height_start
                                break

                    return final_height

            seen_states[state] = (block_num, height)

    return height


answer(17.1, 3161, lambda: simulate_tetris(in17))
answer(17.2, 1575931232076, lambda: simulate_tetris(in17, blocks_to_drop=1000000000000))

# %% Day 18: Boiling Boulders
in18 = parse(18)


def calculate_surface_area(lines, external_only=False):
    """Calculate surface area of lava droplets."""
    cubes = set()
    for line in lines:
        x, y, z = ints(line)
        cubes.add((x, y, z))

    # Six directions for adjacent cubes
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    if not external_only:
        # Part 1: Count all exposed faces
        surface_area = 0
        for cube in cubes:
            x, y, z = cube
            for dx, dy, dz in directions:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor not in cubes:
                    surface_area += 1
        return surface_area

    # Part 2: Only count external faces
    # Find bounding box
    min_x = min(x for x, y, z in cubes) - 1
    max_x = max(x for x, y, z in cubes) + 1
    min_y = min(y for x, y, z in cubes) - 1
    max_y = max(y for x, y, z in cubes) + 1
    min_z = min(z for x, y, z in cubes) - 1
    max_z = max(z for x, y, z in cubes) + 1

    # BFS to find all external air pockets
    external_air = set()
    queue = deque([(min_x, min_y, min_z)])
    external_air.add((min_x, min_y, min_z))

    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            nx, ny, nz = neighbor
            if (
                min_x <= nx <= max_x
                and min_y <= ny <= max_y
                and min_z <= nz <= max_z
                and neighbor not in cubes
                and neighbor not in external_air
            ):
                external_air.add(neighbor)
                queue.append(neighbor)

    # Count faces exposed to external air
    surface_area = 0
    for cube in cubes:
        x, y, z = cube
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in external_air:
                surface_area += 1

    return surface_area


answer(18.1, 4348, lambda: calculate_surface_area(in18))
answer(18.2, 2546, lambda: calculate_surface_area(in18, external_only=True))

# %% Day 19: Not Enough Minerals
in19 = parse(19)


def solve_blueprints(lines, time_limit=24, top_blueprints=None):
    """Solve blueprint optimization problem."""
    total_quality = 0

    for i, line in enumerate(lines):
        if top_blueprints and i >= top_blueprints:
            break

        nums = ints(line)
        blueprint_id = nums[0]
        ore_robot_cost = nums[1]
        clay_robot_cost = nums[2]
        obsidian_robot_ore_cost = nums[3]
        obsidian_robot_clay_cost = nums[4]
        geode_robot_ore_cost = nums[5]
        geode_robot_obsidian_cost = nums[6]

        # State: (time, ore, clay, obsidian, geodes, ore_robots, clay_robots, obsidian_robots, geode_robots)
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def max_geodes(
            time,
            ore,
            clay,
            obsidian,
            geodes,
            ore_robots,
            clay_robots,
            obsidian_robots,
            geode_robots,
        ):
            if time == 0:
                return geodes

            # Prune if we can't possibly beat current best
            max_possible = geodes + geode_robots * time + (time * (time - 1)) // 2
            if max_possible <= max_geodes.best:
                return 0

            best = 0

            # Try building different robots
            # Build geode robot
            if ore >= geode_robot_ore_cost and obsidian >= geode_robot_obsidian_cost:
                result = max_geodes(
                    time - 1,
                    ore + ore_robots - geode_robot_ore_cost,
                    clay + clay_robots,
                    obsidian + obsidian_robots - geode_robot_obsidian_cost,
                    geodes + geode_robots,
                    ore_robots,
                    clay_robots,
                    obsidian_robots,
                    geode_robots + 1,
                )
                best = max(best, result)

            # Build obsidian robot
            if (
                ore >= obsidian_robot_ore_cost
                and clay >= obsidian_robot_clay_cost
                and obsidian_robots < geode_robot_obsidian_cost
            ):
                result = max_geodes(
                    time - 1,
                    ore + ore_robots - obsidian_robot_ore_cost,
                    clay + clay_robots - obsidian_robot_clay_cost,
                    obsidian + obsidian_robots,
                    geodes + geode_robots,
                    ore_robots,
                    clay_robots,
                    obsidian_robots + 1,
                    geode_robots,
                )
                best = max(best, result)

            # Build clay robot
            if ore >= clay_robot_cost and clay_robots < obsidian_robot_clay_cost:
                result = max_geodes(
                    time - 1,
                    ore + ore_robots - clay_robot_cost,
                    clay + clay_robots,
                    obsidian + obsidian_robots,
                    geodes + geode_robots,
                    ore_robots,
                    clay_robots + 1,
                    obsidian_robots,
                    geode_robots,
                )
                best = max(best, result)

            # Build ore robot
            max_ore_needed = max(
                ore_robot_cost,
                clay_robot_cost,
                obsidian_robot_ore_cost,
                geode_robot_ore_cost,
            )
            if ore >= ore_robot_cost and ore_robots < max_ore_needed:
                result = max_geodes(
                    time - 1,
                    ore + ore_robots - ore_robot_cost,
                    clay + clay_robots,
                    obsidian + obsidian_robots,
                    geodes + geode_robots,
                    ore_robots + 1,
                    clay_robots,
                    obsidian_robots,
                    geode_robots,
                )
                best = max(best, result)

            # Don't build anything
            result = max_geodes(
                time - 1,
                ore + ore_robots,
                clay + clay_robots,
                obsidian + obsidian_robots,
                geodes + geode_robots,
                ore_robots,
                clay_robots,
                obsidian_robots,
                geode_robots,
            )
            best = max(best, result)

            max_geodes.best = max(max_geodes.best, best)
            return best

        max_geodes.best = 0
        geodes = max_geodes(time_limit, 0, 0, 0, 0, 1, 0, 0, 0)

        if top_blueprints:
            total_quality = geodes if total_quality == 0 else total_quality * geodes
        else:
            total_quality += blueprint_id * geodes

    return total_quality


answer(19.1, 1389, lambda: solve_blueprints(in19))
answer(19.2, 3003, lambda: solve_blueprints(in19, time_limit=32, top_blueprints=3))

# %% Day 20: Grove Positioning System
in20 = parse(20, int)


def mix_file(numbers, rounds=1, decryption_key=1):
    """Mix the encrypted file."""
    # Apply decryption key
    numbers = [n * decryption_key for n in numbers]

    # Create list of (original_index, value) pairs
    mixed = [(i, num) for i, num in enumerate(numbers)]

    for _ in range(rounds):
        for original_index in range(len(numbers)):
            # Find current position of this number
            current_pos = next(
                i for i, (orig_idx, _) in enumerate(mixed) if orig_idx == original_index
            )

            # Remove the number
            _, value = mixed.pop(current_pos)

            # Calculate new position
            new_pos = (current_pos + value) % len(mixed)

            # Insert at new position
            mixed.insert(new_pos, (original_index, value))

    # Find position of 0
    zero_pos = next(i for i, (_, value) in enumerate(mixed) if value == 0)

    # Get values at positions 1000, 2000, 3000 after 0
    coordinates = []
    for offset in [1000, 2000, 3000]:
        pos = (zero_pos + offset) % len(mixed)
        coordinates.append(mixed[pos][1])

    return sum(coordinates)


answer(20.1, 8372, lambda: mix_file(in20))
answer(20.2, 7865110481723, lambda: mix_file(in20, rounds=10, decryption_key=811589153))

# %% Day 21: Monkey Math
in21 = parse(21)


def solve_monkey_math(lines, part2=False):
    """Solve monkey math puzzle."""
    monkeys = {}

    for line in lines:
        name, operation = line.split(": ")
        if operation.isdigit() or (
            operation.startswith("-") and operation[1:].isdigit()
        ):
            monkeys[name] = int(operation)
        else:
            monkeys[name] = operation

    if not part2:
        # Part 1: Just evaluate
        def evaluate(name):
            if isinstance(monkeys[name], int):
                return monkeys[name]

            operation = monkeys[name]
            parts = operation.split()
            left, op, right = parts[0], parts[1], parts[2]

            left_val = evaluate(left)
            right_val = evaluate(right)

            if op == "+":
                return left_val + right_val
            elif op == "-":
                return left_val - right_val
            elif op == "*":
                return left_val * right_val
            elif op == "/":
                return left_val // right_val

        return evaluate("root")

    # Part 2: Find value for 'humn' that makes root's operands equal
    def evaluate_with_unknown(name):
        if name == "humn":
            return "x"

        if isinstance(monkeys[name], int):
            return monkeys[name]

        operation = monkeys[name]
        parts = operation.split()
        left, op, right = parts[0], parts[1], parts[2]

        left_val = evaluate_with_unknown(left)
        right_val = evaluate_with_unknown(right)

        if (
            left_val == "x"
            or right_val == "x"
            or (isinstance(left_val, tuple) and left_val[0] == "x")
            or (isinstance(right_val, tuple) and right_val[0] == "x")
        ):
            return ("x", left_val, op, right_val)

        if op == "+":
            return left_val + right_val
        elif op == "-":
            return left_val - right_val
        elif op == "*":
            return left_val * right_val
        elif op == "/":
            return left_val // right_val

    # Get root's operands
    root_op = monkeys["root"]
    parts = root_op.split()
    left_name, right_name = parts[0], parts[2]

    left_val = evaluate_with_unknown(left_name)
    right_val = evaluate_with_unknown(right_name)

    # One side should be numeric, the other should contain 'x'
    if isinstance(left_val, int):
        target = left_val
        equation = right_val
    else:
        target = right_val
        equation = left_val

    # Solve equation backwards
    def solve_equation(equation, target):
        if equation == "x":
            return target

        _, left, op, right = equation

        if isinstance(left, int):
            # left op right = target, solve for right
            if op == "+":
                return solve_equation(right, target - left)
            elif op == "-":
                return solve_equation(right, left - target)
            elif op == "*":
                return solve_equation(right, target // left)
            elif op == "/":
                return solve_equation(right, left // target)
        elif isinstance(right, int):
            # left op right = target, solve for left
            if op == "+":
                return solve_equation(left, target - right)
            elif op == "-":
                return solve_equation(left, target + right)
            elif op == "*":
                return solve_equation(left, target // right)
            elif op == "/":
                return solve_equation(left, target * right)
        else:
            # This shouldn't happen if our logic is correct
            raise ValueError(f"Cannot solve equation: {equation} = {target}")

    return solve_equation(equation, target)


answer(21.1, 21208142603224, lambda: solve_monkey_math(in21))
answer(21.2, 3882224466191, lambda: solve_monkey_math(in21, part2=True))

# %% Day 22: Monkey Map
in22 = parse(22, sections=lambda text: text.split("\n\n"))


def solve_monkey_map(sections, cube=False):
    """Solve monkey map navigation."""
    grid_lines = sections[0].split("\n")
    instructions = sections[1].strip()

    # Parse grid
    grid = {}
    for y, line in enumerate(grid_lines):
        for x, char in enumerate(line):
            if char in ".#":
                grid[(x, y)] = char

    # Parse instructions
    moves = []
    i = 0
    while i < len(instructions):
        if instructions[i].isdigit():
            num = ""
            while i < len(instructions) and instructions[i].isdigit():
                num += instructions[i]
                i += 1
            moves.append(int(num))
        else:
            moves.append(instructions[i])
            i += 1

    # Find starting position
    start_x = min(x for x, y in grid.keys() if y == 0)
    pos = (start_x, 0)
    facing = 0  # 0=right, 1=down, 2=left, 3=up

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_cube_face(x, y):
        """Get cube face number based on position."""
        # Assuming 50x50 faces in standard AoC layout:
        #   12
        #   3
        #  45
        #  6
        if 50 <= x < 100 and 0 <= y < 50:
            return 1
        elif 100 <= x < 150 and 0 <= y < 50:
            return 2
        elif 50 <= x < 100 and 50 <= y < 100:
            return 3
        elif 0 <= x < 50 and 100 <= y < 150:
            return 4
        elif 50 <= x < 100 and 100 <= y < 150:
            return 5
        elif 0 <= x < 50 and 150 <= y < 200:
            return 6
        return 0

    def cube_wrap(x, y, facing):
        """Handle cube wrapping for part 2."""
        face = get_cube_face(x, y)
        local_x = x % 50
        local_y = y % 50

        # Define wrapping rules for each face and direction
        if face == 1:  # Top face
            if facing == 3:  # up -> face 6 left
                return (0, 150 + local_x, 0)
            elif facing == 2:  # left -> face 4 left (flipped)
                return (0, 149 - local_y, 0)
        elif face == 2:  # Right face
            if facing == 0:  # right -> face 5 right (flipped)
                return (99, 149 - local_y, 2)
            elif facing == 1:  # down -> face 3 right
                return (99, 50 + local_x, 2)
            elif facing == 3:  # up -> face 6 up
                return (local_x, 199, 3)
        elif face == 3:  # Middle face
            if facing == 0:  # right -> face 2 up
                return (100 + local_y, 49, 3)
            elif facing == 2:  # left -> face 4 down
                return (local_y, 100, 1)
        elif face == 4:  # Bottom left face
            if facing == 2:  # left -> face 1 left (flipped)
                return (50, 49 - local_y, 0)
            elif facing == 3:  # up -> face 3 left
                return (50, 50 + local_x, 0)
        elif face == 5:  # Bottom right face
            if facing == 0:  # right -> face 2 right (flipped)
                return (149, 49 - local_y, 2)
            elif facing == 1:  # down -> face 6 right
                return (49, 150 + local_x, 2)
        elif face == 6:  # Bottom face
            if facing == 0:  # right -> face 5 up
                return (50 + local_y, 149, 3)
            elif facing == 1:  # down -> face 2 down
                return (100 + local_x, 0, 1)
            elif facing == 2:  # left -> face 1 down
                return (50 + local_y, 0, 1)

        # Default: no wrapping
        return (x, y, facing)

    for move in moves:
        if isinstance(move, int):
            # Move forward
            for _ in range(move):
                dx, dy = directions[facing]
                new_pos = (pos[0] + dx, pos[1] + dy)
                new_facing = facing

                if new_pos not in grid:
                    if not cube:
                        # Wrap around (part 1)
                        if facing == 0:  # right
                            new_pos = (
                                min(x for x, y in grid.keys() if y == pos[1]),
                                pos[1],
                            )
                        elif facing == 1:  # down
                            new_pos = (
                                pos[0],
                                min(y for x, y in grid.keys() if x == pos[0]),
                            )
                        elif facing == 2:  # left
                            new_pos = (
                                max(x for x, y in grid.keys() if y == pos[1]),
                                pos[1],
                            )
                        elif facing == 3:  # up
                            new_pos = (
                                pos[0],
                                max(y for x, y in grid.keys() if x == pos[0]),
                            )
                    else:
                        # Cube wrapping (part 2)
                        new_x, new_y, new_facing = cube_wrap(pos[0], pos[1], facing)
                        new_pos = (new_x, new_y)

                if new_pos in grid and grid[new_pos] == ".":
                    pos = new_pos
                    facing = new_facing
                elif new_pos in grid and grid[new_pos] == "#":
                    break
        else:
            # Turn
            if move == "R":
                facing = (facing + 1) % 4
            elif move == "L":
                facing = (facing - 1) % 4

    return 1000 * (pos[1] + 1) + 4 * (pos[0] + 1) + facing


answer(22.1, 66292, lambda: solve_monkey_map(in22))
answer(22.2, 127012, lambda: solve_monkey_map(in22, cube=True))

# %% Day 23: Unstable Diffusion
in23 = parse(23)


def simulate_elves(lines, rounds=10, part2=False):
    """Simulate elf movement."""
    elves = set()
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == "#":
                elves.add((x, y))

    # Directions: N, S, W, E
    directions = [
        [(-1, -1), (0, -1), (1, -1)],  # North
        [(-1, 1), (0, 1), (1, 1)],  # South
        [(-1, -1), (-1, 0), (-1, 1)],  # West
        [(1, -1), (1, 0), (1, 1)],  # East
    ]

    direction_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    round_num = 0
    while True:
        if not part2 and round_num >= rounds:
            break

        # First half: propose moves
        proposals = {}
        proposal_counts = defaultdict(int)

        for elf in elves:
            x, y = elf

            # Check if elf has neighbors
            neighbors = [
                (x + dx, y + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if (dx, dy) != (0, 0)
            ]
            if not any(neighbor in elves for neighbor in neighbors):
                continue  # No neighbors, don't move

            # Check each direction
            for i in range(4):
                dir_idx = (i + round_num) % 4
                direction = directions[dir_idx]

                # Check if all positions in this direction are free
                if all((x + dx, y + dy) not in elves for dx, dy in direction):
                    # Propose move
                    move_dx, move_dy = direction_moves[dir_idx]
                    new_pos = (x + move_dx, y + move_dy)
                    proposals[elf] = new_pos
                    proposal_counts[new_pos] += 1
                    break

        # Second half: move if no conflicts
        moved = False
        new_elves = set()

        for elf in elves:
            if elf in proposals and proposal_counts[proposals[elf]] == 1:
                new_elves.add(proposals[elf])
                moved = True
            else:
                new_elves.add(elf)

        elves = new_elves
        round_num += 1

        if part2 and not moved:
            return round_num

    # Calculate bounding box
    min_x = min(x for x, y in elves)
    max_x = max(x for x, y in elves)
    min_y = min(y for x, y in elves)
    max_y = max(y for x, y in elves)

    return (max_x - min_x + 1) * (max_y - min_y + 1) - len(elves)


answer(23.1, 3689, lambda: simulate_elves(in23))
answer(23.2, 965, lambda: simulate_elves(in23, part2=True))

# %% Day 24: Blizzard Basin
in24 = parse(24)


def solve_blizzard_basin(lines, part2=False):
    """Solve blizzard navigation."""
    grid = [list(line) for line in lines]
    height, width = len(grid), len(grid[0])

    # Find start and end positions
    start = (1, 0)
    end = (width - 2, height - 1)

    # Parse blizzards
    blizzards = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] in "<>^v":
                blizzards.append((x, y, grid[y][x]))

    def get_blizzard_positions(time):
        """Get blizzard positions at given time."""
        positions = set()
        for x, y, direction in blizzards:
            if direction == "<":
                new_x = ((x - 1 - time) % (width - 2)) + 1
                new_y = y
            elif direction == ">":
                new_x = ((x - 1 + time) % (width - 2)) + 1
                new_y = y
            elif direction == "^":
                new_x = x
                new_y = ((y - 1 - time) % (height - 2)) + 1
            elif direction == "v":
                new_x = x
                new_y = ((y - 1 + time) % (height - 2)) + 1

            positions.add((new_x, new_y))
        return positions

    def find_path(start_pos, end_pos, start_time):
        """Find shortest path from start to end."""
        queue = deque([(start_pos[0], start_pos[1], start_time)])
        visited = set()

        while queue:
            x, y, time = queue.popleft()

            if (x, y) == end_pos:
                return time

            if (x, y, time % lcm(width - 2, height - 2)) in visited:
                continue
            visited.add((x, y, time % lcm(width - 2, height - 2)))

            # Get blizzard positions at next time
            next_blizzards = get_blizzard_positions(time + 1)

            # Try all moves (including waiting)
            for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy

                # Check bounds
                if (
                    (new_x, new_y) == start_pos
                    or (new_x, new_y) == end_pos
                    or (1 <= new_x < width - 1 and 1 <= new_y < height - 1)
                ):
                    # Check if position is clear
                    if (new_x, new_y) not in next_blizzards:
                        queue.append((new_x, new_y, time + 1))

        return -1

    # Part 1: Just go to the end
    time1 = find_path(start, end, 0)

    if not part2:
        return time1

    # Part 2: Go to end, back to start, then to end again
    time2 = find_path(end, start, time1)
    time3 = find_path(start, end, time2)

    return time3


answer(24.1, 274, lambda: solve_blizzard_basin(in24))
answer(24.2, 839, lambda: solve_blizzard_basin(in24, part2=True))

# %% Day 25: Full of Hot Air
in25 = parse(25)


def solve_snafu(lines):
    """Convert SNAFU numbers and sum them."""

    def snafu_to_decimal(snafu):
        result = 0
        power = 1
        for char in reversed(snafu):
            if char == "=":
                digit = -2
            elif char == "-":
                digit = -1
            else:
                digit = int(char)
            result += digit * power
            power *= 5
        return result

    def decimal_to_snafu(decimal):
        if decimal == 0:
            return "0"

        result = []
        while decimal > 0:
            remainder = decimal % 5
            decimal //= 5

            if remainder <= 2:
                result.append(str(remainder))
            elif remainder == 3:
                result.append("=")
                decimal += 1
            elif remainder == 4:
                result.append("-")
                decimal += 1

        return "".join(reversed(result))

    # Sum all SNAFU numbers
    total = sum(snafu_to_decimal(line) for line in lines)

    # Convert back to SNAFU
    return decimal_to_snafu(total)


answer(25.1, "2-212-2---=00-1--102", lambda: solve_snafu(in25))

# %% Summary
summary()
