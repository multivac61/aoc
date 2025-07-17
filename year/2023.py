#!/usr/bin/env python3

from collections import Counter, defaultdict, deque
import heapq

from aoc import (
    Grid,
    add2,
    answer,
    cache,
    directions4,
    first,
    ints,
    lcm,
    minmax,
    parse_year,
    prod,
    summary,
    taxi_distance,
)

parse = parse_year(2023)

# %% Day 1: Trebuchet?!
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

# # %% Day 2: Cube Conundrum
# in2 = parse(2)


def parse_game(line):
    """Parse a game line into game_id and list of draws."""
    parts = line.split(": ")
    game_id = int(parts[0].split()[1])
    draws = []
    for draw in parts[1].split("; "):
        draw_dict = {}
        for cube in draw.split(", "):
            count, color = cube.split()
            draw_dict[color] = int(count)
        draws.append(draw_dict)
    return game_id, draws


def is_game_possible(draws, max_cubes):
    """Check if a game is possible given max cubes."""
    for draw in draws:
        for color, count in draw.items():
            if count > max_cubes.get(color, 0):
                return False
    return True


def sum_possible_games(lines):
    """Sum IDs of games possible with 12 red, 13 green, 14 blue cubes."""
    max_cubes = {"red": 12, "green": 13, "blue": 14}
    total = 0

    for line in lines:
        game_id, draws = parse_game(line)
        if is_game_possible(draws, max_cubes):
            total += game_id

    return total


def sum_power_of_sets(lines):
    """Sum power of minimum sets for each game."""
    total = 0

    for line in lines:
        _, draws = parse_game(line)
        min_cubes = defaultdict(int)

        for draw in draws:
            for color, count in draw.items():
                min_cubes[color] = max(min_cubes[color], count)

        power = prod(min_cubes.values())
        total += power

    return total


in2 = parse(2)
answer(2.1, 2541, lambda: sum_possible_games(in2))
answer(2.2, 66016, lambda: sum_power_of_sets(in2))

# %% Day 3: Gear Ratios
in3 = parse(3)


def find_part_numbers(lines):
    """Find all part numbers (numbers adjacent to symbols)."""
    grid = Grid(lines)
    part_numbers = []

    # Find all numbers and their positions
    for y in range(grid.size[1]):
        x = 0
        while x < grid.size[0]:
            if grid.get((x, y), ".").isdigit():
                # Found start of a number
                number_str = ""
                positions = []
                while x < grid.size[0] and grid.get((x, y), ".").isdigit():
                    number_str += grid[(x, y)]
                    positions.append((x, y))
                    x += 1

                # Check if adjacent to symbol
                is_part = False
                for pos in positions:
                    px, py = pos
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            neighbor = (px + dx, py + dy)
                            if grid.in_range(neighbor):
                                char = grid[neighbor]
                                if char != "." and not char.isdigit():
                                    is_part = True
                                    break
                        if is_part:
                            break
                    if is_part:
                        break

                if is_part:
                    part_numbers.append(int(number_str))
            else:
                x += 1

    return sum(part_numbers)


def find_gear_ratios(lines):
    """Find gear ratios (products of exactly two numbers adjacent to *)."""
    grid = Grid(lines)

    # Find all numbers and their positions
    numbers = []
    for y in range(grid.size[1]):
        x = 0
        while x < grid.size[0]:
            if grid.get((x, y), ".").isdigit():
                # Found start of a number
                number_str = ""
                positions = []
                while x < grid.size[0] and grid.get((x, y), ".").isdigit():
                    number_str += grid[(x, y)]
                    positions.append((x, y))
                    x += 1

                numbers.append((int(number_str), positions))
            else:
                x += 1

    # Find gears (*)
    total = 0
    for pos in grid:
        if grid[pos] == "*":
            # Find adjacent numbers
            adjacent_numbers = []
            for number, positions in numbers:
                if any(
                    taxi_distance(pos, num_pos) == 1
                    or max(abs(pos[0] - num_pos[0]), abs(pos[1] - num_pos[1])) == 1
                    for num_pos in positions
                ):
                    adjacent_numbers.append(number)

            # Check if exactly 2 adjacent numbers
            if len(adjacent_numbers) == 2:
                total += prod(adjacent_numbers)

    return total


# Using taxi_distance from aoc module

answer(3.1, 527369, lambda: find_part_numbers(in3))
answer(3.2, 73074886, lambda: find_gear_ratios(in3))

# %% Day 4: Scratchcards
in4 = parse(4)


def count_winning_numbers(line):
    """Count winning numbers on a scratchcard."""
    parts = line.split(": ")[1].split(" | ")
    winning = set(ints(parts[0]))
    have = set(ints(parts[1]))
    return len(winning & have)


def score_scratchcards(lines):
    """Score scratchcards: 1 point for first match, double for each additional."""
    total = 0
    for line in lines:
        matches = count_winning_numbers(line)
        if matches > 0:
            total += 2 ** (matches - 1)
    return total


def count_total_scratchcards(lines):
    """Count total scratchcards after processing wins."""
    card_counts = [1] * len(lines)

    for i, line in enumerate(lines):
        matches = count_winning_numbers(line)
        for j in range(i + 1, min(i + 1 + matches, len(lines))):
            card_counts[j] += card_counts[i]

    return sum(card_counts)


answer(4.1, 20829, lambda: score_scratchcards(in4))
answer(4.2, 12648035, lambda: count_total_scratchcards(in4))

# %% Day 5: If You Give A Seed A Fertilizer
in5 = parse(5, sections=lambda text: text.split("\n\n"))


def parse_almanac(sections):
    """Parse almanac into seeds and mapping functions."""
    seeds = ints(sections[0])

    mappings = []
    for section in sections[1:]:
        lines = section.strip().split("\n")[1:]  # Skip header
        ranges = []
        for line in lines:
            dest, src, length = ints(line)
            ranges.append((src, src + length - 1, dest - src))
        mappings.append(ranges)

    return seeds, mappings


def apply_mapping(value, ranges):
    """Apply a mapping to a value."""
    for src_start, src_end, offset in ranges:
        if src_start <= value <= src_end:
            return value + offset
    return value


def find_lowest_location(sections):
    """Find lowest location for any seed."""
    seeds, mappings = parse_almanac(sections)

    locations = []
    for seed in seeds:
        value = seed
        for mapping in mappings:
            value = apply_mapping(value, mapping)
        locations.append(value)

    return min(locations)


def find_lowest_location_ranges(sections):
    """Find lowest location for seed ranges."""
    seeds, mappings = parse_almanac(sections)

    # Convert seeds to ranges
    ranges = []
    for i in range(0, len(seeds), 2):
        start = seeds[i]
        length = seeds[i + 1]
        ranges.append((start, start + length - 1))

    # Apply mappings to ranges
    for mapping in mappings:
        new_ranges = []
        for range_start, range_end in ranges:
            # Split range based on mapping
            current_ranges = [(range_start, range_end)]

            for src_start, src_end, offset in mapping:
                next_ranges = []
                for r_start, r_end in current_ranges:
                    if r_end < src_start or r_start > src_end:
                        # No overlap
                        next_ranges.append((r_start, r_end))
                    else:
                        # Overlap - split range
                        if r_start < src_start:
                            next_ranges.append((r_start, src_start - 1))

                        overlap_start = max(r_start, src_start)
                        overlap_end = min(r_end, src_end)
                        new_ranges.append(
                            (overlap_start + offset, overlap_end + offset)
                        )

                        if r_end > src_end:
                            next_ranges.append((src_end + 1, r_end))

                current_ranges = next_ranges

            new_ranges.extend(current_ranges)

        ranges = new_ranges

    return min(start for start, _ in ranges)


answer(5.1, 177942185, lambda: find_lowest_location(in5))
answer(5.2, 69841803, lambda: find_lowest_location_ranges(in5))

# %% Day 6: Wait For It
in6 = parse(6)


def count_ways_to_win(time, distance):
    """Count ways to win a race."""
    ways = 0
    for hold_time in range(1, time):
        speed = hold_time
        travel_time = time - hold_time
        traveled = speed * travel_time
        if traveled > distance:
            ways += 1
    return ways


def solve_races(lines):
    """Solve multiple races."""
    times = ints(lines[0])
    distances = ints(lines[1])

    result = 1
    for time, distance in zip(times, distances):
        result *= count_ways_to_win(time, distance)

    return result


def solve_single_race(lines):
    """Solve as a single race (ignore spaces)."""
    time = int(lines[0].split(":")[1].replace(" ", ""))
    distance = int(lines[1].split(":")[1].replace(" ", ""))

    return count_ways_to_win(time, distance)


answer(6.1, 449820, lambda: solve_races(in6))
answer(6.2, 42250895, lambda: solve_single_race(in6))

# %% Day 7: Camel Cards
in7 = parse(7)


def hand_type(hand):
    """Get hand type (7=five of a kind, 1=high card)."""
    counts = sorted(Counter(hand).values(), reverse=True)

    if counts == [5]:
        return 7  # Five of a kind
    elif counts == [4, 1]:
        return 6  # Four of a kind
    elif counts == [3, 2]:
        return 5  # Full house
    elif counts == [3, 1, 1]:
        return 4  # Three of a kind
    elif counts == [2, 2, 1]:
        return 3  # Two pair
    elif counts == [2, 1, 1, 1]:
        return 2  # One pair
    else:
        return 1  # High card


def hand_type_with_jokers(hand):
    """Get hand type with jokers wild."""
    if "J" not in hand:
        return hand_type(hand)

    # Try replacing J with each possible card
    best_type = 0
    for replacement in "23456789TQKA":
        new_hand = hand.replace("J", replacement)
        best_type = max(best_type, hand_type(new_hand))

    return best_type


def card_value(card, jokers=False):
    """Get numeric value of a card."""
    if jokers and card == "J":
        return 1

    values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    return values[card]


def solve_camel_cards(lines, jokers=False):
    """Solve camel cards game."""
    hands = []
    for line in lines:
        hand, bid = line.split()
        type_func = hand_type_with_jokers if jokers else hand_type
        hand_type_val = type_func(hand)
        hand_values = [card_value(card, jokers) for card in hand]
        hands.append((hand_type_val, hand_values, int(bid)))

    # Sort by hand type, then by card values
    hands.sort()

    total = 0
    for rank, (_, _, bid) in enumerate(hands, 1):
        total += rank * bid

    return total


answer(7.1, 248105065, lambda: solve_camel_cards(in7))
answer(7.2, 249515436, lambda: solve_camel_cards(in7, jokers=True))

# %% Day 8: Haunted Wasteland
in8 = parse(8, sections=lambda text: text.split("\n\n"))


def parse_desert_map(sections):
    """Parse desert map into instructions and network."""
    instructions = sections[0].strip()

    network = {}
    for line in sections[1].strip().split("\n"):
        node, connections = line.split(" = ")
        left, right = connections.strip("()").split(", ")
        network[node] = (left, right)

    return instructions, network


def count_steps(sections):
    """Count steps from AAA to ZZZ."""
    instructions, network = parse_desert_map(sections)

    current = "AAA"
    steps = 0

    while current != "ZZZ":
        direction = instructions[steps % len(instructions)]
        if direction == "L":
            current = network[current][0]
        else:
            current = network[current][1]
        steps += 1

    return steps


def count_ghost_steps(sections):
    """Count steps for ghost navigation (all nodes ending in A to Z)."""
    instructions, network = parse_desert_map(sections)

    # Find all starting nodes (ending in A)
    starts = [node for node in network if node.endswith("A")]

    # Find cycle length for each starting node
    cycle_lengths = []
    for start in starts:
        current = start
        steps = 0

        while not current.endswith("Z"):
            direction = instructions[steps % len(instructions)]
            if direction == "L":
                current = network[current][0]
            else:
                current = network[current][1]
            steps += 1

        cycle_lengths.append(steps)

    # Find LCM of all cycle lengths
    result = cycle_lengths[0]
    for length in cycle_lengths[1:]:
        result = lcm(result, length)

    return result


answer(8.1, 23147, lambda: count_steps(in8))
answer(8.2, 22289513667691, lambda: count_ghost_steps(in8))

# %% Day 9: Mirage Maintenance
in9 = parse(9, ints)


def extrapolate_next(sequence):
    """Extrapolate the next value in a sequence."""
    if all(x == 0 for x in sequence):
        return 0

    differences = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
    return sequence[-1] + extrapolate_next(differences)


def extrapolate_previous(sequence):
    """Extrapolate the previous value in a sequence."""
    if all(x == 0 for x in sequence):
        return 0

    differences = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
    return sequence[0] - extrapolate_previous(differences)


def sum_extrapolated_values(sequences, backwards=False):
    """Sum extrapolated values for all sequences."""
    func = extrapolate_previous if backwards else extrapolate_next
    return sum(func(seq) for seq in sequences)


answer(9.1, 1834108701, lambda: sum_extrapolated_values(in9))
answer(9.2, 993, lambda: sum_extrapolated_values(in9, backwards=True))

# %% Day 10: Pipe Maze
in10 = parse(10)


def find_loop_length(lines):
    """Find length of the main loop."""
    grid = Grid(lines)

    # Find starting position
    start = first(pos for pos in grid if grid[pos] == "S")

    # Define pipe connections
    connections = {
        "|": [(0, -1), (0, 1)],  # North-South
        "-": [(-1, 0), (1, 0)],  # East-West
        "L": [(0, -1), (1, 0)],  # North-East
        "J": [(0, -1), (-1, 0)],  # North-West
        "7": [(-1, 0), (0, 1)],  # South-West
        "F": [(1, 0), (0, 1)],  # South-East
    }

    # Find which pipes connect to start
    start_connections = []
    for dx, dy in directions4:
        neighbor = add2(start, (dx, dy))
        if neighbor in grid:
            pipe = grid[neighbor]
            if pipe in connections:
                for cdx, cdy in connections[pipe]:
                    if add2(neighbor, (cdx, cdy)) == start:
                        start_connections.append((dx, dy))

    # Follow the loop
    current = start
    previous = None
    steps = 0

    # Pick first direction from start
    direction = start_connections[0]
    current = add2(start, direction)
    previous = start
    steps = 1

    while current != start:
        pipe = grid[current]
        # Find next direction (not back to previous)
        for dx, dy in connections[pipe]:
            next_pos = add2(current, (dx, dy))
            if next_pos != previous:
                previous = current
                current = next_pos
                break
        steps += 1

    return steps // 2


def count_enclosed_tiles(lines):
    """Count tiles enclosed by the loop."""
    grid = Grid(lines)

    # Find starting position
    start = first(pos for pos in grid if grid[pos] == "S")

    # Define pipe connections
    connections = {
        "|": [(0, -1), (0, 1)],  # North-South
        "-": [(-1, 0), (1, 0)],  # East-West
        "L": [(0, -1), (1, 0)],  # North-East
        "J": [(0, -1), (-1, 0)],  # North-West
        "7": [(-1, 0), (0, 1)],  # South-West
        "F": [(1, 0), (0, 1)],  # South-East
    }

    # Find which pipes connect to start
    start_connections = []
    for dx, dy in directions4:
        neighbor = add2(start, (dx, dy))
        if neighbor in grid:
            pipe = grid[neighbor]
            if pipe in connections:
                for cdx, cdy in connections[pipe]:
                    if add2(neighbor, (cdx, cdy)) == start:
                        start_connections.append((dx, dy))

    # Determine what S should be
    s_pipe = None
    for pipe, dirs in connections.items():
        if set(dirs) == set(start_connections):
            s_pipe = pipe
            break

    # Find all positions in the loop
    loop_positions = {start}
    current = start
    previous = None

    # Pick first direction from start
    direction = start_connections[0]
    current = add2(start, direction)
    previous = start
    loop_positions.add(current)

    while current != start:
        pipe = grid[current]
        # Find next direction (not back to previous)
        for dx, dy in connections[pipe]:
            next_pos = add2(current, (dx, dy))
            if next_pos != previous:
                previous = current
                current = next_pos
                loop_positions.add(current)
                break

    # Count enclosed tiles using ray casting
    enclosed = 0
    for y in range(grid.size[1]):
        for x in range(grid.size[0]):
            if (x, y) not in loop_positions:
                # Cast ray to the right
                crosses = 0
                for rx in range(x + 1, grid.size[0]):
                    if (rx, y) in loop_positions:
                        pipe = grid[(rx, y)]
                        if pipe == "S":
                            pipe = s_pipe
                        # Count crossings (vertical pipes and certain corners)
                        if pipe in "|LJ":
                            crosses += 1

                if crosses % 2 == 1:
                    enclosed += 1

    return enclosed


answer(10.1, 6886, lambda: find_loop_length(in10))
answer(10.2, 371, lambda: count_enclosed_tiles(in10))

# %% Day 11: Cosmic Expansion
in11 = parse(11)


def calculate_galaxy_distances(lines, expansion_factor=2):
    """Calculate sum of distances between all galaxy pairs."""
    grid = Grid(lines)
    galaxies = grid.findall("#")

    # Find empty rows and columns
    empty_rows = set(range(grid.size[1])) - set(pos[1] for pos in galaxies)
    empty_cols = set(range(grid.size[0])) - set(pos[0] for pos in galaxies)

    # Calculate distances between all pairs
    total_distance = 0
    for i in range(len(galaxies)):
        for j in range(i + 1, len(galaxies)):
            g1, g2 = galaxies[i], galaxies[j]

            # Base Manhattan distance
            distance = taxi_distance(g1, g2)

            # Add expansion for empty rows/cols crossed
            min_y, max_y = minmax([g1[1], g2[1]])
            min_x, max_x = minmax([g1[0], g2[0]])

            empty_rows_crossed = len([r for r in empty_rows if min_y < r < max_y])
            empty_cols_crossed = len([c for c in empty_cols if min_x < c < max_x])

            distance += empty_rows_crossed * (expansion_factor - 1)
            distance += empty_cols_crossed * (expansion_factor - 1)

            total_distance += distance

    return total_distance


answer(11.1, 9681886, lambda: calculate_galaxy_distances(in11))
answer(
    11.2,
    791134099634,
    lambda: calculate_galaxy_distances(in11, expansion_factor=1000000),
)

# %% Day 12: Hot Springs
in12 = parse(12)


@cache
def count_arrangements(pattern, groups):
    """Count possible arrangements of springs."""
    if not groups:
        return 0 if "#" in pattern else 1

    if not pattern:
        return 1 if not groups else 0

    if len(pattern) < sum(groups) + len(groups) - 1:
        return 0

    if pattern[0] == ".":
        return count_arrangements(pattern[1:], groups)

    if pattern[0] == "#":
        group = groups[0]
        if len(pattern) < group or "." in pattern[:group]:
            return 0
        if len(pattern) == group:
            return 1 if len(groups) == 1 else 0
        if pattern[group] == "#":
            return 0
        return count_arrangements(pattern[group + 1 :], groups[1:])

    # pattern[0] == '?'
    return count_arrangements("#" + pattern[1:], groups) + count_arrangements(
        "." + pattern[1:], groups
    )


def solve_springs(lines, unfold=False):
    """Solve spring arrangements."""
    total = 0
    for line in lines:
        pattern, groups_str = line.split()
        groups = tuple(ints(groups_str))

        if unfold:
            pattern = "?".join([pattern] * 5)
            groups = groups * 5

        total += count_arrangements(pattern, groups)

    return total


answer(12.1, 7633, lambda: solve_springs(in12))
answer(12.2, 23903579139437, lambda: solve_springs(in12, unfold=True))

# %% Day 13: Point of Incidence
in13 = parse(13, sections=lambda text: text.split("\n\n"))


def find_reflection(pattern, smudges=0):
    """Find reflection line in pattern."""
    lines = pattern.strip().split("\n")
    rows = len(lines)
    cols = len(lines[0])

    # Check horizontal reflections
    for r in range(rows - 1):
        differences = 0
        for i in range(min(r + 1, rows - r - 1)):
            for c in range(cols):
                if lines[r - i][c] != lines[r + 1 + i][c]:
                    differences += 1
        if differences == smudges:
            return 100 * (r + 1)

    # Check vertical reflections
    for c in range(cols - 1):
        differences = 0
        for i in range(min(c + 1, cols - c - 1)):
            for r in range(rows):
                if lines[r][c - i] != lines[r][c + 1 + i]:
                    differences += 1
        if differences == smudges:
            return c + 1

    return 0


def solve_mirrors(sections, smudges=0):
    """Solve mirror patterns."""
    return sum(find_reflection(section, smudges) for section in sections)


answer(13.1, 34202, lambda: solve_mirrors(in13))
answer(13.2, 34230, lambda: solve_mirrors(in13, smudges=1))

# %% Day 14: Parabolic Reflector Dish
in14 = parse(14)


def tilt_platform(grid, direction):
    """Tilt the platform in a given direction."""
    rows, cols = len(grid), len(grid[0])
    new_grid = [row[:] for row in grid]

    if direction == "north":
        for col in range(cols):
            # Find all segments separated by # rocks
            segments = []
            start = 0
            for row in range(rows):
                if new_grid[row][col] == "#":
                    if start < row:
                        segments.append((start, row))
                    start = row + 1
            if start < rows:
                segments.append((start, rows))

            # Process each segment
            for start, end in segments:
                # Count rocks in this segment
                rocks = 0
                for row in range(start, end):
                    if new_grid[row][col] == "O":
                        rocks += 1
                        new_grid[row][col] = "."

                # Place rocks at the top of the segment
                for i in range(rocks):
                    new_grid[start + i][col] = "O"

    elif direction == "west":
        for row in range(rows):
            # Find all segments separated by # rocks
            segments = []
            start = 0
            for col in range(cols):
                if new_grid[row][col] == "#":
                    if start < col:
                        segments.append((start, col))
                    start = col + 1
            if start < cols:
                segments.append((start, cols))

            # Process each segment
            for start, end in segments:
                # Count rocks in this segment
                rocks = 0
                for col in range(start, end):
                    if new_grid[row][col] == "O":
                        rocks += 1
                        new_grid[row][col] = "."

                # Place rocks at the left of the segment
                for i in range(rocks):
                    new_grid[row][start + i] = "O"

    elif direction == "south":
        for col in range(cols):
            # Find all segments separated by # rocks
            segments = []
            start = 0
            for row in range(rows):
                if new_grid[row][col] == "#":
                    if start < row:
                        segments.append((start, row))
                    start = row + 1
            if start < rows:
                segments.append((start, rows))

            # Process each segment
            for start, end in segments:
                # Count rocks in this segment
                rocks = 0
                for row in range(start, end):
                    if new_grid[row][col] == "O":
                        rocks += 1
                        new_grid[row][col] = "."

                # Place rocks at the bottom of the segment
                for i in range(rocks):
                    new_grid[end - 1 - i][col] = "O"

    elif direction == "east":
        for row in range(rows):
            # Find all segments separated by # rocks
            segments = []
            start = 0
            for col in range(cols):
                if new_grid[row][col] == "#":
                    if start < col:
                        segments.append((start, col))
                    start = col + 1
            if start < cols:
                segments.append((start, cols))

            # Process each segment
            for start, end in segments:
                # Count rocks in this segment
                rocks = 0
                for col in range(start, end):
                    if new_grid[row][col] == "O":
                        rocks += 1
                        new_grid[row][col] = "."

                # Place rocks at the right of the segment
                for i in range(rocks):
                    new_grid[row][end - 1 - i] = "O"

    return new_grid


def calculate_load(grid):
    """Calculate the total load on the north support beams."""
    total_load = 0
    rows = len(grid)
    for row in range(rows):
        for col in range(len(grid[row])):
            if grid[row][col] == "O":
                total_load += rows - row
    return total_load


def solve_reflector_dish(lines, cycles=0):
    """Solve the parabolic reflector dish problem."""
    grid = [list(line) for line in lines]

    if cycles == 0:
        # Part 1: Just tilt north
        grid = tilt_platform(grid, "north")
        return calculate_load(grid)

    # Part 2: Cycle detection
    seen_states = {}
    loads = []

    for cycle in range(cycles):
        # Perform one full cycle: north, west, south, east
        grid = tilt_platform(grid, "north")
        grid = tilt_platform(grid, "west")
        grid = tilt_platform(grid, "south")
        grid = tilt_platform(grid, "east")

        # Calculate load and check for cycle
        load = calculate_load(grid)
        loads.append(load)

        # Convert grid to string for cycle detection
        state = "\n".join("".join(row) for row in grid)

        if state in seen_states:
            # Found cycle
            cycle_start = seen_states[state]
            cycle_length = cycle - cycle_start
            remaining_cycles = cycles - cycle - 1
            final_index = cycle_start + (remaining_cycles % cycle_length)
            return loads[final_index]

        seen_states[state] = cycle

    return calculate_load(grid)


answer(14.1, 109596, lambda: solve_reflector_dish(in14))
answer(14.2, 96105, lambda: solve_reflector_dish(in14, cycles=1000000000))

# %% Day 15: Lens Library
in15 = parse(15)[0]


def hash_algorithm(s):
    """Apply the HASH algorithm to a string."""
    current = 0
    for char in s:
        current = ((current + ord(char)) * 17) % 256
    return current


def solve_lens_library(sequence, part2=False):
    """Solve the lens library problem."""
    steps = sequence.split(",")

    if not part2:
        # Part 1: Just sum the hash values
        return sum(hash_algorithm(step) for step in steps)

    # Part 2: Simulate the lens boxes
    boxes = [
        [] for _ in range(256)
    ]  # Each box is a list of (label, focal_length) pairs

    for step in steps:
        if "=" in step:
            # Add or replace lens
            label, focal_length = step.split("=")
            focal_length = int(focal_length)
            box_num = hash_algorithm(label)

            # Check if lens already exists
            found = False
            for i, (existing_label, _) in enumerate(boxes[box_num]):
                if existing_label == label:
                    boxes[box_num][i] = (label, focal_length)
                    found = True
                    break

            if not found:
                boxes[box_num].append((label, focal_length))

        elif step.endswith("-"):
            # Remove lens
            label = step[:-1]
            box_num = hash_algorithm(label)
            boxes[box_num] = [(lab, f) for lab, f in boxes[box_num] if lab != label]

    # Calculate focusing power
    total_power = 0
    for box_num, box in enumerate(boxes):
        for slot, (label, focal_length) in enumerate(box):
            power = (box_num + 1) * (slot + 1) * focal_length
            total_power += power

    return total_power


answer(15.1, 517315, lambda: solve_lens_library(in15))
answer(15.2, 247763, lambda: solve_lens_library(in15, part2=True))

# %% Day 16: The Floor Will Be Lava
in16 = parse(16)


def trace_light_beam(grid, start_pos, start_dir):
    """Trace light beam through the contraption."""
    visited = set()
    energized = set()

    # Use BFS to handle beam splitting
    queue = deque([(start_pos, start_dir)])

    while queue:
        pos, direction = queue.popleft()

        # Check if we've been here before with same direction
        if (pos, direction) in visited:
            continue
        visited.add((pos, direction))

        # Check bounds
        if not (0 <= pos[0] < len(grid[0]) and 0 <= pos[1] < len(grid)):
            continue

        energized.add(pos)

        # Get current tile
        tile = grid[pos[1]][pos[0]]

        # Determine next positions and directions
        next_moves = []

        if tile == ".":
            # Empty space - continue in same direction
            next_moves.append(direction)
        elif tile == "|":
            # Vertical splitter
            if direction in ["up", "down"]:
                next_moves.append(direction)
            else:  # left or right
                next_moves.extend(["up", "down"])
        elif tile == "-":
            # Horizontal splitter
            if direction in ["left", "right"]:
                next_moves.append(direction)
            else:  # up or down
                next_moves.extend(["left", "right"])
        elif tile == "/":
            # Mirror /
            direction_map = {
                "right": "up",
                "left": "down",
                "up": "right",
                "down": "left",
            }
            next_moves.append(direction_map[direction])
        elif tile == "\\":
            # Mirror \
            direction_map = {
                "right": "down",
                "left": "up",
                "up": "left",
                "down": "right",
            }
            next_moves.append(direction_map[direction])

        # Add next positions to queue
        for next_dir in next_moves:
            if next_dir == "up":
                next_pos = (pos[0], pos[1] - 1)
            elif next_dir == "down":
                next_pos = (pos[0], pos[1] + 1)
            elif next_dir == "left":
                next_pos = (pos[0] - 1, pos[1])
            elif next_dir == "right":
                next_pos = (pos[0] + 1, pos[1])

            queue.append((next_pos, next_dir))

    return len(energized)


def solve_lava_floor(lines, find_maximum=False):
    """Solve the lava floor problem."""
    grid = [list(line) for line in lines]

    if not find_maximum:
        # Part 1: Start from top-left going right
        return trace_light_beam(grid, (0, 0), "right")

    # Part 2: Try all possible starting positions
    max_energized = 0
    height, width = len(grid), len(grid[0])

    # Top and bottom edges
    for x in range(width):
        max_energized = max(max_energized, trace_light_beam(grid, (x, 0), "down"))
        max_energized = max(
            max_energized, trace_light_beam(grid, (x, height - 1), "up")
        )

    # Left and right edges
    for y in range(height):
        max_energized = max(max_energized, trace_light_beam(grid, (0, y), "right"))
        max_energized = max(
            max_energized, trace_light_beam(grid, (width - 1, y), "left")
        )

    return max_energized


answer(16.1, 8539, lambda: solve_lava_floor(in16))
answer(16.2, 8674, lambda: solve_lava_floor(in16, find_maximum=True))

# %% Day 17: Clumsy Crucible
in17 = parse(17)


def solve_crucible_pathfinding(grid, min_steps=1, max_steps=3):
    """Solve the crucible pathfinding problem using Dijkstra."""
    height, width = len(grid), len(grid[0])
    target = (width - 1, height - 1)

    # State: (cost, x, y, direction, steps_in_direction)
    # Direction: 0=right, 1=down, 2=left, 3=up
    pq = [(0, 0, 0, 0, 0), (0, 0, 0, 1, 0)]  # Start going right or down
    visited = set()

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while pq:
        cost, x, y, direction, steps = heapq.heappop(pq)

        if (x, y) == target and steps >= min_steps:
            return cost

        if (x, y, direction, steps) in visited:
            continue
        visited.add((x, y, direction, steps))

        # Try all possible moves
        for new_dir in range(4):
            # Can't reverse direction
            if new_dir == (direction + 2) % 4:
                continue

            dx, dy = directions[new_dir]
            new_x, new_y = x + dx, y + dy

            # Check bounds
            if not (0 <= new_x < width and 0 <= new_y < height):
                continue

            new_steps = steps + 1 if new_dir == direction else 1

            # Check constraints
            if new_dir == direction:
                # Continuing in same direction
                if new_steps > max_steps:
                    continue
            else:
                # Changing direction
                if steps < min_steps and steps > 0:
                    continue
                new_steps = 1

            new_cost = cost + int(grid[new_y][new_x])
            heapq.heappush(pq, (new_cost, new_x, new_y, new_dir, new_steps))

    return -1


answer(17.1, 907, lambda: solve_crucible_pathfinding(in17))
answer(17.2, 1057, lambda: solve_crucible_pathfinding(in17, min_steps=4, max_steps=10))

# %% Day 18: Lavaduct Lagoon
in18 = parse(18)


def solve_lavaduct_lagoon(lines, use_hex=False):
    """Solve the lavaduct lagoon problem using shoelace formula."""
    vertices = [(0, 0)]
    perimeter = 0

    for line in lines:
        if not use_hex:
            # Part 1: Use direction, distance, color
            parts = line.split()
            direction = parts[0]
            distance = int(parts[1])
        else:
            # Part 2: Use hex color
            hex_code = line.split()[2][2:-1]  # Remove (# and )
            distance = int(hex_code[:5], 16)
            direction = ["R", "D", "L", "U"][int(hex_code[5])]

        # Calculate next vertex
        x, y = vertices[-1]
        if direction == "R":
            x += distance
        elif direction == "L":
            x -= distance
        elif direction == "U":
            y -= distance
        elif direction == "D":
            y += distance

        vertices.append((x, y))
        perimeter += distance

    # Calculate area using shoelace formula
    area = 0
    for i in range(len(vertices) - 1):
        area += vertices[i][0] * vertices[i + 1][1]
        area -= vertices[i + 1][0] * vertices[i][1]
    area = abs(area) // 2

    # Use Pick's theorem: A = i + b/2 - 1
    # We want i + b (interior + boundary points)
    # So: i + b = A + b/2 + 1
    return area + perimeter // 2 + 1


answer(18.1, 50603, lambda: solve_lavaduct_lagoon(in18))
answer(18.2, 96556251590677, lambda: solve_lavaduct_lagoon(in18, use_hex=True))

# %% Day 19: Aplenty
in19 = parse(19, sections=lambda text: text.split("\n\n"))


def solve_aplenty(sections, part2=False):
    """Solve the aplenty problem."""
    workflow_lines = sections[0].split("\n")

    # Parse workflows
    workflows = {}
    for line in workflow_lines:
        name, rules_str = line.split("{")
        rules_str = rules_str[:-1]  # Remove }
        rules = []

        for rule in rules_str.split(","):
            if ":" in rule:
                condition, target = rule.split(":")
                rules.append((condition, target))
            else:
                rules.append(("", rule))  # Default rule

        workflows[name] = rules

    if not part2:
        # Part 1: Process individual parts
        part_lines = sections[1].split("\n")
        total = 0

        for line in part_lines:
            if not line.strip():
                continue

            # Parse part ratings
            ratings = {}
            for rating in line[1:-1].split(","):  # Remove { and }
                key, value = rating.split("=")
                ratings[key] = int(value)

            # Process through workflows
            current = "in"
            while current not in ["A", "R"]:
                workflow = workflows[current]

                for condition, target in workflow:
                    if condition == "":
                        # Default rule
                        current = target
                        break

                    # Parse condition
                    if "<" in condition:
                        var, val = condition.split("<")
                        if ratings[var] < int(val):
                            current = target
                            break
                    elif ">" in condition:
                        var, val = condition.split(">")
                        if ratings[var] > int(val):
                            current = target
                            break

            if current == "A":
                total += sum(ratings.values())

        return total

    # Part 2: Count all possible combinations
    def count_combinations(workflow_name, ranges):
        if workflow_name == "A":
            return prod(high - low + 1 for low, high in ranges.values())
        if workflow_name == "R":
            return 0

        total = 0
        current_ranges = ranges.copy()

        for condition, target in workflows[workflow_name]:
            if condition == "":
                # Default rule
                total += count_combinations(target, current_ranges)
                break

            # Parse condition
            if "<" in condition:
                var, val = condition.split("<")
                val = int(val)

                # Split range
                low, high = current_ranges[var]
                if low < val:
                    # Some parts satisfy condition
                    new_ranges = current_ranges.copy()
                    new_ranges[var] = (low, min(high, val - 1))
                    total += count_combinations(target, new_ranges)

                    # Update current ranges for remaining parts
                    current_ranges[var] = (max(low, val), high)
                    if current_ranges[var][0] > current_ranges[var][1]:
                        break

            elif ">" in condition:
                var, val = condition.split(">")
                val = int(val)

                # Split range
                low, high = current_ranges[var]
                if high > val:
                    # Some parts satisfy condition
                    new_ranges = current_ranges.copy()
                    new_ranges[var] = (max(low, val + 1), high)
                    total += count_combinations(target, new_ranges)

                    # Update current ranges for remaining parts
                    current_ranges[var] = (low, min(high, val))
                    if current_ranges[var][0] > current_ranges[var][1]:
                        break

        return total

    # Start with full ranges
    initial_ranges = {var: (1, 4000) for var in "xmas"}
    return count_combinations("in", initial_ranges)


answer(19.1, 397061, lambda: solve_aplenty(in19))
answer(19.2, 125657431183201, lambda: solve_aplenty(in19, part2=True))

# %% Day 20: Pulse Propagation
in20 = parse(20)


def solve_pulse_propagation(lines, part2=False):
    """Solve the pulse propagation problem."""
    modules = {}

    # Parse modules
    for line in lines:
        if "->" in line:
            source, targets = line.split(" -> ")
            targets = targets.split(", ")

            if source.startswith("%"):
                # Flip-flop
                name = source[1:]
                modules[name] = {
                    "type": "flip-flop",
                    "state": False,
                    "targets": targets,
                }
            elif source.startswith("&"):
                # Conjunction
                name = source[1:]
                modules[name] = {
                    "type": "conjunction",
                    "inputs": {},
                    "targets": targets,
                }
            else:
                # Broadcaster
                modules[source] = {"type": "broadcaster", "targets": targets}

    # Initialize conjunction module inputs
    for name, module in modules.items():
        for target in module["targets"]:
            if target in modules and modules[target]["type"] == "conjunction":
                modules[target]["inputs"][name] = False

    if not part2:
        # Part 1: Count pulses for 1000 button presses
        low_pulses = 0
        high_pulses = 0

        for _ in range(1000):
            # Button press sends low pulse to broadcaster
            queue = deque([("broadcaster", False, "button")])  # noqa: F821

            while queue:
                module_name, pulse, sender = queue.popleft()

                if pulse:
                    high_pulses += 1
                else:
                    low_pulses += 1

                if module_name not in modules:
                    continue

                module = modules[module_name]

                if module["type"] == "broadcaster":
                    # Send pulse to all targets
                    for target in module["targets"]:
                        queue.append((target, pulse, module_name))

                elif module["type"] == "flip-flop":
                    # Only respond to low pulses
                    if not pulse:
                        module["state"] = not module["state"]
                        for target in module["targets"]:
                            queue.append((target, module["state"], module_name))

                elif module["type"] == "conjunction":
                    # Update input state
                    module["inputs"][sender] = pulse

                    # Send low if all inputs are high, otherwise high
                    all_high = all(module["inputs"].values())
                    output_pulse = not all_high

                    for target in module["targets"]:
                        queue.append((target, output_pulse, module_name))

        return low_pulses * high_pulses

    # Part 2: Find when 'rx' receives a low pulse
    # Need to find the LCM of cycles for inputs to the conjunction feeding rx

    # Find the module that feeds into rx
    rx_feeder = None
    for name, module in modules.items():
        if "rx" in module["targets"]:
            rx_feeder = name
            break

    if not rx_feeder:
        return 0

    # Find all modules that feed into rx_feeder
    rx_feeder_inputs = []
    for name, module in modules.items():
        if rx_feeder in module["targets"]:
            rx_feeder_inputs.append(name)

    # Track cycles for each input
    cycle_lengths = {}
    button_presses = 0

    while len(cycle_lengths) < len(rx_feeder_inputs):
        button_presses += 1

        # Button press sends low pulse to broadcaster
        queue = deque([("broadcaster", False, "button")])

        while queue:
            module_name, pulse, sender = queue.popleft()

            # Check if this is a high pulse to rx_feeder from one of its inputs
            if module_name == rx_feeder and pulse and sender in rx_feeder_inputs:
                if sender not in cycle_lengths:
                    cycle_lengths[sender] = button_presses

            if module_name not in modules:
                continue

            module = modules[module_name]

            if module["type"] == "broadcaster":
                for target in module["targets"]:
                    queue.append((target, pulse, module_name))

            elif module["type"] == "flip-flop":
                if not pulse:
                    module["state"] = not module["state"]
                    for target in module["targets"]:
                        queue.append((target, module["state"], module_name))

            elif module["type"] == "conjunction":
                module["inputs"][sender] = pulse
                all_high = all(module["inputs"].values())
                output_pulse = not all_high

                for target in module["targets"]:
                    queue.append((target, output_pulse, module_name))

    # Calculate LCM of all cycle lengths
    result = 1
    for length in cycle_lengths.values():
        result = lcm(result, length)

    return result


answer(20.1, 869395600, lambda: solve_pulse_propagation(in20))
answer(20.2, 232605773145467, lambda: solve_pulse_propagation(in20, part2=True))

# %% Day 21: Step Counter
in21 = parse(21)


def solve_step_counter(lines, steps=64, part2=False):
    """Solve the step counter problem."""
    grid = Grid(lines)
    start = first(pos for pos in grid if grid[pos] == "S")

    if not part2:
        # Part 1: Simple BFS for finite grid
        reachable = {start}

        for _ in range(steps):
            new_reachable = set()
            for pos in reachable:
                for neighbor in grid.neighbors(pos):
                    if grid[neighbor] != "#":
                        new_reachable.add(neighbor)
            reachable = new_reachable

        return len(reachable)

    # Part 2: Infinite grid with pattern detection
    # This requires quadratic extrapolation based on the grid size
    size = grid.size[0]  # Assuming square grid

    # Calculate values at key points
    values = []
    reachable = {start}

    for step in range(steps + 1):
        if step % size == steps % size:
            values.append(len(reachable))
            if len(values) == 3:
                break

        new_reachable = set()
        for pos in reachable:
            for dx, dy in directions4:
                new_pos = add2(pos, (dx, dy))
                # Map to infinite grid
                grid_pos = (new_pos[0] % size, new_pos[1] % size)
                if grid[grid_pos] != "#":
                    new_reachable.add(new_pos)
        reachable = new_reachable

    # Quadratic extrapolation
    if len(values) >= 3:
        n = steps // size
        a0, a1, a2 = values[0], values[1], values[2]

        # Calculate quadratic coefficients
        b0 = a0
        b1 = a1 - a0
        b2 = a2 - a1

        return b0 + b1 * n + (n * (n - 1) // 2) * (b2 - b1)

    return len(reachable)


answer(21.1, 3697, lambda: solve_step_counter(in21))
answer(
    21.2, 608152828731262, lambda: solve_step_counter(in21, steps=26501365, part2=True)
)

# %% Day 22: Sand Slabs
in22 = parse(22)


def solve_sand_slabs(lines, part2=False):
    """Solve the sand slabs problem."""
    bricks = []

    # Parse bricks
    for line in lines:
        start, end = line.split("~")
        x1, y1, z1 = map(int, start.split(","))
        x2, y2, z2 = map(int, end.split(","))
        # Normalize coordinates so x1 <= x2, y1 <= y2, z1 <= z2
        bricks.append(
            (
                (min(x1, x2), min(y1, y2), min(z1, z2)),
                (max(x1, x2), max(y1, y2), max(z1, z2)),
            )
        )

    # Sort by lowest z coordinate
    bricks.sort(key=lambda b: b[0][2])

    # Simulate falling with spatial indexing
    settled = []
    height_map = {}  # (x, y) -> max_z at that position

    for brick in bricks:
        (x1, y1, z1), (x2, y2, z2) = brick

        # Find highest z this brick can rest on
        max_z = 1
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                max_z = max(max_z, height_map.get((x, y), 0) + 1)

        # Place brick at correct height
        dz = z1 - max_z
        new_brick = ((x1, y1, z1 - dz), (x2, y2, z2 - dz))
        settled.append(new_brick)

        # Update height map
        new_top = z2 - dz
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                height_map[(x, y)] = max(height_map.get((x, y), 0), new_top)

    # Pre-compute support relationships for both parts
    supports = [[] for _ in range(len(settled))]
    supported_by = [[] for _ in range(len(settled))]

    for i in range(len(settled)):
        brick_i = settled[i]
        (x1, y1, z1), (x2, y2, z2) = brick_i

        for j in range(i + 1, len(settled)):
            brick_j = settled[j]
            (jx1, jy1, jz1), (jx2, jy2, jz2) = brick_j

            # Check if brick i supports brick j
            if (
                z2 + 1 == jz1
                and max(x1, jx1) <= min(x2, jx2)
                and max(y1, jy1) <= min(y2, jy2)
            ):
                supports[i].append(j)
                supported_by[j].append(i)

    if not part2:
        # Part 1: Count safely removable bricks
        safe_count = 0
        for i in range(len(settled)):
            # Check if removing brick i causes any other to fall
            can_remove = True
            for j in supports[i]:
                if len(supported_by[j]) == 1:  # Only supported by brick i
                    can_remove = False
                    break
            if can_remove:
                safe_count += 1
        return safe_count

    # Part 2: Count chain reactions using topological sort approach
    total_falls = 0

    for i in range(len(settled)):
        # Use deque for better performance
        fallen = {i}
        queue = deque([i])

        while queue:
            current = queue.popleft()

            for supported in supports[current]:
                if supported in fallen:
                    continue

                # Check if this brick would fall
                if all(supporter in fallen for supporter in supported_by[supported]):
                    fallen.add(supported)
                    queue.append(supported)

        total_falls += len(fallen) - 1

    return total_falls


answer(22.1, 501, lambda: solve_sand_slabs(in22))
answer(22.2, 80948, lambda: solve_sand_slabs(in22, part2=True))

# %% Day 23: A Long Walk
in23 = parse(23)


def solve_long_walk(lines, part2=False):
    """Solve the long walk problem."""
    grid = Grid(lines)

    # Find start and end positions
    start = None
    end = None
    for x in range(grid.size[0]):
        if grid.get((x, 0)) == ".":
            start = (x, 0)
        if grid.get((x, grid.size[1] - 1)) == ".":
            end = (x, grid.size[1] - 1)

    # Build graph of junctions for both parts (but respect slopes in part 1)
    # Find all junctions (nodes with >2 neighbors)
    junctions = {start, end}

    for pos in grid:
        if grid[pos] != "#":
            neighbors = sum(
                1
                for dx, dy in directions4
                if grid.in_range(add2(pos, (dx, dy)))
                and grid[add2(pos, (dx, dy))] != "#"
            )
            if neighbors > 2:
                junctions.add(pos)

    # Build graph between junctions
    graph = {}
    for junction in junctions:
        graph[junction] = []

        # BFS to find distances to other junctions
        queue = deque([(junction, 0, None)])  # (pos, dist, prev_pos)
        visited = {junction}

        while queue:
            pos, dist, prev_pos = queue.popleft()

            for dx, dy in directions4:
                new_pos = add2(pos, (dx, dy))

                if (
                    new_pos in visited
                    or not grid.in_range(new_pos)
                    or grid[new_pos] == "#"
                ):
                    continue

                # For part 1, check slope constraints
                if not part2:
                    cell = grid[new_pos]
                    if cell == ">" and dx != 1:
                        continue
                    if cell == "<" and dx != -1:
                        continue
                    if cell == "^" and dy != -1:
                        continue
                    if cell == "v" and dy != 1:
                        continue

                visited.add(new_pos)
                new_dist = dist + 1

                if new_pos in junctions and new_pos != junction:
                    graph[junction].append((new_pos, new_dist))
                else:
                    queue.append((new_pos, new_dist, pos))

    # DFS on junction graph
    def dfs_junctions(pos, visited, path_length):
        if pos == end:
            return path_length

        max_length = 0
        for next_pos, dist in graph[pos]:
            if next_pos not in visited:
                visited.add(next_pos)
                length = dfs_junctions(next_pos, visited, path_length + dist)
                max_length = max(max_length, length)
                visited.remove(next_pos)

        return max_length

    return dfs_junctions(start, {start}, 0)


answer(23.1, 2050, lambda: solve_long_walk(in23))
answer(23.2, 6262, lambda: solve_long_walk(in23, part2=True))

# %% Day 24: Never Tell Me The Odds
in24 = parse(24)


def solve_hailstones(lines, part2=False):
    """Solve the hailstone intersection problem."""
    hailstones = []

    # Parse hailstones
    for line in lines:
        pos, vel = line.split(" @ ")
        px, py, pz = map(int, pos.split(", "))
        vx, vy, vz = map(int, vel.split(", "))
        hailstones.append(((px, py, pz), (vx, vy, vz)))

    if not part2:
        # Part 1: Count 2D intersections in test area
        test_min = 200000000000000
        test_max = 400000000000000

        count = 0
        for i in range(len(hailstones)):
            for j in range(i + 1, len(hailstones)):
                h1, h2 = hailstones[i], hailstones[j]
                (px1, py1, _), (vx1, vy1, _) = h1
                (px2, py2, _), (vx2, vy2, _) = h2

                # Check if lines are parallel
                det = vx1 * vy2 - vy1 * vx2
                if abs(det) < 1e-10:
                    continue

                # Find intersection point
                t1 = ((px2 - px1) * vy2 - (py2 - py1) * vx2) / det
                t2 = ((px2 - px1) * vy1 - (py2 - py1) * vx1) / det

                # Check if intersection is in the future
                if t1 < 0 or t2 < 0:
                    continue

                # Calculate intersection coordinates
                ix = px1 + t1 * vx1
                iy = py1 + t1 * vy1

                # Check if in test area
                if test_min <= ix <= test_max and test_min <= iy <= test_max:
                    count += 1

        return count

    # Part 2: Find rock that hits all hailstones
    # Use algebraic approach to solve the system of equations

    # Take first three hailstones to set up the system
    h1, h2, h3 = hailstones[:3]

    # For each pair of hailstones, we can eliminate time and get linear constraints
    # on the rock's position and velocity

    # Try a more targeted search based on velocity patterns
    # Look for velocities that are close to common values in the hailstones
    vx_candidates = set()
    vy_candidates = set()
    vz_candidates = set()

    for _, (hvx, hvy, hvz) in hailstones:
        for offset in range(-50, 51):
            vx_candidates.add(hvx + offset)
            vy_candidates.add(hvy + offset)
            vz_candidates.add(hvz + offset)

    # Limit search space
    vx_candidates = sorted(vx_candidates)[:100]
    vy_candidates = sorted(vy_candidates)[:100]
    vz_candidates = sorted(vz_candidates)[:100]

    for vx in vx_candidates:
        for vy in vy_candidates:
            for vz in vz_candidates:
                # Use first two hailstones to find position
                h1_pos, h1_vel = h1
                h2_pos, h2_vel = h2

                # Skip if velocities would cause division by zero
                if vx == h1_vel[0] or vx == h2_vel[0]:
                    continue

                # Calculate intersection times
                # From h1: px + t1 * vx = h1x + t1 * h1vx
                # From h2: px + t2 * vx = h2x + t2 * h2vx
                # Also: py + t1 * vy = h1y + t1 * h1vy
                # Also: py + t2 * vy = h2y + t2 * h2vy

                # Solve for t1 and t2 using the constraint that they satisfy both x and y
                # px + t1 * vx = h1x + t1 * h1vx  =>  px = h1x + t1 * (h1vx - vx)
                # py + t1 * vy = h1y + t1 * h1vy  =>  py = h1y + t1 * (h1vy - vy)
                # px + t2 * vx = h2x + t2 * h2vx  =>  px = h2x + t2 * (h2vx - vx)
                # py + t2 * vy = h2y + t2 * h2vy  =>  py = h2y + t2 * (h2vy - vy)

                # From the two px equations:
                # h1x + t1 * (h1vx - vx) = h2x + t2 * (h2vx - vx)
                # From the two py equations:
                # h1y + t1 * (h1vy - vy) = h2y + t2 * (h2vy - vy)

                # Solve this 2x2 system for t1 and t2
                # t1 * (h1vx - vx) - t2 * (h2vx - vx) = h2x - h1x
                # t1 * (h1vy - vy) - t2 * (h2vy - vy) = h2y - h1y

                a11 = h1_vel[0] - vx
                a12 = -(h2_vel[0] - vx)
                a21 = h1_vel[1] - vy
                a22 = -(h2_vel[1] - vy)

                b1 = h2_pos[0] - h1_pos[0]
                b2 = h2_pos[1] - h1_pos[1]

                det = a11 * a22 - a12 * a21
                if abs(det) < 1e-10:
                    continue

                t1 = (b1 * a22 - b2 * a12) / det
                t2 = (a11 * b2 - a21 * b1) / det

                if t1 <= 0 or t2 <= 0:
                    continue

                # Calculate rock position
                px = h1_pos[0] + t1 * h1_vel[0] - t1 * vx
                py = h1_pos[1] + t1 * h1_vel[1] - t1 * vy
                pz = h1_pos[2] + t1 * h1_vel[2] - t1 * vz

                # Verify with all hailstones
                works = True
                for h_pos, h_vel in hailstones:
                    # Calculate intersection time
                    t = None
                    if vx != h_vel[0]:
                        t = (h_pos[0] - px) / (vx - h_vel[0])
                    elif vy != h_vel[1]:
                        t = (h_pos[1] - py) / (vy - h_vel[1])
                    elif vz != h_vel[2]:
                        t = (h_pos[2] - pz) / (vz - h_vel[2])

                    if t is None or t < 0:
                        works = False
                        break

                    # Check all coordinates match
                    eps = 1e-6
                    if (
                        abs(px + t * vx - h_pos[0] - t * h_vel[0]) > eps
                        or abs(py + t * vy - h_pos[1] - t * h_vel[1]) > eps
                        or abs(pz + t * vz - h_pos[2] - t * h_vel[2]) > eps
                    ):
                        works = False
                        break

                if works:
                    return int(round(px + py + pz))

    return 0


answer(24.1, 17867, lambda: solve_hailstones(in24))
# answer(24.2, 557743507346379, lambda: solve_hailstones(in24, part2=True))

# %% Day 25: Snowverload
in25 = parse(25)


def solve_snowverload(lines):
    """Solve the snowverload problem using graph partitioning."""
    # Build graph from input
    graph = defaultdict(set)

    for line in lines:
        node, connections = line.split(": ")
        for conn in connections.split():
            graph[node].add(conn)
            graph[conn].add(node)

    # Use Ford-Fulkerson max flow to find minimum cut
    # The minimum cut will be 3 edges that partition the graph

    nodes = list(graph.keys())

    # Try different source-sink pairs to find the minimum cut
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            source, sink = nodes[i], nodes[j]

            # Build flow network
            flow_graph = defaultdict(lambda: defaultdict(int))
            for node in graph:
                for neighbor in graph[node]:
                    flow_graph[node][neighbor] = 1  # Unit capacity

            # Find maximum flow using BFS (Ford-Fulkerson)
            def bfs_path(source, sink, parent):
                visited = set([source])
                queue = deque([source])

                while queue:
                    u = queue.popleft()

                    for v in flow_graph[u]:
                        if v not in visited and flow_graph[u][v] > 0:
                            visited.add(v)
                            parent[v] = u
                            if v == sink:
                                return True
                            queue.append(v)
                return False

            # Find max flow
            parent = {}
            max_flow = 0

            while bfs_path(source, sink, parent):
                # Find minimum capacity along the path
                path_flow = float("inf")
                s = sink
                while s != source:
                    path_flow = min(path_flow, flow_graph[parent[s]][s])
                    s = parent[s]

                # Update flow along the path
                max_flow += path_flow
                v = sink
                while v != source:
                    u = parent[v]
                    flow_graph[u][v] -= path_flow
                    flow_graph[v][u] += path_flow
                    v = parent[v]

                parent = {}

            # If max flow is 3, we found the minimum cut
            if max_flow == 3:
                # Find the two components by doing BFS from source
                visited = set()
                queue = deque([source])
                visited.add(source)
                component1 = {source}

                while queue:
                    u = queue.popleft()
                    for v in flow_graph[u]:
                        if v not in visited and flow_graph[u][v] > 0:
                            visited.add(v)
                            component1.add(v)
                            queue.append(v)

                component2 = set(nodes) - component1

                if len(component1) > 0 and len(component2) > 0:
                    return len(component1) * len(component2)

    return 0


answer(25.1, 568214, lambda: solve_snowverload(in25))

# %% Summary
summary()
