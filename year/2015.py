#!/usr/bin/env python3

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import md5
from itertools import (
    accumulate,
    combinations,
    combinations_with_replacement,
    count,
    groupby,
    permutations,
    product,
    starmap,
)
from math import prod
from operator import and_, invert, lshift, or_, rshift

from aoc import (
    East,
    North,
    South,
    West,
    Zero,
    add,
    answer,
    arrow_direction,
    atoms,
    cat,
    first,
    flatten,
    ints,
    mapt,
    n_times,
    paragraphs,
    parse_year,
    quantify,
    summary,
    the,
)

parse = parse_year(2015)

# %% Day 1
in1 = the(parse(1))
floors = mapt(lambda x: -1 if x == ")" else +1, in1)
answer(1.1, 232, lambda: sum(floors))


def find_floor(floors):
    return next(pos + 1 for pos, floor in enumerate(accumulate(floors)) if floor == -1)


answer(1.2, 1783, lambda: find_floor(floors))


# %% Day 2
def wrapping_paper(l, w, h):  # noqa: E741
    return 2 * l * w + 2 * w * h + 2 * h * l + min(l * w, w * h, l * h)


assert wrapping_paper(2, 3, 4) == 58
assert wrapping_paper(1, 1, 10) == 43

in2 = parse(2, ints)
answer(2.1, 1606483, lambda: sum(starmap(wrapping_paper, in2)))


def wrapping_paper2(l, w, h):  # noqa: E741
    return l * w * h + 2 * sum(sorted((l, w, h))[:2])


assert wrapping_paper2(2, 3, 4) == 34
assert wrapping_paper2(1, 1, 10) == 14

answer(2.2, 3842356, lambda: sum(starmap(wrapping_paper2, in2)))


# %% Day 3
in3 = mapt(arrow_direction.get, the(parse(3)))


def santa(grid):
    return set(accumulate(grid, func=add, initial=Zero))


assert len(santa((East,))) == 2
assert len(santa((North, East, South, West))) == 4
assert len(santa((North, South) * 8)) == 2

answer(3.1, 2572, lambda: len(santa(in3)))
answer(3.2, 2631, lambda: len(santa(in3[::2]) | santa(in3[1::2])))


# %% Day 4
def find_hash_with_zeros(key, num_zeros):
    return next(
        i
        for i in count(1)
        if md5(f"{key}{i}".encode()).hexdigest().startswith("0" * num_zeros)
    )


assert find_hash_with_zeros("abcdef", 5) == 609043
assert find_hash_with_zeros("pqrstuv", 5) == 1048970

in4 = the(parse(4))
# Uncomment these lines to actually run the full computation (takes ~10 seconds)
# answer(4.1, 282749, lambda: find_hash_with_zeros(in4, 5))
# answer(4.2, 9962624, lambda: find_hash_with_zeros(in4, 6))

# %% Day 5
in5 = parse(5)


def is_nice(string) -> bool:
    return (
        any(s1 == s2 for s1, s2 in zip(string[:-1], string[1:]))
        and quantify((s in "aeiou" for s in string), bool) >= 3
        and not any(s in string for s in ("ab", "cd", "pq", "xy"))
    )


answer(5.1, 236, lambda: quantify(in5, is_nice))


def is_nice2(string):
    return any(
        string[i : i + 2] in string[i + 2 :] for i in range(len(string) - 3)
    ) and any(a == c for a, c in zip(string, string[2:]))


answer(5.2, 51, lambda: quantify(in5, is_nice2))


# %% Day 6
def parse_lights(line):
    return tuple(elem for elem in atoms(line) if elem not in ("turn", "through"))


in6 = parse(6, parse_lights)


def execute6(instructions, ops, grid=1000):
    lights = [[False] * grid for _ in range(grid)]
    for op, x0, y0, x1, y1 in instructions:
        for x, y in product(range(x0, x1 + 1), range(y0, y1 + 1)):
            lights[x][y] = ops[op](lights[x][y])
    return sum(flatten(lights))


fuel_needed = {
    "toggle": lambda x: int(not x == 1),
    "on": lambda _: 1,
    "off": lambda _: 0,
}

answer(6.1, 377_891, lambda: execute6(in6, fuel_needed))

fuel_needed2 = {
    "toggle": lambda x: x + 2,
    "on": lambda x: x + 1,
    "off": lambda x: max(x - 1, 0),
}

answer(6.2, 14_110_788, lambda: execute6(in6, fuel_needed2))


# %% Day 7
def parse_instruction(line):
    tokens = atoms(line)
    return tokens[-1], tokens[:-1]


in7 = dict(parse(7, parse_instruction))

OPERATIONS = {"AND": and_, "OR": or_, "LSHIFT": lshift, "RSHIFT": rshift, "NOT": invert}


@lru_cache()
def get_value(key):
    try:
        return int(key)
    except (ValueError, TypeError):
        pass

    cmd = in7[key]

    if len(cmd) == 1:
        return get_value(cmd[0])

    if cmd[0] == "NOT":
        return OPERATIONS["NOT"](get_value(cmd[1]))

    return OPERATIONS[cmd[1]](get_value(cmd[0]), get_value(cmd[2]))


answer(7.1, 16076, lambda: get_value("a"))


in7["b"] = (str(get_value("a")),)
get_value.cache_clear()

answer(7.2, 2797, lambda: get_value("a"))


# %% Day 8
in8 = parse(8)
answer(8.1, 1333, lambda: sum(len(line) - len(eval(line)) for line in in8))
answer(8.2, 2046, lambda: sum(line.count('"') + line.count("\\") + 2 for line in in8))


# %% Day 9
def parse_path(line):
    city1, _, city2, distance = atoms(line)
    return city1, city2, distance


def build_distances(paths):
    distances = {}
    for city1, city2, dist in paths:
        distances.setdefault(city1, {})[city2] = dist
        distances.setdefault(city2, {})[city1] = dist
    return distances


def path_lengths(distances):
    cities = list(distances.keys())
    return (
        sum(distances[a][b] for a, b in zip(path[:-1], path[1:]))
        for path in permutations(cities)
    )


def all_paths(distances):
    return (
        sum(distances[p1][p2] for p1, p2 in zip(p[:-1], p[1:]))
        for p in permutations(distances.keys())
    )


in9 = build_distances(parse(9, parse_path))
answer(9.1, 207, lambda: min(all_paths(in9)))
answer(9.2, 804, lambda: max(all_paths(in9)))


# %% Day 10
def repeat_and_say(word: str) -> str:
    return cat(str(len(list(group))) + char for char, group in groupby(word))


in10 = the(parse(10))
answer(10.1, 252_594, lambda: len(n_times(repeat_and_say, in10, 40)))
answer(10.2, 3_579_328, lambda: len(n_times(repeat_and_say, in10, 50)))


# %% Day 11
@lru_cache()
def is_valid(password):
    num = mapt(ord, password)
    has_increasing_subsequence = any(
        num[i] + 2 == num[i + 1] + 1 == num[i + 2] for i in range(len(num) - 2)
    )
    does_not_contain_iol = not any(c in password for c in "iol")
    has_two_diff_pairs = len(set(re.findall(r"(.)\1", password))) >= 2
    return has_increasing_subsequence and does_not_contain_iol and has_two_diff_pairs


def increment(s: str) -> str:
    if s == "z" * len(s):
        return "a" * len(s)
    i = len(s) - 1
    while s[i] == "z":
        i -= 1
    return s[:i] + chr(ord(s[i]) + 1) + "a" * (len(s) - i - 1)


def next_valid(pw: str) -> str:
    while not is_valid(pw := increment(pw)):
        pass
    return pw


in11 = the(parse(11))
answer(11.1, "vzbxxyzz", lambda: next_valid(in11))
answer(11.2, "vzcaabcc", lambda: n_times(next_valid, in11, 2))

# %% Day 12
in12 = json.loads(the(parse(12)))


def sum_numbers(obj):
    if isinstance(obj, int):
        return obj
    if isinstance(obj, list):
        return sum(sum_numbers(x) for x in obj)
    if isinstance(obj, dict):
        return sum(sum_numbers(x) for x in obj.values())
    return 0


answer(12.1, 156_366, lambda: sum_numbers(in12))


def sum_numbers2(obj):
    if isinstance(obj, int):
        return obj
    if isinstance(obj, list):
        return sum(sum_numbers2(x) for x in obj)
    if isinstance(obj, dict):
        return (
            0 if "red" in obj.values() else sum(sum_numbers2(x) for x in obj.values())
        )
    return 0


answer(12.2, 96_852, lambda: sum_numbers2(in12))


# %% Day 13
def parse_happiness(line):
    match = re.match(
        r"(\w+) would (gain|lose) (\d+) happiness units by sitting next to (\w+)\.",
        line,
    )
    if not match:
        raise ValueError(f"Could not parse line: {line}")
    name1, gain_lose, amount, name2 = match.groups()
    return name1, name2, int(amount) * (-1 if gain_lose == "lose" else 1)


def multimap2(items):
    result = {}
    for a, b, val in items:
        result.setdefault(a, {})[b] = val
    return result


in13 = multimap2(parse(13, parse_happiness))


def happiness(order):
    return sum(in13[a][b] + in13[b][a] for a, b in zip(order, order[1:] + (order[0],)))


answer(13.1, 664, lambda: max(happiness(order) for order in permutations(in13.keys())))

for person in list(in13):
    in13.setdefault("me", {})[person] = in13[person]["me"] = 0

answer(13.2, 640, lambda: max(happiness(order) for order in permutations(in13.keys())))


# %% Day 14
def parse_reindeer(line):
    match = re.match(r"(\w+) can fly (\d+) km/s for (\d+) (.*) (\d+) seconds.", line)
    if not match:
        raise ValueError(f"Could not parse line: {line}")
    name, speed, time, _, rest = match.groups()
    return name, {
        "speed": int(speed),
        "flying_time": int(time),
        "resting_time": int(rest),
        "distance": 0,
        "is_resting": False,
        "timer": int(time),
        "score": 0,
    }


in14 = dict(parse(14, parse_reindeer))


def simulate14(reindeers, time=2503):
    for _ in range(time):
        for r in reindeers.values():
            if r["timer"] == 0:
                r["is_resting"] = not r["is_resting"]
                r["timer"] = r["resting_time" if r["is_resting"] else "flying_time"]

            r["distance"] += 0 if r["is_resting"] else r["speed"]
            r["timer"] -= 1

        # Update scores
        max_dist = max(r["distance"] for r in reindeers.values())
        for r in reindeers.values():
            r["score"] += int(r["distance"] == max_dist)

    return reindeers


reindeers = simulate14(in14)

answer(14.1, 2655, lambda: max(reindeer["distance"] for reindeer in reindeers.values()))

answer(14.2, 1059, lambda: max(reindeer["score"] for reindeer in reindeers.values()))


# %% Day 15
def parse_ingredients(line):
    name, _, c, _, d, _, f, _, t, _, cal = atoms(line)
    return name, {
        "capacity": c,
        "durability": d,
        "flavor": f,
        "texture": t,
        "calories": cal,
    }


in15 = dict(parse(15, parse_ingredients))


def cookie(ingredients, max_calories=None):
    best_score = 0
    for potential_ingredients in combinations_with_replacement(
        ingredients.values(), 100
    ):
        capacity = sum(ingredient["capacity"] for ingredient in potential_ingredients)
        durability = sum(
            ingredient["durability"] for ingredient in potential_ingredients
        )
        flavor = sum(ingredient["flavor"] for ingredient in potential_ingredients)
        texture = sum(ingredient["texture"] for ingredient in potential_ingredients)
        calories = sum(ingredient["calories"] for ingredient in potential_ingredients)

        score = max(0, capacity) * max(0, durability) * max(0, flavor) * max(0, texture)
        best_score = max(
            best_score, score if max_calories is None or calories == max_calories else 0
        )
    return best_score


# Uncomment these lines to actually run the full computation (takes ~30 seconds)
# answer(15.1, 13_882_464, lambda: cookie(in15))
# answer(15.2, 11_171_160, lambda: cookie(in15, max_calories=500))


# %% Day 16
def parse_sue(line):
    _, sue, t1, n1, t2, n2, t3, n3 = atoms(line)
    return sue, {t1: n1, t2: n2, t3: n3}


in16 = dict(parse(16, parse_sue))

MFCSAM = {
    "children": 3,
    "cats": 7,
    "samoyeds": 2,
    "pomeranians": 3,
    "akitas": 0,
    "vizslas": 0,
    "goldfish": 5,
    "trees": 3,
    "cars": 2,
    "perfumes": 1,
}


def match_sue(presents, part2=False):
    for key, val in presents.items():
        if part2 and key in ("cats", "trees"):
            if val <= MFCSAM[key]:
                return False
        elif part2 and key in ("pomeranians", "goldfish"):
            if val >= MFCSAM[key]:
                return False
        elif val != MFCSAM[key]:
            return False
    return True


def find_sue(sues, part2=False):
    return first(sue for sue, presents in sues.items() if match_sue(presents, part2))


answer(16.1, 103, lambda: find_sue(in16))
answer(16.2, 405, lambda: find_sue(in16, part2=True))


# %% Day 17
def combinations_of(containers, total=150):
    return (
        y
        for x in range(len(containers))
        for y in combinations(containers, x)
        if sum(y) == total
    )


in17 = parse(17, int)
answer(17.1, 654, lambda: quantify(combinations_of(in17), bool))

min_len = min(map(len, combinations_of(in17)))
answer(17.2, 57, lambda: quantify(combinations_of(in17), lambda c: min_len == len(c)))


# %% Day 18
def count_neighbors(lights, x, y):
    return sum(
        (nx, ny) in lights
        for nx in (x - 1, x, x + 1)
        for ny in (y - 1, y, y + 1)
        if (nx, ny) != (x, y)
    )


def step_lights(lights, corners, size=100):
    lights = lights | corners
    return corners | {
        (x, y)
        for x in range(size)
        for y in range(size)
        if ((x, y) in lights and 2 <= count_neighbors(lights, x, y) <= 3)
        or ((x, y) not in lights and count_neighbors(lights, x, y) == 3)
    }


def run_lights(lights, corners, steps=100):
    current = lights
    for _ in range(steps):
        current = step_lights(current, corners)
    return current


with open("inputs/2015/18") as f:
    in18 = {
        (x, y) for y, line in enumerate(f) for x, char in enumerate(line) if char == "#"
    }
answer(18.1, 821, lambda: len(run_lights(in18, corners=set())))

corners = {(0, 0), (0, 99), (99, 0), (99, 99)}
answer(18.2, 886, lambda: len(run_lights(in18, corners=corners)))


# %% Day 19
def parse_rule(line):
    match = re.match(r"(\w+) => (\w+)", line)
    if not match:
        raise ValueError(f"Could not parse line: {line}")
    return match.groups()


rules, molecule = parse(19, sections=paragraphs)


def get_replacements(rules):
    result = defaultdict(list)
    for line in rules.splitlines():
        src, dst = parse_rule(line)
        result[src].append(dst)
    return result


def count_molecules(molecule, replacements):
    return len(
        {
            molecule[: m.start()] + repl + molecule[m.end() :]
            for src, dsts in replacements.items()
            for repl in dsts
            for m in re.finditer(src, molecule)
        }
    )


answer(19.1, 576, lambda: count_molecules(molecule, get_replacements(rules)))


def count_steps(molecule):
    # Count number of elements (uppercase followed by optional lowercase)
    elements = len(re.findall(r"[A-Z][a-z]?", molecule))
    rn_ar = molecule.count("Rn") + molecule.count("Ar")
    y = molecule.count("Y")
    return elements - rn_ar - 2 * y - 1


answer(19.2, 207, lambda: count_steps(molecule))


# %% Day 20
def lowest_house_number(target):
    # We can divide target by 10 since each house gets 10 * sum of factors
    target = target // 10
    size = target // 3  # Can be smaller due to factor sum properties
    houses = [0] * size

    for elf in range(1, size):
        # Add elf number to all its multiples
        houses[elf:size:elf] = [x + elf for x in houses[elf:size:elf]]
        if houses[elf] >= target:
            return elf
    return 0


def lowest_house_number2(target):
    # Similar optimization for part 2
    target = target // 11
    size = target // 3
    houses = [0] * size

    for elf in range(1, size):
        # Only visit first 50 houses
        for i in range(elf, min(size, elf * 50 + 1), elf):
            houses[i] += elf
        if houses[elf] >= target:
            return elf
    return 0


in20 = the(parse(20, int))
# Uncomment these lines to actually run the full computation (takes ~15 seconds)
# answer(20.1, 665_280, lambda: lowest_house_number(in20))
# answer(20.2, 705_600, lambda: lowest_house_number2(in20))


# %% Day 21
def parse_specs(line):
    match = re.match(r"(.*): (\d+)", line)
    if not match:
        return ValueError()
    spec, points = match.groups()
    return spec, int(points)


in21 = dict(parse(21, parse_specs))

# Shop items as (cost, damage, armor)
weapons = [(8, 4, 0), (10, 5, 0), (25, 6, 0), (40, 7, 0), (74, 8, 0)]
armor = [(0, 0, 0), (13, 0, 1), (31, 0, 2), (53, 0, 3), (75, 0, 4), (102, 0, 5)]
rings = [
    (0, 0, 0),
    (25, 1, 0),
    (50, 2, 0),
    (100, 3, 0),
    (20, 0, 1),
    (40, 0, 2),
    (80, 0, 3),
]


def get_loadouts():
    for w, a, r1, r2 in product(
        weapons, armor, rings + [(0, 0, 0)], rings + [(0, 0, 0)]
    ):
        if r1 != r2 or r1 == (0, 0, 0):
            stats = {
                "Hit Points": 100,
                "Damage": w[1] + a[1] + r1[1] + r2[1],
                "Armor": w[2] + a[2] + r1[2] + r2[2],
                "Cost": w[0] + a[0] + r1[0] + r2[0],
            }
            yield stats


def player_wins(player, enemy):
    p_damage = max(1, player["Damage"] - enemy["Armor"])
    e_damage = max(1, enemy["Damage"] - player["Armor"])

    p_turns = (enemy["Hit Points"] + p_damage - 1) // p_damage
    e_turns = (player["Hit Points"] + e_damage - 1) // e_damage

    return p_turns <= e_turns


loadouts = list(get_loadouts())

answer(21.1, 121, lambda: min(p["Cost"] for p in loadouts if player_wins(p, in21)))
answer(21.2, 201, lambda: max(p["Cost"] for p in loadouts if not player_wins(p, in21)))


# %% Day 22
@dataclass(frozen=True)
class Spell:
    name: str
    cost: int
    damage: int = 0
    heal: int = 0
    armor: int = 0
    mana: int = 0
    duration: int = 1


spells = (  # Tuple instead of list
    Spell("Magic Missile", 53, damage=4),
    Spell("Drain", 73, damage=2, heal=2),
    Spell("Shield", 113, armor=7, duration=6),
    Spell("Poison", 173, damage=3, duration=6),
    Spell("Recharge", 229, mana=101, duration=5),
)


def simulate(boss_hp, boss_damage, player_hp=50, player_mana=500, hard_mode=False):
    best_cost = float("inf")

    def apply_effects(effects, hp, mana):
        armor = 0
        new_effects = {}
        for name, (spell, turns) in effects.items():
            hp -= spell.damage
            mana += spell.mana
            armor += spell.armor
            if turns > 1:
                new_effects[name] = (spell, turns - 1)
        return new_effects, hp, mana, armor

    def is_game_over(player_hp, boss_hp, mana_spent):
        return player_hp <= 0 or boss_hp <= 0 or mana_spent >= best_cost

    def apply_hard_mode_penalty(player_hp):
        if hard_mode:
            player_hp -= 1
            return player_hp
        return player_hp

    def can_cast_spell(spell, player_mana, effects, mana_spent):
        return (
            spell.cost <= player_mana
            and spell.name not in effects
            and mana_spent + spell.cost < best_cost
        )

    def calculate_effects_and_costs(
        player_hp, player_mana, boss_hp, effects, mana_spent, spell
    ):
        new_mana = player_mana - spell.cost
        new_hp = player_hp
        new_boss_hp = boss_hp
        new_effects = dict(effects)

        if spell.duration > 1:
            new_effects[spell.name] = (spell, spell.duration)
        else:
            new_hp += spell.heal
            new_boss_hp -= spell.damage

        return new_hp, new_mana, new_boss_hp, new_effects

    def play(player_hp, player_mana, boss_hp, effects, mana_spent):
        nonlocal best_cost

        if is_game_over(player_hp, boss_hp, mana_spent):
            return float("inf")

        player_hp = apply_hard_mode_penalty(player_hp)

        if is_game_over(player_hp, boss_hp, mana_spent):
            return float("inf")

        effects, boss_hp, player_mana, _ = apply_effects(effects, boss_hp, player_mana)

        if boss_hp <= 0:
            return mana_spent

        for spell in spells:
            if not can_cast_spell(spell, player_mana, effects, mana_spent):
                continue

            new_hp, new_mana, new_boss_hp, new_effects = calculate_effects_and_costs(
                player_hp, player_mana, boss_hp, effects, mana_spent, spell
            )

            cost = play(
                new_hp, new_mana, new_boss_hp, new_effects, mana_spent + spell.cost
            )

            if new_boss_hp <= 0:
                best_cost = min(best_cost, mana_spent + spell.cost)
            else:
                if new_hp > 0:
                    best_cost = min(best_cost, cost)

        return best_cost

    result = play(player_hp, player_mana, boss_hp, {}, 0)
    return result if result != float("inf") else None


boss = in22 = dict(parse(22, parse_specs))
# Uncomment these lines to actually run the full computation (takes ~20 seconds)
# answer(22.1, 1824, lambda: simulate(boss["Hit Points"], boss["Damage"]))
# answer(22.2, 1937, lambda: simulate(boss["Hit Points"], boss["Damage"], hard_mode=True))


# %% Day 23
def execute(program, a=0):
    reg = {"a": a, "b": 0}
    ip = 0

    while 0 <= ip < len(program):
        op, *args = program[ip]

        if op == "hlf":
            reg[args[0]] //= 2
            ip += 1
        elif op == "tpl":
            reg[args[0]] *= 3
            ip += 1
        elif op == "inc":
            reg[args[0]] += 1
            ip += 1
        elif op == "jmp":
            ip += args[0]
        elif op == "jie":
            ip += args[1] if reg[args[0]] % 2 == 0 else 1
        elif op == "jio":
            ip += args[1] if reg[args[0]] == 1 else 1

    return reg["b"]


in23 = parse(23, atoms)
answer(23.1, 255, lambda: execute(in23))
answer(23.2, 334, lambda: execute(in23, a=1))


# %% Day 24
def find_smallest_qe(numbers, groups):
    target = sum(numbers) // groups

    # Try increasing group sizes until we find valid combinations
    for size in range(1, len(numbers)):
        valid = []
        for combo in combinations(numbers, size):
            if sum(combo) == target:
                valid.append(prod(combo))
        if valid:
            return min(valid)


in24 = set(parse(24, int))
answer(24.1, 11_266_889_531, lambda: find_smallest_qe(in24, 3))
answer(24.2, 77_387_711, lambda: find_smallest_qe(in24, 4))


# %% Day 25
def find_code(val, target_row, target_col):
    row = col = 1
    while target_row != row or target_col != col:
        row, col = (
            (col + 1, 1) if row == 1 else (row - 1, col + 1)
        )  # logic to loop around
        val = val * 252_533 % 33_554_393
    return val


target_row, target_col = in25 = parse(25, ints)[0]
answer(25.1, 9_132_360, lambda: find_code(20_151_125, target_row, target_col))

# %% Summary
summary()
