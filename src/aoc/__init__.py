# Adopted from github.com:norvig/pytudes
import functools
import heapq
import operator
import pathlib
import re
import time
from collections import Counter, abc, defaultdict, deque
from itertools import chain, combinations, islice
from math import gcd, inf
from statistics import mean, median
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

lines = str.splitlines  # By default, split input text into lines


def paragraphs(text):
    "Split text into paragraphs"
    return text.split("\n\n")


def parse_year(current_year):
    return functools.partial(parse, current_year=current_year)


def parse(
    day_or_text: Union[int, str],
    parser: Callable[[str], Any] = str,
    sections=lines,
    show=0,
    current_year=2023,
) -> Tuple:
    """Split the input text into `sections`, and apply `parser` to each.
    The first argument is either the text itself, or the day number of a text file."""
    if isinstance(day_or_text, str) and show == 8:
        show = 0  # By default, don't show lines when parsing example text.
    if isinstance(day_or_text, str):
        text = day_or_text
    else:
        filename = f"inputs/{current_year}/{day_or_text}"
        text = pathlib.Path(filename).read_text()
    show_items("Puzzle input", text.splitlines(), show)
    records = mapt(parser, sections(text.rstrip()))
    if parser is not str or sections != lines:
        show_items("Parsed representation", records, show)
    return records


def show_items(source, items, show: int, hr="─" * 100):
    """Show the first few items, in a pretty format."""
    if show:
        types = Counter(map(type, items))
        counts = ", ".join(
            f'{n} {t.__name__}{"" if n == 1 else "s"}' for t, n in types.items()
        )
        print(f"{hr}\n{source} ➜ {counts}:\n{hr}")
        for line in items[:show]:
            print(truncate(line))
        if show < len(items):
            print("...")


Char = str  # Intended as the type of a one-character string
Atom = Union[str, float, int]  # The type of a string or number
Ints = Sequence[int]


def ints(text: str) -> Tuple[int]:
    """A tuple of all the integers in text, ignoring non-number characters."""
    return mapt(int, re.findall(r"-?[0-9]+", text))


def positive_ints(text: str) -> Tuple[int]:
    """A tuple of all the integers in text, ignoring non-number characters."""
    return mapt(int, re.findall(r"[0-9]+", text))


def digits(text: str) -> Tuple[int]:
    """A tuple of all the digits in text (as ints 0–9), ignoring non-digit characters."""
    return mapt(int, re.findall(r"[0-9]", text))


def words(text: str) -> Tuple[str]:
    """A tuple of all the alphabetic words in text, ignoring non-letters."""
    return tuple(re.findall(r"[a-zA-Z]+", text))


def atoms(text: str) -> Tuple[Atom, ...]:
    """A tuple of all the atoms (numbers or identifiers) in text. Skip punctuation."""
    return mapt(atom, re.findall(r"[+-]?\d+\.?\d*|\w+", text))


def atom(text: str) -> Atom:
    """Parse text into a single float or int or str."""
    try:
        x = float(text)
        return round(x) if x.is_integer() else x
    except ValueError:
        return text.strip()


def n_times(fn, init: str, times: int) -> str:
    return functools.reduce(lambda x, _: fn(x), range(times), init)


answers = {}  # `answers` is a dict of {puzzle_number: answer}

unknown = "unknown"


class answer:
    """Verify that calling `code` computes the `solution` to `puzzle`.
    Record results in the dict `answers`."""

    def __init__(self, puzzle: float, solution, code: Callable = lambda: unknown):
        self.puzzle, self.solution, self.code = puzzle, solution, code
        answers[puzzle] = self
        self.check()

    def check(self) -> bool:
        """Check if the code computes the correct solution; record run time."""
        start = time.time()
        self.got = self.code()
        self.secs = time.time() - start
        self.ok = self.got == self.solution
        assert self.ok, self.__repr__()
        return self.ok

    def __repr__(self) -> str:
        """The repr of an answer shows what happened."""
        secs = f"{self.secs:6.3f}".replace(" 0.", "  .")
        comment = (
            ""
            if self.got == unknown
            else " ok"
            if self.ok
            else f" WRONG; expected answer is {self.solution}"
        )
        return (
            f"Puzzle {self.puzzle:4.1f}: {secs} seconds, answer {self.got:<17}{comment}"
        )


def summary():
    """Print a report that summarizes the answers."""
    for d in sorted(answers):
        print(answers[d])
    times = [answers[d].secs for d in answers]
    print(
        f"\nCorrect: {quantify((answers[d].ok for d in answers), bool)}/{len(answers)}"
    )
    print(
        f"\nTime in seconds: {median(times):.3f} median, {mean(times):.3f} mean, {sum(times):.3f} total."
    )


class multimap(defaultdict):
    """A mapping of {key: [val1, val2, ...]}."""

    def __init__(self, pairs: Iterable[Tuple] = (), symmetric=False):
        """Given (key, val) pairs, return {key: [val, ...], ...}.
        If `symmetric` is True, treat (key, val) as (key, val) plus (val, key)."""
        self.default_factory = list
        for key, val in pairs:
            self[key].append(val)
            if symmetric:
                self[val].append(key)


def prod(numbers) -> float:  # Will be math.prod in Python 3.8
    """The product formed by multiplying `numbers` together."""
    result = 1
    for x in numbers:
        result *= x
    return result


def T(matrix: Sequence[Sequence]) -> List[Tuple]:
    """The transpose of a matrix: T([(1,2,3), (4,5,6)]) == [(1,4), (2,5), (3,6)]"""
    return list(zip(*matrix))


def total(counter: Counter) -> int:
    """The sum of all the counts in a Counter."""
    return sum(counter.values())


def minmax(numbers) -> Tuple[int, int]:
    """A tuple of the (minimum, maximum) of numbers."""
    numbers = list(numbers)
    return min(numbers), max(numbers)


def cover(*integers) -> range:
    """A `range` that covers all the given integers, and any in between them.
    cover(lo, hi) is an inclusive (or closed) range, equal to range(lo, hi + 1).
    The same range results from cover(hi, lo) or cover([hi, lo])."""
    if len(integers) == 1 and not isinstance(integers[0], (int, bool)):
        nums = integers[0]
    else:
        nums = integers
    return range(min(nums), max(nums) + 1)


def the(sequence) -> object:
    """Return the one item in a sequence. Raise error if not exactly one."""
    it = iter(sequence)
    try:
        item = next(it)
        if next(it, None) is not None:
            raise ValueError("Expected exactly one item in sequence")
        return item
    except StopIteration:
        raise ValueError("Expected exactly one item in sequence")


def split_at(sequence, i) -> Tuple[Sequence, Sequence]:
    """The sequence split into two pieces: (before position i, and i-and-after)."""
    return sequence[:i], sequence[i:]


def ignore(*args) -> None:
    "Just return None."
    return None


def is_int(x) -> bool:
    "Is x an int?"
    return isinstance(x, int)


def sign(x) -> int:
    "0, +1, or -1"
    return 0 if x == 0 else +1 if x > 0 else -1


def lcm(i, j) -> int:
    "Least common multiple"
    return i * j // gcd(i, j)


def union(sets) -> set:
    "Union of several sets"
    return set().union(*sets)


def intersection(sets):
    "Intersection of several sets; error if no sets."
    first, *rest = sets
    return set(first).intersection(*rest)


def accumulate(item_count_pairs: Iterable[Tuple[object, int]]) -> Counter:
    """Add up all the (item, count) pairs into a Counter."""
    counter = Counter()
    for item, count in item_count_pairs:
        counter[item] += count
    return counter


def range_intersection(range1, range2) -> range:
    """Return a range that is the intersection of these two ranges."""
    return range(max(range1.start, range2.start), min(range1.stop, range2.stop))


def clock_mod(i, m) -> int:
    """i % m, but replace a result of 0 with m"""
    # This is like a clock, where 24 mod 12 is 12, not 0.
    return (i % m) or m


def invert_dict(dic) -> dict:
    """Invert a dict, e.g. {1: 'a', 2: 'b'} -> {'a': 1, 'b': 2}."""
    return {dic[x]: x for x in dic}


def walrus(name, value):
    """If you're not in 3.8 or more, and you can't do `x := val`,
    then you can use `walrus('x', val)`, if `x` is global."""
    globals()[name] = value
    return value


def truncate(object, width=100, ellipsis=" ...") -> str:
    """Use elipsis to truncate `str(object)` to `width` characters, if necessary."""
    string = str(object)
    return (
        string if len(string) <= width else string[: width - len(ellipsis)] + ellipsis
    )


def mapt(function: Callable, *sequences) -> tuple:
    """`map`, with the result as a tuple."""
    return tuple(map(function, *sequences))


def mapl(function: Callable, *sequences) -> list:
    """`map`, with the result as a list."""
    return list(map(function, *sequences))


def cat(things: Collection, sep="") -> str:
    """Concatenate the things."""
    return sep.join(map(str, things))


cache = functools.lru_cache(None)
Ø = frozenset()  # empty set


def quantify(iterable, pred) -> int:
    """Count the number of items in iterable for which pred is true."""
    return sum(1 for item in iterable if pred(item))


def dotproduct(vec1, vec2):
    """The dot product of two vectors."""
    return sum(map(operator.mul, vec1, vec2))


def powerset(iterable) -> Iterable[tuple]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return flatten(combinations(s, r) for r in range(len(s) + 1))


flatten = chain.from_iterable  # Yield items from each sequence in turn


def append(sequences) -> Sequence:
    "Append into a list"
    return list(flatten(sequences))


def batched(iterable, n) -> Iterable[tuple]:
    "Batch data into non-overlapping tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if batch:
            yield batch
        else:
            return


def sliding_window(sequence, n) -> Iterable[Sequence]:
    """All length-n subsequences of sequence."""
    return (sequence[i : i + n] for i in range(len(sequence) + 1 - n))


def first(iterable, default=None) -> Optional[object]:
    """The first element in an iterable, or the default if iterable is empty."""
    return next(iter(iterable), default)


def last(iterable) -> Optional[object]:
    """The last element in an iterable."""
    return next(iter(deque(iterable, maxlen=1)), None)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def first_true(iterable, default=False):
    """Returns the first true value in the iterable.
    If no true value is found, returns `default`."""
    return next((x for x in iterable if x), default)


Point = Tuple[Union[int, float], ...]  # Type for points
Vector = Point  # E.g., (1, 0) can be a point, or can be a direction, a Vector
Zero: Point = (0, 0)

directions4 = East, South, West, North = ((1, 0), (0, 1), (-1, 0), (0, -1))
diagonals = SE, NE, SW, NW = ((1, 1), (1, -1), (-1, 1), (-1, -1))
directions8 = directions4 + diagonals
directions5 = directions4 + (Zero,)
directions9 = directions8 + (Zero,)
arrow_direction = {
    "^": North,
    "v": South,
    ">": East,
    "<": West,
    ".": Zero,
    "U": North,
    "D": South,
    "R": East,
    "L": West,
}


def X_(point) -> int:
    "X coordinate of a point"
    return point[0]


def Y_(point) -> int:
    "Y coordinate of a point"
    return point[1]


def Z_(point) -> int:
    "Z coordinate of a point"
    return point[2]


def Xs(points) -> Tuple[int]:
    "X coordinates of a collection of points"
    return mapt(X_, points)


def Ys(points) -> Tuple[int]:
    "Y coordinates of a collection of points"
    return mapt(Y_, points)


def Zs(points) -> Tuple[int]:
    "X coordinates of a collection of points"
    return mapt(Z_, points)


def add(p: Point, q: Point) -> Point:
    "Add points"
    return mapt(operator.add, p, q)


def sub(p: Point, q: Point) -> Point:
    "Subtract points"
    return mapt(operator.sub, p, q)


def neg(p: Point) -> Vector:
    "Negate a point"
    return mapt(operator.neg, p)


def mul(p: Point, k: float) -> Vector:
    "Scalar multiply"
    return tuple(k * c for c in p)


def distance(p: Point, q: Point) -> float:
    """Euclidean (L2) distance between two points."""
    d = sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5
    return int(d) if d.is_integer() else d


def slide(points: Set[Point], delta: Vector) -> Set[Point]:
    """Slide all the points in the set of points by the amount delta."""
    return {add(p, delta) for p in points}


def make_turn(facing: Vector, turn: str) -> Vector:
    """Turn 90 degrees left or right. `turn` can be 'L' or 'Left' or 'R' or 'Right' or lowercase."""
    (x, y) = facing
    return (y, -x) if turn[0] in ("L", "l") else (-y, x)


# Profiling found that `add` and `taxi_distance` were speed bottlenecks;
# I define below versions that are specialized for 2D points only.


def add2(p: Point, q: Point) -> Point:
    """Specialized version of point addition for 2D Points only. Faster."""
    return (p[0] + q[0], p[1] + q[1])


def sub2(p: Point, q: Point) -> Point:
    """Specialized version of point subtraction for 2D Points only. Faster."""
    return (p[0] - q[0], p[1] - q[1])


def taxi_distance(p: Point, q: Point) -> Union[int, float]:
    """Manhattan (L1) distance between two 2D Points."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def distance_squared(p: Point, q: Point) -> float:
    """Square of the Euclidean (L2) distance between two 2D points."""
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


class Grid(dict):
    """A 2D grid, implemented as a mapping of {(x, y): cell_contents}."""

    def __init__(self, grid=(), directions=directions4, skip=(), default=None):
        """Initialize one of four ways:
        `Grid({(0, 0): '#', (1, 0): '.', ...})`
        `Grid(another_grid)
        `Grid(["#..", "..#"])
        `Grid("#..\n..#")`."""
        self.directions, self.skip, self.default = directions, skip, default
        if isinstance(grid, abc.Mapping):
            self.update(grid)
            self.size = (len(cover(Xs(self))), len(cover(Ys(self))))
        else:
            if isinstance(grid, str):
                grid = grid.splitlines()
            self.size = (max(map(len, grid)), len(grid))
            self.update(
                {
                    (x, y): val
                    for y, row in enumerate(grid)
                    for x, val in enumerate(row)
                    if val not in skip
                }
            )

    def __missing__(self, point):
        """If asked for a point off the grid, either return default or raise error."""
        if self.default is KeyError:
            raise KeyError(point)
        else:
            return self.default

    def in_range(self, point) -> bool:
        """Is the point within the range of the grid's size?"""
        return 0 <= X_(point) < X_(self.size) and 0 <= Y_(point) < Y_(self.size)

    def follow_line(self, start: Point, direction: Vector) -> Iterable[Point]:
        while self.in_range(start):
            yield start
            start = add2(start, direction)

    def copy(self):
        return Grid(
            self, directions=self.directions, skip=self.skip, default=self.default
        )

    def neighbors(self, point) -> List[Point]:
        """Points on the grid that neighbor `point`."""
        return [
            add2(point, Δ)
            for Δ in self.directions
            if (add2(point, Δ) in self) or (self.default not in (KeyError, None))
        ]

    def neighbor_contents(self, point) -> Iterable:
        """The contents of the neighboring points."""
        return (self[p] for p in self.neighbors(point))

    def findall(self, contents: Collection) -> List[Point]:
        """All points that contain one of the given contents, e.g. grid.findall('#')."""
        return [p for p in self if self[p] in contents]

    def to_rows(self, xrange=None, yrange=None) -> List[List[object]]:
        """The contents of the grid, as a rectangular list of lists.
        You can define a window with an xrange and yrange; or they default to the whole grid."""
        xrange = xrange or cover(Xs(self))
        yrange = yrange or cover(Ys(self))
        default = " " if self.default in (KeyError, None) else self.default
        return [[self.get((x, y), default) for x in xrange] for y in yrange]

    def print(self, sep="", xrange=None, yrange=None):
        """Print a representation of the grid."""
        for row in self.to_rows(xrange, yrange):
            print(*row, sep=sep)

    def __str__(self):
        return cat(self.to_rows())


def neighbors(point, directions=directions4) -> List[Point]:
    """Neighbors of this point, in the given directions.
    (This function can be used outside of a Grid class.)"""
    return [add(point, Δ) for Δ in directions]


def A_star_search(problem, h=None):
    """Search nodes with minimum f(n) = path_cost(n) + h(n) value first."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: n.path_cost + h(n))


def best_first_search(problem, f) -> "Node":
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return search_failure


class SearchProblem:
    """The abstract class for a search problem. A new domain subclasses this,
    overriding `actions` and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.initial, self.goal = initial, goal
        for key, value in kwds.items():
            setattr(self, key, value)

    def __str__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        return action  # Simplest case: action is result state

    def is_goal(self, state):
        return state == self.goal

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        return 0  # Never overestimate!


class GridProblem(SearchProblem):
    """Problem for searching a grid from a start to a goal location.
    A state is just an (x, y) location in the grid."""

    def actions(self, state):
        return [p for p in self.grid.neighbors(state) if self.grid[state] != "#"]

    def h(self, node):
        return taxi_distance(node.state, self.goal)


class Node:
    "A Node in a search tree."

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        return f"Node({self.state}, path_cost={self.path_cost})"

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


search_failure = Node(
    "failure", path_cost=inf
)  # Indicates an algorithm couldn't find a solution.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s2 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s2)
        yield Node(s2, node, action, cost)


def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (search_failure, None):
        return []
    return path_states(node.parent) + [node.state]


class PriorityQueue:
    """A queue in which the item with minimum key(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queue."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


class Hdict(dict):
    """A dict, but it is hashable."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class HCounter(Counter):
    """A Counter, but it is hashable."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class Graph(defaultdict):
    """A graph of {node: [neighboring_nodes...]}.
    Can store other kwd attributes on it (which you can't do with a dict)."""

    def __init__(self, contents, **kwds):
        self.update(contents)
        self.default_factory = list
        self.__dict__.update(**kwds)


class AttrCounter(Counter):
    """A Counter, but `ctr['name']` and `ctr.name` are the same."""

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def tests():
    """Run tests on utility functions. Also serves as usage examples."""

    # PARSER

    # assert parse("hello\nworld", show=0) == ("hello", "world")
    # assert parse("123\nabc7", digits, show=0) == ((1, 2, 3), (7,))
    assert truncate("hello world", 99) == "hello world"
    assert truncate("hello world", 8) == "hell ..."

    assert atoms("hello, cruel_world! 24-7") == ("hello", "cruel_world", 24, -7)
    assert words("hello, cruel_world! 24-7") == ("hello", "cruel", "world")
    assert digits("hello, cruel_world! 24-7") == (2, 4, 7)
    assert ints("hello, cruel_world! 24-7") == (24, -7)
    assert positive_ints("hello, cruel_world! 24-7") == (24, 7)

    # UTILITIES

    assert multimap(((i % 3), i) for i in range(9)) == {
        0: [0, 3, 6],
        1: [1, 4, 7],
        2: [2, 5, 8],
    }
    assert prod([2, 3, 5]) == 30
    assert total(Counter("hello, world")) == 12
    assert cover(3, 1, 4, 1, 5) == range(1, 6)
    assert minmax([3, 1, 4, 1, 5, 9]) == (1, 9)
    assert T([(1, 2, 3), (4, 5, 6)]) == [(1, 4), (2, 5), (3, 6)]
    assert the({1}) == 1
    assert split_at("hello, world", 6) == ("hello,", " world")
    assert is_int(-42) and not is_int("one")
    assert sign(-42) == -1 and sign(0) == 0 and sign(42) == +1
    assert union([{1, 2}, {3, 4}, {5, 6}]) == {1, 2, 3, 4, 5, 6}
    assert intersection([{1, 2, 3}, {2, 3, 4}, {2, 4, 6, 8}]) == {2}
    assert clock_mod(24, 12) == 12 and 24 % 12 == 0
    assert cat(["hello", "world"]) == "helloworld"

    # ITERTOOL RECIPES

    assert quantify(words("This is a test"), str.islower) == 3
    assert dotproduct([1, 2, 3, 4], [1000, 100, 10, 1]) == 1234
    assert list(flatten([{1, 2, 3}, (4, 5, 6), [7, 8, 9]])) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert append(([1, 2], [3, 4], [5, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(batched(range(11), 3)) == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10)]
    assert list(sliding_window("abcdefghi", 3)) == [
        "abc",
        "bcd",
        "cde",
        "def",
        "efg",
        "fgh",
        "ghi",
    ]
    assert first("abc") == "a"
    assert first("") is None
    assert last("abc") == "c"
    assert first_true([0, None, False, 42, 99]) == 42
    assert first_true([0, None, "", 0.0]) is False

    # POINTS

    p, q = (0, 3), (4, 0)
    assert Y_(p) == 3 and X_(q) == 4
    assert distance(p, q) == 5
    assert taxi_distance(p, q) == 7
    assert add(p, q) == (4, 3)
    assert sub(p, q) == (-4, 3)
    assert add(North, South) == (0, 0)


tests()
