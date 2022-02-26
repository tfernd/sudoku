#%%
from __future__ import annotations

from time import perf_counter
import numpy as np
import pycosat


# http://anytime.cs.umass.edu/aimath06/proceedings/P34.pdf
# https://easychair.org/publications/open/VF3m
# https://t-dillon.github.io/tdoku/
# https://www.f-puzzles.com/


def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start} seconds")
        return result

    return wrapper


class Sudoku:
    """Sudoku SAT solver.
    
    No nasty nested for loops allowed!
    """

    clauses: list[list[int]]

    val: np.ndarray
    bval: np.ndarray

    def __init__(self, *, n: int = 9):
        ns = int(np.sqrt(n))
        assert ns ** 2 == n, f"`n` ({n}) must be a perfect square"

        # variables
        self.n, self.ns = n, ns
        self.clauses = []

        # (row, col, num)
        self.val = 1 + np.arange(n ** 3).reshape(n, n, n)

        # (brow, bcol, srow, scol, num)
        bval = self.val.reshape(ns, ns, ns, ns, n)
        self.bval = bval.transpose(0, 2, 1, 3, 4)

        # pairs (i, j) where i > j
        i = np.arange(n)
        ij = np.stack(np.meshgrid(i, i), axis=0)
        self.pairs = ij[:, ij[1] < ij[0]].T

        # Standard rules
        self.unique_cell()
        self.unique_col()
        self.unique_row()
        self.unique_box()

        self.some_cell()
        self.some_col()
        self.some_row()
        self.some_box()

        ## Variant rules

        # chess
        self.unique_king()
        self.unique_knight()
        # self.unique_queen()

        # sandwich
        # thermo
        # killer
        # litte-killer
        # arrow
        # BetweenLineConstraint
        # CloneConstraint
        # diagonal
        # non-consecutive
        # kropki (diference, ratio)
        # even/odd
        # Orthogonal
        # PalindromeConstraint
        # disjoint group
        # XV
        # RenbanConstraint
        # SelfTaxicabConstraint
        # TaxicabConstraint
        # WhispersConstraint

    def _extend(self, clauses: np.ndarray, /):
        "Add new clauses to the solver."

        self.clauses.extend(clauses.tolist())

    def some_cell(self):
        "At least one value in each cell."

        # (row*col, num)
        clauses = self.val.transpose(0, 1, 2).reshape(-1, self.n)
        self._extend(clauses)

    def some_col(self):
        "At least one value in each column."

        # (col*num, row)
        clauses = self.val.transpose(1, 2, 0).reshape(-1, self.n)
        self._extend(clauses)

    def some_row(self):
        "At least one value in each row."

        # (row*num, col)
        clauses = self.val.transpose(0, 2, 1).reshape(-1, self.n)
        self._extend(clauses)

    def some_box(self):
        "At least one value in each box."

        # (brow*bcol*num, srow*scol)
        clauses = self.bval.transpose(0, 1, 4, 2, 3).reshape(-1, self.n)
        self._extend(clauses)

    def unique_cell(self):
        "At most one value in each cell."

        clauses = self.val[:, :, self.pairs].transpose(0, 1, 2, 3)
        self._extend(-clauses.reshape(-1, 2))

    def unique_col(self):
        "At most one value in each col."

        clauses = self.val[:, self.pairs, :].transpose(0, 1, 3, 2)
        self._extend(-clauses.reshape(-1, 2))

    def unique_row(self):
        "At most one value in each row."

        clauses = self.val[self.pairs, :, :].transpose(0, 2, 3, 1)
        self._extend(-clauses.reshape(-1, 2))

    def unique_box(self):
        "At most one value in each box."

        # (brow*bcol*num, srow*scol)
        clauses = self.bval.transpose(0, 1, 4, 2, 3).reshape(-1, self.n)
        self._extend(-clauses[:, self.pairs].reshape(-1, 2))

    def _unique_by_offset(self, offsets: list[tuple[int, int]], /):
        "At most one value in the neighboorhood given by an `offset`."

        # ! a bit of madness...
        size = len(offsets)
        offsets = np.array(offsets).T
        offsets = offsets.reshape(2, 1, -1)

        src = np.stack(np.nonzero(self.val[..., 0]))
        src = src.reshape(2, -1, 1)
        src = np.tile(src, [1, 1, size])

        dst = src + offsets

        src = src.reshape(2, -1)
        dst = dst.reshape(2, -1)

        pairs = np.concatenate([src, dst], axis=0).T
        mask = (pairs >= 0) & (pairs < self.n)
        mask = np.all(mask, axis=1)
        pairs = pairs[mask]
        pairs = pairs.reshape(-1, 2, 2)

        norm2 = (pairs ** 2).sum(2).T
        mask = norm2[0] < norm2[1]
        pairs = pairs[mask]

        sr, sc = pairs[:, 0].T
        dr, dc = pairs[:, 1].T

        src = self.val[sr, sc]
        dst = self.val[dr, dc]

        clauses = np.stack([src, dst]).reshape(2, -1).T
        self._extend(-clauses)

    def parse(self, board: str, /) -> np.ndarray:
        "Parse a board string into a numpy array."

        # trying to avoid list comprehension
        data = board.strip().replace(".", "0")
        data = list(filter(lambda s: s.isalnum(), data))
        data = list(map(int, data))
        assert len(data) == self.n ** 2, f"board must be {self.n}x{self.n}"
        data = np.array(data).reshape(self.n, self.n)

        return data

    def solve(self, board: str, /):
        "Solve a sudoku."

        data = self.parse(board)

        # add initial values
        row, col = data.nonzero()
        num = data[row, col]

        clauses = self.val[row, col, num - 1].reshape(-1, 1).tolist()
        clauses.extend(self.clauses)

        # solve
        start = perf_counter()
        sol = pycosat.solve(clauses)
        assert sol != "UNSAT", "unsatisfiable"
        dt = perf_counter() - start
        print(f"Solved in {dt*1e3:.3f}ms")

        # get filled values
        sol = np.array(sol)
        sol = sol[sol > 0]
        assert sol.size == self.n ** 2, "invalid solution"

        # decompose: sol = row*n**2 + n*col + num + 1
        n = self.n
        row = (sol - 1) // n ** 2
        col = (sol - row * n ** 2 - 1) // n
        num = sol - row * n ** 2 - col * n - 1

        # fill in solved values
        data[row, col] = num + 1

        return data

    def unique_king(self):
        "At most one value in each king's neighborhood."

        offsets = [
            (+0, +1),
            (+0, -1),
            (+1, +0),
            (-1, +0),
            (+1, +1),
            (+1, -1),
            (-1, +1),
            (-1, -1),
        ]
        self._unique_by_offset(offsets)

    def unique_knight(self):
        "At most one value in each knight's neighborhood."

        offsets = [
            (+2, +1),
            (+2, -1),
            (+1, +2),
            (-1, +2),
            (-2, +1),
            (-2, -1),
            (+1, -2),
            (-1, -2),
        ]
        self._unique_by_offset(offsets)


if __name__ == "__main__":
    s = Sudoku()

    data = s.solve("." * 81)
    print(data)
