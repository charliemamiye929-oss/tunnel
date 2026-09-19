"""A tiny, safe expression language for strategy rules.

Rules are strings such as::

    rsi14 < 30 and close > sma200 and mkt_above_sma200 == 1
    cross_above(macd, macd_signal) and volume_ratio > 1.5

They are parsed into an AST once and evaluated per bar against a dict of
features.  Nothing is ``eval``-ed: the grammar below is the whole language, so a
genome written by an LLM cannot reach anything outside the feature vocabulary.

Grammar (lowest to highest precedence)::

    expr    := or_expr
    or_expr := and_expr ('or' and_expr)*
    and_expr:= not_expr ('and' not_expr)*
    not_expr:= 'not' not_expr | comparison
    comparison := sum (('<'|'<='|'>'|'>='|'=='|'!=') sum)*
    sum     := product (('+'|'-') product)*
    product := unary (('*'|'/') unary)*
    unary   := '-' unary | atom
    atom    := number | 'true' | 'false' | call | name | '(' expr ')'
    call    := funcname '(' expr (',' expr)* ')'
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Set, Tuple


class DslError(ValueError):
    """Raised for a malformed rule or an unknown feature/function."""


# ---------------------------------------------------------------- tokenizer

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+\.\d*|\.\d+|\d+)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op><=|>=|==|!=|<|>|\+|-|\*|/|\(|\)|,)
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "true", "false"}


@dataclass(frozen=True)
class Token:
    kind: str   # 'number' | 'name' | 'op' | 'kw' | 'end'
    text: str
    pos: int


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise DslError(f"unexpected character {src[i]!r} at {i} in {src!r}")
        i = m.end()
        if m.lastgroup == "ws":
            continue
        text = m.group()
        if m.lastgroup == "name" and text.lower() in _KEYWORDS:
            tokens.append(Token("kw", text.lower(), m.start()))
        else:
            tokens.append(Token(m.lastgroup, text, m.start()))
    tokens.append(Token("end", "", len(src)))
    return tokens


# ------------------------------------------------------------------- nodes

class Node:
    """Base AST node.  ``eval`` takes the current and previous feature dicts."""

    def eval(self, cur: Mapping[str, float], prev: Mapping[str, float]) -> float:
        raise NotImplementedError

    def names(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class Const(Node):
    value: float

    def eval(self, cur, prev) -> float:
        return self.value


@dataclass(frozen=True)
class Name(Node):
    name: str

    def eval(self, cur, prev) -> float:
        try:
            return float(cur[self.name])
        except KeyError as exc:  # pragma: no cover - validation catches this first
            raise DslError(f"unknown feature {self.name!r}") from exc

    def names(self) -> Set[str]:
        return {self.name}


@dataclass(frozen=True)
class Unary(Node):
    op: str
    operand: Node

    def eval(self, cur, prev) -> float:
        v = self.operand.eval(cur, prev)
        if self.op == "-":
            return -v
        return 0.0 if v else 1.0  # 'not'

    def names(self) -> Set[str]:
        return self.operand.names()


@dataclass(frozen=True)
class Binary(Node):
    op: str
    left: Node
    right: Node

    def eval(self, cur, prev) -> float:
        op = self.op
        if op == "and":
            return 1.0 if (self.left.eval(cur, prev) and self.right.eval(cur, prev)) else 0.0
        if op == "or":
            return 1.0 if (self.left.eval(cur, prev) or self.right.eval(cur, prev)) else 0.0
        a = self.left.eval(cur, prev)
        b = self.right.eval(cur, prev)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0 else 0.0
        if op == "<":
            return float(a < b)
        if op == "<=":
            return float(a <= b)
        if op == ">":
            return float(a > b)
        if op == ">=":
            return float(a >= b)
        if op == "==":
            return float(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
        if op == "!=":
            return float(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
        raise DslError(f"unknown operator {op!r}")  # pragma: no cover

    def names(self) -> Set[str]:
        return self.left.names() | self.right.names()


@dataclass(frozen=True)
class Call(Node):
    func: str
    args: Tuple[Node, ...]

    def eval(self, cur, prev) -> float:
        f = self.func
        if f == "cross_above":
            a, b = self.args
            return float(a.eval(cur, prev) > b.eval(cur, prev)
                         and a.eval(prev, prev) <= b.eval(prev, prev))
        if f == "cross_below":
            a, b = self.args
            return float(a.eval(cur, prev) < b.eval(cur, prev)
                         and a.eval(prev, prev) >= b.eval(prev, prev))
        if f == "prev":
            return self.args[0].eval(prev, prev)
        if f == "change":
            return self.args[0].eval(cur, prev) - self.args[0].eval(prev, prev)
        vals = [a.eval(cur, prev) for a in self.args]
        if f == "abs":
            return abs(vals[0])
        if f == "min":
            return min(vals)
        if f == "max":
            return max(vals)
        if f == "clamp":
            return max(vals[1], min(vals[2], vals[0]))
        raise DslError(f"unknown function {f!r}")  # pragma: no cover

    def names(self) -> Set[str]:
        out: Set[str] = set()
        for a in self.args:
            out |= a.names()
        return out


#: name -> (min_args, max_args)
FUNCTIONS: Dict[str, Tuple[int, int]] = {
    "cross_above": (2, 2), "cross_below": (2, 2), "prev": (1, 1),
    "change": (1, 1), "abs": (1, 1), "min": (2, 4), "max": (2, 4),
    "clamp": (3, 3),
}

#: Handed to Claude verbatim so generated rules stay inside the language.
FUNCTION_DOCS = {
    "cross_above(a, b)": "true on the bar where a crosses from at-or-below b to above b",
    "cross_below(a, b)": "true on the bar where a crosses from at-or-above b to below b",
    "prev(x)": "value of x on the previous bar",
    "change(x)": "x minus its previous value",
    "abs(x)": "absolute value",
    "min(a, b, ...)": "smallest argument",
    "max(a, b, ...)": "largest argument",
    "clamp(x, lo, hi)": "x limited to the range [lo, hi]",
}


# ------------------------------------------------------------------ parser

class _Parser:
    def __init__(self, tokens: Sequence[Token], src: str):
        self.tokens = tokens
        self.src = src
        self.i = 0

    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Token:
        if self.tok.text != text:
            raise DslError(f"expected {text!r} but found {self.tok.text or 'end'!r} "
                           f"in {self.src!r}")
        return self.advance()

    def parse(self) -> Node:
        node = self.parse_or()
        if self.tok.kind != "end":
            raise DslError(f"trailing input {self.tok.text!r} in {self.src!r}")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.tok.kind == "kw" and self.tok.text == "or":
            self.advance()
            node = Binary("or", node, self.parse_and())
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.tok.kind == "kw" and self.tok.text == "and":
            self.advance()
            node = Binary("and", node, self.parse_not())
        return node

    def parse_not(self) -> Node:
        if self.tok.kind == "kw" and self.tok.text == "not":
            self.advance()
            return Unary("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Node:
        node = self.parse_sum()
        while self.tok.kind == "op" and self.tok.text in ("<", "<=", ">", ">=", "==", "!="):
            op = self.advance().text
            node = Binary(op, node, self.parse_sum())
        return node

    def parse_sum(self) -> Node:
        node = self.parse_product()
        while self.tok.kind == "op" and self.tok.text in ("+", "-"):
            op = self.advance().text
            node = Binary(op, node, self.parse_product())
        return node

    def parse_product(self) -> Node:
        node = self.parse_unary()
        while self.tok.kind == "op" and self.tok.text in ("*", "/"):
            op = self.advance().text
            node = Binary(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Node:
        if self.tok.kind == "op" and self.tok.text == "-":
            self.advance()
            return Unary("-", self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        t = self.tok
        if t.kind == "number":
            self.advance()
            return Const(float(t.text))
        if t.kind == "kw" and t.text in ("true", "false"):
            self.advance()
            return Const(1.0 if t.text == "true" else 0.0)
        if t.kind == "op" and t.text == "(":
            self.advance()
            node = self.parse_or()
            self.expect(")")
            return node
        if t.kind == "name":
            self.advance()
            if self.tok.kind == "op" and self.tok.text == "(":
                return self.parse_call(t.text)
            return Name(t.text)
        raise DslError(f"unexpected {t.text or 'end of rule'!r} in {self.src!r}")

    def parse_call(self, func: str) -> Node:
        if func not in FUNCTIONS:
            raise DslError(f"unknown function {func!r}; available: {sorted(FUNCTIONS)}")
        self.expect("(")
        args: List[Node] = [self.parse_or()]
        while self.tok.kind == "op" and self.tok.text == ",":
            self.advance()
            args.append(self.parse_or())
        self.expect(")")
        lo, hi = FUNCTIONS[func]
        if not lo <= len(args) <= hi:
            raise DslError(f"{func} takes {lo}..{hi} arguments, got {len(args)}")
        return Call(func, tuple(args))


def parse(src: str) -> Node:
    """Parse a rule string into an AST."""
    if not isinstance(src, str) or not src.strip():
        raise DslError("empty rule")
    if len(src) > 600:
        raise DslError(f"rule too long ({len(src)} chars, max 600)")
    return _Parser(tokenize(src), src).parse()


@dataclass(frozen=True)
class Rule:
    """A compiled, validated rule expression."""

    source: str
    ast: Node
    features: frozenset

    def __call__(self, cur: Mapping[str, float], prev: Mapping[str, float]) -> bool:
        return bool(self.ast.eval(cur, prev))


def compile_rule(src: str, allowed: Set[str] | frozenset) -> Rule:
    """Parse ``src`` and check every referenced name is a known feature."""
    ast = parse(src)
    used = ast.names()
    unknown = sorted(n for n in used if n not in allowed)
    if unknown:
        raise DslError(f"unknown feature(s) {unknown} in rule {src!r}")
    return Rule(src, ast, frozenset(used))
