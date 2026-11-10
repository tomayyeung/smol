"""Constant folding optimization."""

from dataclasses import dataclass

import smol.front.ast as ast
import smol.middle.tir as tir
from smol.int64 import to_i64

# An abstract value for this analysis.
type AbsVal = Top | Bot | Const


@dataclass(frozen=True)
class Top:
    def __repr__(self) -> str:
        return "⊤"  # noqa: RUF001


@dataclass(frozen=True)
class Bot:
    def __repr__(self) -> str:
        return "⊥"


@dataclass(frozen=True)
class Const:
    value: int

    def __repr__(self) -> str:
        return f"Const({self.value})"


def join_val(left: AbsVal, right: AbsVal) -> AbsVal:
    """Join two abstract values"""
    match (left, right):
        case (Bot(), x) | (x, Bot()):
            return x
        case (Top(), _) | (_, Top()):
            return Top()
        case (Const(a), Const(b)):
            return Const(a) if a == b else Top()


# The abstract state
type State = dict[str, AbsVal]


def join(left: State, right: State) -> State:
    """Join two abstract states"""
    result = left.copy()
    for x, v in right.items():
        result[x] = join_val(v, result.get(x, Bot()))
    return result


def execute(op: tir.BOp, left: AbsVal, right: AbsVal) -> AbsVal:
    # can add arithmetic identities too
    match (left, right):
        case (Bot(), _) | (_, Bot()):
            return Bot()
        case (Top(), _) | (_, Top()):
            return Top()
        case (Const(lhs), Const(rhs)):
            pass

    # Python trick: now, we have lhs and rhs available
    match op:
        case ast.BOp.ADD:
            result = lhs + rhs
        case ast.BOp.SUB:
            result = lhs - rhs
        case ast.BOp.MUL:
            result = lhs * rhs
        case ast.BOp.DIV:
            result = lhs // rhs if rhs != 0 else -1
        case ast.BOp.LT:
            result = int(lhs < rhs)
    return Const(to_i64(result))


def analyze_insn(s: State, i: tir.Instruction) -> State:
    # Copying is inefficient in Python, but efficient with sharing data structures.
    s = s.copy()
    match i:
        case tir.Copy(dst, src):
            s[dst] = s[src]
        case tir.Const(dst, n):
            s[dst] = Const(to_i64(n))
        case tir.Arith(op, dst, lhs, rhs):
            s[dst] = execute(op, s[lhs], s[rhs])
        case tir.Read(x):
            s[x] = Top()
        case tir.Print(_):
            pass
    return s


def analyze_term(s: State, t: tir.Terminator) -> list[str]:
    match t:
        case tir.Exit():
            return []
        case tir.Jump(lbl):
            return [lbl]
        case tir.Branch(guard, tt, ff):
            match s[guard]:
                case Const(0):
                    return [ff]
                case Const(_):
                    return [tt]
                case Top():
                    return [tt, ff]
                case Bot():
                    return []


def analyze_bb(s: State, b: tir.Block) -> tuple[State, list[str], list[State]]:
    """Return the block post state, successors, and post state of each instruction."""
    posts = []
    for i in b.instructions:
        s = analyze_insn(s, i)
        posts.append(s)
    return s, analyze_term(s, b.terminator), posts


def analyze(program: tir.Program) -> dict[str, State]:
    initial_state: State = {
        x: Bot() if x.startswith("_") else Const(0) for x in program.declarations
    }
    pre = {"entry": initial_state}
    worklist = ["entry"]
    while worklist:
        current = worklist[-1]
        worklist.pop()
        post, next, _ = analyze_bb(pre[current], program.blocks[current])
        for bb in next:
            if bb not in pre:
                new = post.copy()
                changed = True
            else:
                new = join(post, pre[bb])
                changed = new != pre[bb]
            if changed:
                pre[bb] = new
                worklist.append(bb)
    return pre
