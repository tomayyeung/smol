'''Constant folding optimization.'''

from dataclasses import dataclass

import smol.middle.tir as tir
import smol.front.ast as ast

# An abstract value for this analysis.
type AbsVal = Top | Bot | Const

@dataclass(frozen=True)
class Top:
    def __repr__(self) -> str: return "⊤"  # noqa: RUF001

@dataclass(frozen=True)
class Bot:
    def __repr__(self) -> str: return "⊥"

@dataclass(frozen=True)
class Const:
    value: int
    def __repr__(self) -> str: return f"Const({self.value})"

def join_val(left: AbsVal, right: AbsVal) -> AbsVal:
    '''Join two abstract values'''
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
    '''Join two abstract states'''
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
            return Const(lhs + rhs)
        case ast.BOp.SUB:
            return Const(lhs - rhs)
        case ast.BOp.MUL:
            return Const(lhs * rhs)
        case ast.BOp.DIV:
            return Const(lhs // rhs if rhs != 0 else 0)
        case ast.BOp.LT:
            return Const(int(lhs < rhs))

def analyze_insn(s: State, i: tir.Instruction) -> State:
    s = s.copy()  # this is inefficient in Python, but efficient with sharing data structures
    match i:
        case tir.Copy(dst, src):
            s[dst] = s[src]
        case tir.Const(dst, n):
            s[dst] = Const(n)
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

def analyze_bb(s: State, b: tir.Block) -> tuple[State, list[str]]:
    for i in b.instructions:
        s = analyze_insn(s, i)
    return s, analyze_term(s, b.terminator)

def analyze(program: tir.Program) -> dict[str, State]:
    initial_state: State = {x: Bot() for x in program.declarations}
    pre = {'entry': initial_state}
    worklist = ['entry']
    while worklist:
        current = worklist[-1]
        worklist.pop()
        post, next = analyze_bb(pre[current], program.blocks[current])
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

