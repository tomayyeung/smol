# Copyright (C) 2026 Mehmet Emre
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of smol.
#
# smol is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free
# Software Foundation.
#
# smol is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# smol. If not, see <https://www.gnu.org/licenses/>.

"""Command-line interface for the smol compiler."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from smol.back.codegen import code_gen
from smol.front.lex import Lexer
from smol.front.lower import lower
from smol.front.parse import parse
from smol.middle.opt import optimize
from smol.middle.const import analyze
from smol.middle.tir import Program


def _get_ir(source: str, should_optimize: bool) -> Program:
    program = lower(parse(source))
    return optimize(program) if should_optimize else program


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smol", description="Compile a smol program")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("file", type=Path, help="the input file")
    parser.add_argument(
        "-o",
        "--out",
        choices=("tokens", "ast", "tir", "asm", "const-prop"),
        default="tokens",
        help="the output format (default: tokens)",
    )
    parser.add_argument("-O", dest="optimize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    source = args.file.read_text(encoding="utf-8")

    if args.out == "tokens":
        lexer = Lexer(source)
        while (token := lexer.next()) is not None:
            print(token)
    elif args.out == "ast":
        print(parse(source))
    elif args.out == "tir":
        print(_get_ir(source, args.optimize), end="")
    elif args.out == "const-prop":
        for bb, state in sorted(analyze(_get_ir(source, args.optimize)).items()):
            print(f'{bb}:')
            for x, v in sorted(state.items()):
                print(f'\t{x} -> {v}')
    else:
        print(code_gen(_get_ir(source, args.optimize)).asm_code(), end="")


if __name__ == "__main__":
    main()
