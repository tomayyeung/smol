// Copyright (C) 2026 Mehmet Emre
// SPDX-License-Identifier: GPL-3.0-only
//
// This file is part of smol.
//
// smol is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License version 3 as published by the Free
// Software Foundation.
//
// smol is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// smol. If not, see <https://www.gnu.org/licenses/>.

//! The parser

use std::fmt::Debug;

use derive_more::derive::Display;

use super::ast::*;
use super::lex::*;
use crate::common::id;

#[derive(Display)]
#[display("Parse error: {}", self.0)]
pub struct ParseError(String);

impl Debug for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

type ParseResult<T> = Result<T, ParseError>;

/// helper function to create a ParseError
fn parse_err<T>(err: String) -> ParseResult<T> {
    ParseResult::Err(ParseError(err))
}

pub fn parse(input: &str) -> Result<Program, ParseError> {
    let mut parser = Parser::new(input);
    let program = parser.parse_program()?;
    if !parser.tokens.is_empty() {
        Err(ParseError(
            "There are still leftover tokens after reading a whole program.".to_string(),
        ))
    } else {
        Ok(program)
    }
}

struct Parser<'input> {
    /// Rest of the input, ordered in reverse.
    tokens: Vec<Token<'input>>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut tokens = get_tokens(input);
        tokens.reverse();
        Parser { tokens }
    }

    /// Return the next token
    fn peek(&self) -> Option<Token<'_>> {
        self.tokens.last().copied()
    }

    /// Pop and return the next token
    fn next(&mut self) -> ParseResult<Token<'_>> {
        if let Some(t) = self.tokens.pop() {
            ParseResult::Ok(t)
        } else {
            parse_err("reached EOF too early".to_string())
        }
    }

    /// Does the next token match the given kind?
    fn next_is(&self, kind: TokenKind) -> bool {
        self.peek().is_some_and(|x| x.kind == kind)
    }

    /// Consume a token of given type if possible, return if it is consumed
    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.next_is(kind) {
            self.tokens.pop();
            true
        } else {
            false
        }
    }

    /// Consume and return a token of the given type; fail on mismatch
    fn expect(&mut self, kind: TokenKind) -> ParseResult<Token<'_>> {
        let t = self.next()?;

        if t.kind != kind {
            parse_err(format!(
                "next token does not match given type: {}, was: {}",
                kind, t.kind
            ))
        } else {
            ParseResult::Ok(t)
        }
    }

    fn parse_program(&mut self) -> ParseResult<Program> {
        let mut stmts: Vec<Stmt> = Vec::new();
        while !self.tokens.is_empty() {
            stmts.push(self.parse_stmt()?);
        }

        ParseResult::Ok(Program { stmts })
    }

    fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        use Stmt::*;

        let t = self.next()?;
        ParseResult::Ok(match t.kind {
            TokenKind::Assign => {
                let lhs = id(self.expect(TokenKind::Id)?.text);
                let rhs = self.parse_expr()?;
                Assign(lhs, rhs)
            }
            TokenKind::Print => Print(self.parse_expr()?),
            TokenKind::Read => {
                let var = id(self.expect(TokenKind::Id)?.text);
                Read(var)
            }
            TokenKind::If => {
                let guard = self.parse_expr()?;
                let tt = self.parse_block()?;
                let ff = self.parse_block()?;
                If { guard, tt, ff }
            }
            _ => return parse_err(format!("unexpected token {} when parsing stmt", t)),
        })
    }

    fn parse_block(&mut self) -> ParseResult<Vec<Stmt>> {
        self.expect(TokenKind::LBrace)?;

        let mut stmts = Vec::new();
        while !self.eat(TokenKind::RBrace) {
            stmts.push(self.parse_stmt()?)
        }

        ParseResult::Ok(stmts)
    }

    fn parse_expr(&mut self) -> ParseResult<Expr> {
        use Expr::*;

        let t = self.next()?;
        ParseResult::Ok(match t.kind {
            TokenKind::Id => Var(id(t.text)),
            TokenKind::Num => Const(
                t.text
                    .parse()
                    .map_err(|err: std::num::ParseIntError| ParseError(err.to_string()))?,
            ),
            TokenKind::Tilde => Negate(Box::new(self.parse_expr()?)),
            _ => {
                let op = match t.text {
                    "*" => BOp::Mul,
                    "/" => BOp::Div,
                    "+" => BOp::Add,
                    "-" => BOp::Sub,
                    "<" => BOp::Lt,
                    s => {
                        return parse_err(format!("unrecognized token {} when parsing binop", s));
                    }
                };
                self.parse_binop(op)?
            }
        })
    }

    /// helper: read and parse both sides of given binary operation
    fn parse_binop(&mut self, op: BOp) -> ParseResult<Expr> {
        use Expr::*;

        let lhs = Box::new(self.parse_expr()?);
        let rhs = Box::new(self.parse_expr()?);

        ParseResult::Ok(BinOp { op, lhs, rhs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BOp::*;
    use Expr::*;
    use Stmt::*;

    // SECTION: helpers

    // Move a value to the heap
    fn b<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    // Build a binary operation expression
    fn bop(op: BOp, lhs: Expr, rhs: Expr) -> Expr {
        BinOp {
            op,
            lhs: b(lhs),
            rhs: b(rhs),
        }
    }

    // Build a negation expression
    fn negate(inner: Expr) -> Expr {
        Negate(b(inner))
    }

    // Build a variable node
    fn var(name: &str) -> Expr {
        Var(id(name))
    }

    // SECTION: tests

    #[test]
    fn empty() {
        assert_eq!(parse("").unwrap().stmts, vec![]);
    }

    #[test]
    fn print() {
        assert_eq!(parse("$print 0").unwrap().stmts, vec![Print(Const(0))]);
    }

    #[test]
    fn read() {
        assert_eq!(parse("$read x").unwrap().stmts, vec![Read(id("x"))]);
    }

    #[test]
    fn var_test() {
        assert_eq!(parse("$print x").unwrap().stmts, vec![Print(var("x"))]);
    }

    #[test]
    fn binop() {
        assert_eq!(
            parse("$print + x x").unwrap().stmts,
            vec![Print(bop(Add, var("x"), var("x")))]
        );
        assert_eq!(
            parse("$print * x x").unwrap().stmts,
            vec![Print(bop(Mul, var("x"), var("x")))]
        );
        assert_eq!(
            parse("$print / x x").unwrap().stmts,
            vec![Print(bop(Div, var("x"), var("x")))]
        );
        assert_eq!(
            parse("$print - x x").unwrap().stmts,
            vec![Print(bop(Sub, var("x"), var("x")))]
        );
        assert_eq!(
            parse("$print < x x").unwrap().stmts,
            vec![Print(bop(Lt, var("x"), var("x")))]
        );
    }

    #[test]
    fn negate_test() {
        assert_eq!(
            parse("$print ~ x").unwrap().stmts,
            vec![Print(negate(var("x")))]
        );
    }

    #[test]
    fn complex_expr() {
        assert_eq!(
            parse("$print * + x 3 / ~ 7 y").unwrap().stmts,
            vec![Print(bop(
                Mul,
                bop(Add, var("x"), Const(3)),
                bop(Div, negate(Const(7)), var("y"))
            ))]
        );
    }

    #[test]
    fn assign() {
        assert_eq!(
            parse(":= x 3").unwrap().stmts,
            vec![Assign(id("x"), Const(3))]
        );
        assert_eq!(
            parse(":= x + x 3").unwrap().stmts,
            vec![Assign(id("x"), bop(Add, var("x"), Const(3)))]
        );
    }

    #[test]
    fn if_test() {
        assert_eq!(
            parse("$if x {} {}").unwrap().stmts,
            vec![If {
                guard: var("x"),
                tt: vec![],
                ff: vec![]
            }]
        );
        assert_eq!(
            parse("$if x {$print 0} {:= x 3}").unwrap().stmts,
            vec![If {
                guard: var("x"),
                tt: vec![Print(Const(0))],
                ff: vec![Assign(id("x"), Const(3))]
            }]
        );
        assert_eq!(
            parse("$if x {$print 0 $read x} {:= x 3 := y x}")
                .unwrap()
                .stmts,
            vec![If {
                guard: var("x"),
                tt: vec![Print(Const(0)), Read(id("x"))],
                ff: vec![Assign(id("x"), Const(3)), Assign(id("y"), var("x"))]
            }]
        );
        assert_eq!(
            parse("$if < x y {$print 0} {:= x 3}").unwrap().stmts,
            vec![If {
                guard: bop(Lt, var("x"), var("y")),
                tt: vec![Print(Const(0))],
                ff: vec![Assign(id("x"), Const(3))]
            }]
        );
    }

    #[test]
    fn death_test1() {
        // illegal tokens to start a program
        assert!(parse("x").is_err());
        assert!(parse("0").is_err());
        assert!(parse("<").is_err());

        // extra lexemes after a statement
        assert!(parse(":= x y + z").is_err());
        assert!(parse(":= x y + z t").is_err());
    }

    #[test]
    fn death_test_print() {
        assert!(parse("$print").is_err());
    }

    #[test]
    fn death_test_read() {
        assert!(parse("$read").is_err());
    }

    #[test]
    fn death_test_assign() {
        assert!(parse(":=").is_err());
        assert!(parse(":= x").is_err());
        assert!(parse(":= 3 x").is_err());
    }

    #[test]
    fn death_test_if() {
        assert!(parse("$if").is_err());
        assert!(parse("$if x {}").is_err());
        assert!(parse("$if {} {}").is_err());
        assert!(parse("$if x y {}").is_err());
        assert!(parse("$if x $print x {}").is_err());
    }

    #[test]
    fn death_test_expr() {
        assert!(parse("$print 3 + x").is_err());
        assert!(parse("$print + x").is_err());
        assert!(parse("$print - x").is_err());
        assert!(parse("$print * x").is_err());
        assert!(parse("$print / x").is_err());
        assert!(parse("$print < x").is_err());
        assert!(parse("$print ~").is_err());
        assert!(parse("$print ~ x y").is_err());
        assert!(parse("$print + + x y").is_err());
        assert!(parse("$print < y").is_err());
        assert!(parse("$print < - y z").is_err());
    }
}
