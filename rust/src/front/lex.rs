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

//! The lexer.

use derive_more::Display;
use regex::Regex;
use TokenKind::*;

/// Tokens in the program
#[derive(Clone, Copy, PartialEq, Eq, Hash, Display, Debug)]
#[display("kind: '{kind}', part of input: '{text}'")]
pub struct Token<'src> {
    /// What token class this token belongs to.
    pub kind: TokenKind,
    /// What part of the input this token carries.
    pub text: &'src str,
}

/// Token classes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Display, Debug)]
pub enum TokenKind {
    #[display("id")]
    Id,
    #[display("num")]
    Num,
    #[display(":=")]
    Assign,
    #[display("$print")]
    Print,
    #[display("$read")]
    Read,
    #[display("$if")]
    If,
    #[display("{{")]
    LBrace,
    #[display("}}")]
    RBrace,
    #[display("+")]
    Plus,
    #[display("-")]
    Minus,
    #[display("*")]
    Mul,
    #[display("/")]
    Div,
    #[display("<")]
    Lt,
    #[display("~")]
    Tilde,
    #[display("error")]
    /// For unrecognized characters.
    Error,
}

pub struct Lexer<'input> {
    input: &'input str,
    pos: usize,
    whitespace: Regex,
    matchers: Vec<(Regex, TokenKind)>,
}

impl<'input> Lexer<'input> {
    pub fn new(input: &'input str) -> Self {
        let matchers = [
            (r"\$print", Print),
            (r"\$read", Read),
            (r"\$if", If),
            (r"\{", LBrace),
            (r"\}", RBrace),
            (r":=", Assign),
            (r"\+", Plus),
            (r"-", Minus),
            (r"\*", Mul),
            (r"/", Div),
            (r"<", Lt),
            (r"[a-zA-Z_][a-zA-Z0-9_]*", Id),
            (r"[0-9]+", Num),
            (r"~", Tilde),
        ]
        .into_iter()
        .map(|(regex, kind)| (Regex::new(&format!(r"\A{regex}")).unwrap(), kind))
        .collect::<Vec<_>>();
        // the following cases special regexes that are slightly different from the printed token
        Lexer {
            input,
            pos: 0,
            whitespace: Regex::new(r"\A(?:[ \t\f\r\n\v]|(?://.*))*").unwrap(),
            matchers,
        }
    }

    /// Has the lexer reached the end of input?
    pub fn end_of_input(&self) -> bool {
        self.pos == self.input.len()
    }

    // Skip comments and whitespace
    fn skip_whitespace(&mut self) {
        if self.end_of_input() {
            return;
        }

        let skip_match = self.whitespace.find(&self.input[self.pos..]);
        if let Some(skip_match) = skip_match {
            self.pos += skip_match.len();
        }
    }

    /// Get the next token if possible.
    ///
    /// The return value should be:
    /// - None if there are no more tokens (reached the end of input).
    /// - Some(token) where the token is the next token.
    /// - Some(Error) if none of the recognizers work, i.e. if there is a lexer error.
    pub fn next<'a>(&'a mut self) -> Option<Token<'input>> {
        // println!("pos: {}", self.pos);
        self.skip_whitespace();
        if self.end_of_input() {
            return None;
        }

        // Drop what doesn't match
        let mut matches: Vec<(regex::Match<'_>, &TokenKind)> = self
            .matchers
            .iter()
            .filter_map(|(re, token)| {
                if let Some(m) = re.find(&self.input[self.pos..]) {
                    Some((m, token))
                } else {
                    None
                }
            })
            .collect();

        // Keep longest match
        matches.sort_by_key(|(m, _)| m.len());
        let longest = matches.last();
        Some(match longest {
            Some((m, token)) => {
                self.pos += m.len();

                Token {
                    kind: **token,
                    text: m.as_str(),
                }
            }
            None => {
                self.pos += 1;
                Token {
                    kind: TokenKind::Error,
                    text: &self.input[self.pos - 1..self.pos],
                }
            }
        })
    }
}

/// Read all the tokens from input
pub fn get_tokens(input: &'_ str) -> Vec<Token<'_>> {
    let mut lexer = Lexer::new(input);

    let mut tokens = vec![];
    while let Some(token) = lexer.next() {
        tokens.push(token);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    // SECTION: helpers

    // Create an id token
    fn id(text: &'_ str) -> Token<'_> {
        Token { kind: Id, text }
    }

    // Create a num token
    fn num(text: &'_ str) -> Token<'_> {
        Token { kind: Num, text }
    }

    // Create an error token
    fn error(text: &'_ str) -> Token<'_> {
        Token { kind: Error, text }
    }

    // Create a token with only one lexeme (anything except id, num, error).
    fn t(kind: TokenKind) -> Token<'static> {
        Token {
            kind,
            text: match kind {
                Id | Num | Error => unreachable!(),
                Assign => ":=",
                Print => "$print",
                Read => "$read",
                If => "$if",
                LBrace => "{",
                RBrace => "}",
                Plus => "+",
                Minus => "-",
                Mul => "*",
                Div => "/",
                Lt => "<",
                Tilde => "~",
            },
        }
    }

    // SECTION: tests

    #[test]
    fn skip_whitespace() {
        let mut lexer = Lexer::new("foo");
        lexer.skip_whitespace();
        assert_eq!(lexer.pos, 0);
        let mut lexer = Lexer::new(" \n\t  foo");
        lexer.skip_whitespace();
        assert_eq!(lexer.pos, 5);
        let mut lexer = Lexer::new(" // stuff\n\t  foo ");
        lexer.skip_whitespace();
        assert_eq!(lexer.pos, 13);
    }

    #[test]
    fn empty() {
        assert_eq!(get_tokens(""), vec![]);
        assert_eq!(get_tokens("  \n//hello\n"), vec![]);
        assert_eq!(get_tokens("  \n//hi"), vec![]);
    }

    #[test]
    fn single_token() {
        let tests = [
            ("x", vec![id("x")]),
            ("print", vec![id("print")]),
            ("if", vec![id("if")]),
            ("yolo", vec![id("yolo")]),
            ("3", vec![num("3")]),
            ("0345678910", vec![num("0345678910")]),
            ("%", vec![error("%")]),
            (":=", vec![t(Assign)]),
            ("$print", vec![t(Print)]),
            ("$read", vec![t(Read)]),
            ("$if", vec![t(If)]),
            ("{", vec![t(LBrace)]),
            ("}", vec![t(RBrace)]),
            ("+", vec![t(Plus)]),
            ("-", vec![t(Minus)]),
            ("*", vec![t(Mul)]),
            ("/", vec![t(Div)]),
            ("<", vec![t(Lt)]),
        ];

        for (input, expected) in tests {
            assert_eq!(
                get_tokens(input),
                expected,
                "the lexer produced the wrong results for the input {input:?}"
            )
        }
    }

    #[test]
    fn multi_token() {
        assert_eq!(
            get_tokens("x$print$read$if{}+0-*$/<"),
            vec![
                id("x"),
                t(Print),
                t(Read),
                t(If),
                t(LBrace),
                t(RBrace),
                t(Plus),
                num("0"),
                t(Minus),
                t(Mul),
                error("$"),
                t(Div),
                t(Lt),
            ]
        );
        assert_eq!(
            get_tokens("x yz $print $read $if { } +  0   -  //hi\n * $ read / < ~"),
            vec![
                id("x"),
                id("yz"),
                t(Print),
                t(Read),
                t(If),
                t(LBrace),
                t(RBrace),
                t(Plus),
                num("0"),
                t(Minus),
                t(Mul),
                error("$"),
                id("read"),
                t(Div),
                t(Lt),
                t(Tilde),
            ]
        );
    }
}
