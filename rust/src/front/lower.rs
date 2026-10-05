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

//! Lowering

mod hidden;
use super::ast;
use super::BOp;
use crate::{
    common::{id, Id},
    middle::tir,
};
use ast::Expr;
use ast::Stmt;
use hidden::*;
use std::collections::{BTreeMap as Map, BTreeSet as Set};
use tir::{Block, Instruction, Terminator};
use TvEntry::*;

pub fn lower(program: ast::Program) -> tir::Program {
    let lower = Lower::new();
    lower.lower_program(program)
}

// Entries in the translation vector
#[derive(Debug, Clone)]
enum TvEntry {
    // A basic block label
    Label(Id),
    // An inner (non-terminating) instruction
    Inner(Instruction),
    // A terminal instruction
    Term(Terminator),
}

impl TvEntry {
    fn get_inner(self) -> Option<Instruction> {
        if let Inner(i) = self {
            Some(i)
        } else {
            None
        }
    }
}

// Lowering data
struct Lower {
    decl: Set<Id>,
    // translation vector
    tv: Vec<TvEntry>,
    // for creating fresh locals
    fresh_ctr: i64,
    // for creating fresh block labels
    bb_ctr: i64,
}

impl Lower {
    fn new() -> Self {
        Lower {
            decl: Set::new(),
            tv: vec![],
            fresh_ctr: 0,
            bb_ctr: 0,
        }
    }

    // add given variable to declared variables
    fn add_decl(&mut self, var: Id) {
        self.decl.insert(var);
    }

    /// add given instruction to tv
    fn emit(&mut self, insn: Instruction) {
        self.tv.push(TvEntry::Inner(insn));
    }

    /// add given id to tv
    fn emit_label(&mut self, lbl: Id) {
        self.tv.push(TvEntry::Label(lbl));
    }

    /// add given terminator to tv
    fn emit_term(&mut self, term: Terminator) {
        self.tv.push(TvEntry::Term(term));
    }

    fn lower_program(mut self, program: ast::Program) -> tir::Program {
        self.tv.push(Label(id("entry")));

        for stmt in program.stmts {
            self.lower_stmt(stmt);
        }
        // Close the last basic block
        self.tv.push(Term(Terminator::Exit));

        tir::Program {
            decl: self.decl,
            block: construct_cfg(self.tv),
        }
    }

    fn lower_stmt(&mut self, stmt: Stmt) {
        match stmt {
            Stmt::Assign(name, value) => {
                let rhs = self.lower_expr(value);
                self.add_decl(name);
                self.emit(Instruction::Copy {
                    dst: name,
                    src: rhs,
                })
            }
            Stmt::Print(value) => {
                let rhs = self.lower_expr(value);
                self.emit(Instruction::Print(rhs));
            }
            Stmt::Read(name) => {
                self.add_decl(name);
                self.emit(Instruction::Read(name));
            }
            Stmt::If { guard, tt, ff } => {
                let cond = self.lower_expr(guard);
                let b_tt = self.mk_label();
                let b_ff = self.mk_label();
                let join = self.mk_label();

                // end pre-conditional block
                self.emit_term(Terminator::Branch {
                    guard: cond,
                    tt: b_tt,
                    ff: b_ff,
                });

                // true block
                self.emit_label(b_tt);
                for stmt in tt {
                    self.lower_stmt(stmt);
                }
                self.emit_term(Terminator::Jump(join));

                // false block
                self.emit_label(b_ff);
                for stmt in ff {
                    self.lower_stmt(stmt);
                }
                self.emit_term(Terminator::Jump(join));

                // start post-conditional block
                self.emit_label(join);
            }
        }
    }

    /// return the variable holding the value
    fn lower_expr(&mut self, e: Expr) -> Id {
        match e {
            Expr::Var(id) => {
                self.add_decl(id);
                id
            }
            Expr::Const(n) => {
                let lhs = self.mk_var("_const");
                self.emit(Instruction::Const { dst: lhs, src: n });
                lhs
            }
            Expr::BinOp { op, lhs, rhs } => {
                let lhs = self.lower_expr(*lhs);
                let rhs = self.lower_expr(*rhs);
                let dst = self.mk_var("_t");

                self.emit(Instruction::Arith { op, dst, lhs, rhs });
                dst
            }
            Expr::Negate(inner) => {
                let inner = self.lower_expr(*inner);
                let zero = self.lower_expr(Expr::Const(0));
                let dst = self.mk_var("_t");

                self.emit(Instruction::Arith {
                    op: BOp::Sub,
                    dst,
                    lhs: zero,
                    rhs: inner,
                });
                dst
            }
        }
    }

    /// return the variable name
    fn mk_var(&mut self, prefix: &str) -> Id {
        let x = id(&format!("{}{}", prefix, self.fresh_ctr));
        self.fresh_ctr += 1;
        self.add_decl(x);
        x
    }

    /// create and return a fresh label
    fn mk_label(&mut self) -> Id {
        let x = id(&format!("bb{}", self.bb_ctr));
        self.bb_ctr += 1;
        x
    }
}
