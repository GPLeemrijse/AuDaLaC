pub mod ast;
mod ast_validator;

lalrpop_mod!(pub grammar, "/parser/grammar.rs"); // synthesized by LALRPOP

pub use ast_validator::validate_ast;

pub use grammar::*;