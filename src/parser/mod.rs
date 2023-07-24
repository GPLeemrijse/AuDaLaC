pub mod ast;
mod ast_validator;
mod validation_error;


lalrpop_mod!(pub grammar, "/parser/grammar.rs"); // synthesized by LALRPOP

pub use ast_validator::validate_ast;

#[allow(dead_code)]
pub use grammar::*;
