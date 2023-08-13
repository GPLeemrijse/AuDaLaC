pub mod ast;
mod ast_validator;
mod validation_error;

lalrpop_mod!(#[allow(dead_code)]pub grammar, "/frontend/grammar.rs"); // synthesized by LALRPOP

pub use ast_validator::validate_ast;

pub use grammar::*;
