#[macro_use]
extern crate lalrpop_util;
use crate::ast_validator::validate_ast;
use crate::lopl::ProgramParser;
use std::fs;

use clap::clap_app;
mod ast;
mod ast_validator;

lalrpop_mod!(pub lopl); // synthesized by LALRPOP

fn main() {
    let args = clap_app!(LOPL =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Parses \"LOPL\" programs")
        (@arg print_ast: -p --print-ast "Print the AST of the program")
        (@arg file: +required "\"LOPL\" file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
    let lopl_file_loc = args.value_of("file").unwrap();
    let lopl_program_text = fs::read_to_string(lopl_file_loc).expect("Could not open file");


    let lopl_program = ProgramParser::new().parse(&lopl_program_text);

    match lopl_program {
        Ok(program) => {
            if print_ast {
                println!("@@@@@@@@@@@@@@@@@@ AST @@@@@@@@@@@@@@@@@@");
                println!("{:#?}", program)
            }

            let errors = validate_ast(&program);
            if errors.len() > 0 {
                println!("@@@@@@@@@@@@@@@@@@ ERRORS @@@@@@@@@@@@@@@@@@");
                for e in errors {
                    println!("{:#?}", e);
                }
            }
        },
        Err(e) => {
            panic!("{}", e)
        },
    }
}
