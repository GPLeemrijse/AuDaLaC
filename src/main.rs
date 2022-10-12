#[macro_use]
extern crate lalrpop_util;
use crate::lopl::ProgramParser;
use std::fs;

use clap::clap_app;
mod ast;

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

    if print_ast {
        let lopl_program = ProgramParser::new().parse(&lopl_program_text);
        match lopl_program {
            Ok(program) => println!("{:#?}", program),
            Err(e) => panic!("{}", e),
        }
    }
}
