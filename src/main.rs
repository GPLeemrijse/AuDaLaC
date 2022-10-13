#[macro_use]
extern crate lalrpop_util;

use codespan::FileId;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self};

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
                println!("{:#?}", program)
            }

            let errors = validate_ast(&program);

            if errors.len() > 0 {
                let file = SimpleFile::new(lopl_file_loc, lopl_program_text);
                let writer = StandardStream::stderr(ColorChoice::Always);
                let config = codespan_reporting::term::Config::default();

                for e in errors {
                    let diagnostic = e.to_diagnostic();
                    let _term_result = term::emit(&mut writer.lock(), &config, &file, &diagnostic);
                }
            }
        }
        Err(e) => {
            panic!("{}", e)
        }
    }
}
