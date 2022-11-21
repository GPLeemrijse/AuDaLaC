#[macro_use]
extern crate lalrpop_util;
use crate::basic_transpiler::BasicCUDATranspiler;
use crate::basic_schedule_manager::BasicScheduleManager;
use crate::basic_struct_manager::BasicStructManager;
use crate::transpilation_traits::Transpiler;
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self};

use crate::ast_validator::validate_ast;
use crate::adl::ProgramParser;
use std::fs;

use clap::clap_app;
mod ast;
mod ast_validator;
mod transpilation_traits;
mod basic_transpiler;
mod basic_schedule_manager;
mod basic_struct_manager;

lalrpop_mod!(pub adl); // synthesized by LALRPOP

fn main() {
    let args = clap_app!(ADL =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Parses \"ADL\" programs")
        (@arg print_ast: -p --print-ast "Print the AST of the program")
        (@arg output: -o --output +takes_value +required "Output .cu file")
        (@arg file: +required "\"ADL\" file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
    let adl_file_loc = args.value_of("file").unwrap();
    let output_file = args.value_of("output").unwrap();
    let adl_program_text = fs::read_to_string(adl_file_loc).expect("Could not open file");

    let adl_program = ProgramParser::new().parse(&adl_program_text);

    match adl_program {
        Ok(program) => {
            if print_ast {
                println!("{:#?}", program)
            }

            let errors = validate_ast(&program);

            if errors.is_empty() {
                let schedule_manager = BasicScheduleManager::new(&program);
                let struct_manager = BasicStructManager::new(&program);
                let result = BasicCUDATranspiler::transpile(&schedule_manager, &struct_manager);

                fs::write(output_file, result).expect("Unable to write output file.");

            } else { // Print errors
                let file = SimpleFile::new(adl_file_loc, adl_program_text);
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
