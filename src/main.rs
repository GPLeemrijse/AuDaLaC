#[macro_use]
extern crate lalrpop_util;
use std::io::BufWriter;
use std::fs::File;
use std::io::Write;
use crate::basic_transpiler::BasicCUDATranspiler;
use crate::basic_schedule_manager::BasicScheduleManager;
use crate::basic_struct_manager::BasicStructManager;
use crate::transpilation_traits::Transpiler;
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self};

use crate::ast_validator::validate_ast;
use crate::init_file_generator::generate_init_file;
use crate::adl::ProgramParser;
use std::fs;

use clap::clap_app;
mod ast;
mod ast_validator;
mod transpilation_traits;
mod basic_transpiler;
mod basic_schedule_manager;
mod basic_struct_manager;
mod init_file_generator;

lalrpop_mod!(pub adl); // synthesized by LALRPOP

fn main() {
    let args = clap_app!(ADL =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Parses \"ADL\" programs")
        (@arg print_ast: -a --ast "Output the AST of the program (skips validation)")
        (@arg init_file: -i --init_file "Output the init file of the program (skips validation)")
        (@arg nrofstructs: -n --nrofstructs +takes_value "nrof structs memory is allocated for.")
        (@arg output: -o --output +takes_value +required "Output file")
        (@arg file: +required "\"ADL\" file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
    let init_file = args.is_present("init_file");
    let nrof_structs = if args.is_present("nrofstructs") { args.value_of("nrofstructs").unwrap().parse::<u64>().unwrap()} else {100};
    let adl_file_loc = args.value_of("file").unwrap();
    let output_file = args.value_of("output").unwrap();

    let adl_program_text = fs::read_to_string(adl_file_loc).expect("Could not open ADL file.");
    let mut output_writer = BufWriter::new(
        File::create(output_file)
        .expect("Could not open output file.")
    );

    if print_ast && init_file {
        eprintln!("Use either --print-ast (-p) or --init-file (-i), not both.");
        std::process::exit(1);
    }

    let adl_program = ProgramParser::new().parse(&adl_program_text);

    match adl_program {
        Ok(program) => {
            if print_ast {
                output_writer.write(
                    format!("{:#?}\n", program).as_bytes()
                ).expect("Could not write to output file.");
            } else if init_file {
                generate_init_file(&program, &mut output_writer);
            } else {
                let errors = validate_ast(&program);

                if errors.is_empty() {
                    let schedule_manager = BasicScheduleManager::new(&program);
                    let struct_manager = BasicStructManager::new(&program, nrof_structs);
                    let result = BasicCUDATranspiler::transpile(&schedule_manager, &struct_manager);

                    output_writer.write(result.as_bytes()).expect("Could not write to output file.");
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
        }
        Err(e) => {
            panic!("{}", e)
        }
    }
}
