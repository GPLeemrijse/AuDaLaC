#[macro_use]
extern crate lalrpop_util;
use crate::analysis::fixpoint_depth;
use crate::coalesced_compiler::*;
use crate::compiler::components::*;
use crate::compiler::fp_strategies::*;
use crate::compiler::utils::{MemOrder, Scope};
use crate::compiler::*;
use crate::init_file_generator::generate_init_file;
use crate::parser::validate_ast;
use crate::parser::ProgramParser;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self};
use lalrpop_util::ParseError;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use clap::clap_app;
mod coalesced_compiler;
mod compiler;
mod init_file_generator;
mod parser;
mod analysis;

/* https://github.com/clap-rs/clap/blob/master/examples/typed-derive.rs */
fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
    U: std::str::FromStr,
    U::Err: Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

fn main() {
    let args = clap_app!(ADL =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Parses \"ADL\" programs")
        (@arg print_ast: -a --ast "Output the AST of the program (skips validation)")
        (@arg dynamic_bs: -d --dynamic "Dynamically determine the block size.")
        (@arg time: -t --time "Print timing information.")
        (@arg init_file: -i --init_file "Output the init file of the program (skips validation)")
        (@arg schedule_strat: -S --schedule_strat possible_value("in-kernel") possible_value("on-host") default_value("in-kernel") "Which schedule strategy to use.")
        (@arg memorder: -m --memorder possible_value("weak") possible_value("relaxed") possible_value("acqrel") possible_value("seqcons") default_value("relaxed") "Which memory order to use.")
        (@arg voting: -v --vote_strat possible_value("on-host-alternating") possible_value("on-host-naive") possible_value("naive") possible_value("naive-alternating") default_value("naive-alternating") "Which fixpoint stability voting strategy to use.")
        (@arg weak_ld_st: -w --weak_ld_st possible_value("1") possible_value("0") default_value("1") "Use weak loads and stores for non-racing parameters.")
        (@arg scope: -s --scope possible_value("system") possible_value("device") default_value("device") "Which scope for atomics to use.")
        (@arg nrofinstances: -N --nrofinstances +takes_value required(false) multiple(true) value_parser(parse_key_val::<String, usize>) "nrof struct instances memory is allocated for.")
        (@arg threads_per_block: -T --threadsperblock +takes_value default_value("256") value_parser(clap::value_parser!(usize)) "Number of threads per block.")
        (@arg buffersize: -b --buffersize +takes_value default_value("1024") value_parser(clap::value_parser!(usize)) "CUDA printf buffer size (KB).")
        (@arg printunstable: -u --printunstable "Print which step changed the stability stack.")
        (@arg output: -o --output +takes_value "Output file")
        (@arg file: +required "\"ADL\" file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
    let dynamic_block_size = args.is_present("dynamic_bs");
    let weak_ld_st = args.value_of("weak_ld_st").unwrap() == "1";
    let time = args.is_present("time");
    let init_file = args.is_present("init_file");
    let print_unstable = args.is_present("printunstable");
    let nrof_instances_per_struct: HashMap<String, usize> = args
        .get_many::<(String, usize)>("nrofinstances")
        .unwrap_or_default()
        .map(|(s, n)| (s.clone(), *n))
        .collect();
    let buffer_size: usize = *args.get_one("buffersize").unwrap();
    let threads_per_block: usize = *args.get_one("threads_per_block").unwrap();
    let adl_file_loc = args.value_of("file").unwrap();
    let output_file = args.value_of("output");
    let schedule_strat: &str = args.value_of("schedule_strat").unwrap();
    let voting_strat: &str = args.value_of("voting").unwrap();
    let memorder = MemOrder::from_str(args.value_of("memorder").unwrap());
    let scope = Scope::from_str(args.value_of("scope").unwrap());

    let adl_program_text = fs::read_to_string(adl_file_loc).expect("Could not open ADL file.");

    let mut output_writer = match output_file {
        Some(x) => Box::new(BufWriter::new(
            File::create(x).expect("Could not open output file."),
        )) as Box<dyn Write>,
        None => Box::new(BufWriter::new(std::io::stdout())) as Box<dyn Write>,
    };

    if print_ast && init_file {
        eprintln!("Use either --print-ast (-p) or --init-file (-i), not both.");
        std::process::exit(1);
    }

    let adl_program = ProgramParser::new().parse(&adl_program_text);
    let mut errors = Vec::new();

    match adl_program {
        Ok(program) => {
            if print_ast {
                output_writer
                    .write(format!("{:#?}\n", program).as_bytes())
                    .expect("Could not write to output file.");
            } else if init_file {
                generate_init_file(&program, output_writer);
            } else {
                let (validation_errors, type_info) = validate_ast(&program);

                if !validation_errors.is_empty() {
                    errors.append(
                        &mut validation_errors
                            .iter()
                            .map(|e| e.to_diagnostic())
                            .collect(),
                    );
                } else {
                    let fp_strat: Box<dyn FPStrategy> = match (schedule_strat, voting_strat) {
                        ("in-kernel", "naive") => {
                            Box::new(NaiveFixpoint::new(fixpoint_depth(&program.schedule)))
                        }
                        ("in-kernel", "naive-alternating") => Box::new(
                            NaiveAlternatingFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("on-host", "on-host-naive") => Box::new(
                            OnHostNaiveFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("on-host", "on-host-alternating") => Box::new(
                            OnHostAlternatingFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        _ => panic!(
                            "Voting strategy not found, or combined with wrong schedule strategy."
                        ),
                    };

                    let step_transpiler =
                        StepBodyCompiler::new(&type_info, true, print_unstable, weak_ld_st, &memorder, &program);

                    let work_divisor = WorkDivisor::new(
                        threads_per_block,
                        DivisionStrategy::Dynamic,
                        print_unstable,
                        schedule_strat == "on-host", // We inline the executeStep function with the on-host schedule
                    );

                    let struct_managers = StructManagers::new(
                        &program,
                        &nrof_instances_per_struct,
                        &memorder,
                        scope,
                        true,
                    );

                    let schedule: Box<dyn CompileComponent> = match schedule_strat {
                        "in-kernel" => Box::new(SingleKernelSchedule::new(
                            &program,
                            &*fp_strat,
                            &step_transpiler,
                            &work_divisor,
                        )),
                        "on-host" => Box::new(OnHostSchedule::new(
                            &program,
                            &*fp_strat,
                            &step_transpiler,
                            &work_divisor,
                            dynamic_block_size
                        )),
                        _ => panic!("Schedule strategy not found."),
                    };

                    let result = compiler::compile2(vec![
                        &InitFileReader {},
                        &PrintbufferSizeAdjuster::new(buffer_size),
                        &work_divisor,
                        &struct_managers,
                        &*schedule,
                        &Timer::new(time),
                    ]);

                    output_writer
                        .write(result.as_bytes())
                        .expect("Could not write to output file.");
                }
            }
        }
        Err(e) => {
            let range = match e {
                ParseError::InvalidToken { location } => location..(location + 1),
                ParseError::UnrecognizedEOF {
                    location,
                    expected: _,
                } => location..(location + 1),
                ParseError::UnrecognizedToken { token, expected: _ } => token.0..token.2,
                ParseError::ExtraToken { token } => token.0..token.2,
                _ => panic!("{:?}", e),
            };

            let d_err = Diagnostic::error()
                .with_message("Parsing error.")
                .with_labels(vec![Label::primary((), range).with_message("here")]);
            errors.push(d_err);
        }
    }

    let file = SimpleFile::new(adl_file_loc, adl_program_text);
    let writer = StandardStream::stderr(ColorChoice::Always);
    let config = codespan_reporting::term::Config::default();

    for e in errors {
        let _term_result = term::emit(&mut writer.lock(), &config, &file, &e);
    }
}
