#[macro_use]
extern crate lalrpop_util;
use crate::analysis::fixpoint_depth;
use crate::backend::components::*;
use crate::backend::fp_strategies::*;
use crate::backend::utils::{MemOrder, Scope};
use crate::backend::*;
use crate::init_file_generator::generate_init_file;
use crate::frontend::validate_ast;
use crate::frontend::ProgramParser;
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
mod backend;
mod init_file_generator;
mod frontend;
mod analysis;

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
    let args = clap_app!(AuDauLaC =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Compiles AuDauLa programs")
        (@arg print_ast: -a --ast "Output the AST of the program (skips validation)")
        (@arg time: -t --time "Print timing information.")
        (@arg init_file: -i --init_file "Output the init file of the program (skips validation)")
        (@arg schedule_strat: -S --schedule_strat
            possible_value("graph")
            possible_value("in-kernel")
            possible_value("on-host")
            default_value("graph")
            "Which schedule strategy to use.")
        (@arg memorder: -m --memorder
            possible_value("weak")
            possible_value("relaxed")
            possible_value("acqrel")
            possible_value("seqcons")
            default_value("relaxed")
            "Which memory order to use.")
        (@arg fpstrat: -f --fp_strat
            possible_value("graph-simple")
            possible_value("graph-shared")
            possible_value("graph-shared-banks")
            possible_value("graph-shared-opportunistic")
            possible_value("on-host-alternating")
            possible_value("on-host-simple")
            possible_value("in-kernel-simple")
            possible_value("in-kernel-alternating")
            default_value("graph-simple")
            "Which fixpoint stability synchronisation strategy to use.")
        (@arg weak_ld_st: -w --weak_ld_st
            possible_value("1")
            possible_value("0")
            default_value("1")
            "Use weak loads and stores for non-racing parameters.")
        (@arg scope: -s --scope possible_value("system") possible_value("device") possible_value("block") default_value("device") "Which scope for atomics to use.")
        (@arg nrofinstances: -N --nrofinstances +takes_value required(false) multiple(true) value_parser(parse_key_val::<String, usize>) "nrof struct instances memory is allocated for.")
        (@arg buffersize: -b --buffersize +takes_value default_value("1024") value_parser(clap::value_parser!(usize)) "CUDA printf buffer size (KB).")
        (@arg printunstable: -u --printunstable "Print which step changed the stability stack.")
        (@arg output: -o --output +takes_value "Output file")
        (@arg file: +required "AuDauLa (.adl) file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
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
    let adl_file_loc = args.value_of("file").unwrap();
    let output_file = args.value_of("output");
    let schedule_strat: &str = args.value_of("schedule_strat").unwrap();
    let fp_strat: &str = args.value_of("fpstrat").unwrap();
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
                    let fp_strat: Box<dyn FPStrategy> = match (schedule_strat, fp_strat) {
                        ("in-kernel", "in-kernel-simple") => {
                            Box::new(InKernelSimpleFixpoint::new(fixpoint_depth(&program.schedule)))
                        }
                        ("in-kernel", "in-kernel-alternating") => Box::new(
                            InKernelAlternatingFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("on-host", "on-host-simple") => Box::new(
                            OnHostSimpleFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("on-host", "on-host-alternating") => Box::new(
                            OnHostAlternatingFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("graph", "graph-shared") => Box::new(
                            GraphSharedFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("graph", "graph-shared-banks") => Box::new(
                            GraphSharedBanksFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("graph", "graph-shared-opportunistic") => Box::new(
                            GraphSharedBanksOpportunisticFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        ("graph", "graph-simple") => Box::new(
                            GraphSimpleFixpoint::new(fixpoint_depth(&program.schedule)),
                        ),
                        _ => panic!(
                            "Voting strategy not found, or combined with wrong schedule strategy."
                        ),
                    };

                    let step_transpiler =
                        StepBodyCompiler::new(&type_info, true, print_unstable, weak_ld_st, &memorder, &program, schedule_strat != "in-kernel");

                    let work_divisor = WorkDivisor::new(
                        DivisionStrategy::Dynamic,
                        print_unstable
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
                            &step_transpiler
                        )),
                        "graph" => Box::new(GraphLaunchSchedule::new(
                            &program,
                            &*fp_strat,
                            &step_transpiler,
                        )),
                        _ => panic!("Schedule strategy not found."),
                    };

                    let result = backend::compile(vec![
                        &InitFileReader {},
                        &PrintbufferSizeAdjuster::new(buffer_size),
                        &work_divisor,
                        &struct_managers,
                        &*schedule,
                        &Timer::new(
                            time,
                            if schedule_strat == "in-kernel" {
                                None
                            } else {
                                Some("kernel_stream".to_string())
                            }
                        ),
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

    if !errors.is_empty() {
        let file = SimpleFile::new(adl_file_loc, adl_program_text);
        let writer = StandardStream::stderr(ColorChoice::Always);
        let config = codespan_reporting::term::Config::default();

        for e in errors {
            let _term_result = term::emit(&mut writer.lock(), &config, &file, &e);
        }
        std::process::exit(1);
    }
}
