#[macro_use]
extern crate lalrpop_util;
use crate::in_kernel_compiler::{WorkDivisor, DivisionStrategy};
use crate::transpilation_traits::FPStrategy;
use crate::fp_strategies::NaiveAlternatingFixpoint;
use crate::in_kernel_compiler::StepBodyTranspiler;
use crate::fp_strategies::NaiveFixpoint;
use crate::compilation_components::*;
use crate::in_kernel_compiler::SingleKernelSchedule;
use std::io::BufWriter;
use std::fs::File;
use std::io::Write;
use crate::basic_compiler::*;
use crate::coalesced_compiler::*;
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use codespan_reporting::term::{self};

use crate::ast_validator::validate_ast;
use crate::init_file_generator::generate_init_file;
use crate::adl::ProgramParser;
use std::fs;
use crate::cuda_atomics::{MemOrder, Scope};

use clap::clap_app;
mod ast;
mod cuda_atomics;
mod ast_validator;
mod transpilation_traits;
mod basic_compiler;
mod coalesced_compiler;
mod init_file_generator;
mod transpile;
mod compilation_components;
mod in_kernel_compiler;
mod fp_strategies;
mod utils;

lalrpop_mod!(pub adl); // synthesized by LALRPOP



fn main() {
    let args = clap_app!(ADL =>
        (version: "0.1")
        (author: "GPLeemrijse <g.p.leemrijse@student.tue.nl>")
        (about: "Parses \"ADL\" programs")
        (@arg print_ast: -a --ast "Output the AST of the program (skips validation)")
        (@arg init_file: -i --init_file "Output the init file of the program (skips validation)")
        (@arg compiler: -c --compiler possible_value("basic") possible_value("coalesced") possible_value("in-kernel") default_value("coalesced") "Which compiler to use.")
        (@arg memorder: -m --memorder possible_value("weak") possible_value("relaxed") possible_value("acqrel") possible_value("seqcons") default_value("seqcons") "Which memory order to use.")
        (@arg voting: -v --vote_strat possible_value("naive") possible_value("naive-alternating") default_value("naive-alternating") "Which fixpoint stability voting strategy to use.")
        (@arg division_strat: -D --division_strat possible_value("blocksize") possible_value("gridsize") default_value("blocksize") "What division strategy to use. 'blocksize' lets blocks execute a continuous sequence of instances, while 'gridsize' evenly distributes over the blocks.")
        (@arg scope: -s --scope possible_value("system") possible_value("device") default_value("device") "Which scope for atomics to use.")
        (@arg nrofinstances: -N --nrofinstances +takes_value default_value("1000") value_parser(clap::value_parser!(usize)) "nrof struct instances memory is allocated for.")
        (@arg instsperthread: -M --instsperthread +takes_value default_value("8") value_parser(clap::value_parser!(usize)) "Instances executed per thread.")
        (@arg threads_per_block: -T --threadsperblock +takes_value default_value("256") value_parser(clap::value_parser!(usize)) "Number of threads per block.")
        (@arg buffersize: -b --buffersize +takes_value default_value("2048") value_parser(clap::value_parser!(usize)) "CUDA printf buffer size (KB).")
        (@arg printnthinst: -d --printnthinst +takes_value default_value("0") value_parser(clap::value_parser!(usize)) "Print every n'th allocated instance.")
        (@arg printunstable: -u --printunstable "Print which step changed the stability stack.")
        (@arg output: -o --output +takes_value "Output file")
        (@arg file: +required "\"ADL\" file")
    )
    .get_matches();

    let print_ast = args.is_present("print_ast");
    let init_file = args.is_present("init_file");
    let print_unstable = args.is_present("printunstable");
    let nrof_instances_per_struct : usize = *args.get_one("nrofinstances").unwrap();
    let buffer_size : usize = *args.get_one("buffersize").unwrap();
    let instances_per_thread : usize = *args.get_one("instsperthread").unwrap();
    let threads_per_block : usize = *args.get_one("threads_per_block").unwrap();
    let printnthinst : usize = *args.get_one("printnthinst").unwrap();
    let adl_file_loc = args.value_of("file").unwrap();
    let output_file = args.value_of("output");
    let compiler : &str = args.value_of("compiler").unwrap();
    let voting_strat : &str = args.value_of("voting").unwrap();
    let division_strat : &str = args.value_of("division_strat").unwrap();
    let memorder = MemOrder::from_str(args.value_of("memorder").unwrap());
    let scope = Scope::from_str(args.value_of("scope").unwrap());
    

    let adl_program_text = fs::read_to_string(adl_file_loc).expect("Could not open ADL file.");

    let mut output_writer = match output_file {
        Some(x) => {
            Box::new(BufWriter::new(
                File::create(x)
                      .expect("Could not open output file.")
            )) as Box<dyn Write>
        },
        None => Box::new(BufWriter::new(
            std::io::stdout()
        )) as Box<dyn Write>
    };


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
                generate_init_file(&program, output_writer);
            } else {
                let (errors, type_info) = validate_ast(&program);

                if errors.is_empty() {
                    let result : String;

                    match compiler {
                        "basic" => {
                            let struct_manager = BasicStructManager::new(&program, nrof_instances_per_struct);
                            let schedule_manager = BasicScheduleManager::new(&program);
                            result = transpile::transpile(&schedule_manager, &struct_manager);
                        },
                        "coalesced" => {
                            let struct_manager = CoalescedStructManager::new(&program, nrof_instances_per_struct, memorder, scope);
                            let schedule_manager = CoalescedScheduleManager::new(&program, &struct_manager, printnthinst, print_unstable);
                            result = transpile::transpile(&schedule_manager, &struct_manager);
                        },
                        "in-kernel" => {

                            let fp_strat : Box<dyn FPStrategy> = match voting_strat {
                                "naive" => Box::new(NaiveFixpoint::new()),
                                "naive-alternating" => Box::new(NaiveAlternatingFixpoint::new()),
                                _ => panic!("voting strategy not found.")
                            };

                            let div_strat = match division_strat {
                                "blocksize" => DivisionStrategy::BlockSizeIncrease,
                                "gridsize" => DivisionStrategy::GridSizeIncrease,
                                _ => panic!("division strategy not found.")
                            };

                            let step_transpiler = StepBodyTranspiler::new(
                                &type_info,
                                true
                            );

                            let work_divisor = WorkDivisor::new(
                                &program,
                                instances_per_thread,
                                threads_per_block, // tpb
                                nrof_instances_per_struct,
                                div_strat
                            );

                            result = transpile::transpile2(vec![
                                &InitFileReader{},
                                &PrintbufferSizeAdjuster::new(buffer_size),
                                &work_divisor,
                                &StructManagers::new(
                                    &program,
                                    nrof_instances_per_struct,
                                    &memorder,
                                    scope,
                                    true
                                ),
                                &SingleKernelSchedule::new(
                                    &program,
                                    &*fp_strat,
                                    &step_transpiler,
                                    &work_divisor
                                )
                            ]);
                        },
                        _ => unreachable!()
                    }
                    
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
            panic!("{:?}", e)
        }
    }
}
