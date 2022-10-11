#[macro_use] extern crate lalrpop_util;

mod ast;

lalrpop_mod!(pub lopl); // synthesized by LALRPOP

fn main() {
    println!("Hello, world!");
}

