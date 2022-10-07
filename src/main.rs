#[macro_use] extern crate lalrpop_util;

use crate::ast::{Exp, Literal, BinOpcode, UnOpcode};

mod ast;

lalrpop_mod!(pub lopl); // synthesized by LALRPOP

fn main() {
    println!("Hello, world!");
}


#[test]
fn test_literals() {
    check_expression("this", Exp::Lit(Literal::ThisLit));
    check_expression("null", Exp::Lit(Literal::NullLit));
    check_expression("123", Exp::Lit(Literal::NumLit(123)));
    check_expression("true", Exp::Lit(Literal::BoolLit(true)));
}

#[test]
fn test_binops() {
    check_expression("1 + 2",
        Exp::BinOp(
            Box::new(Exp::Lit(Literal::NumLit(1))),
            BinOpcode::Plus,
            Box::new(Exp::Lit(Literal::NumLit(2)))
        )
    );

    check_expression("1 + 2 * 3",
        Exp::BinOp(
            Box::new(Exp::Lit(Literal::NumLit(1))),
            BinOpcode::Plus,
            Box::new(Exp::BinOp(
                Box::new(Exp::Lit(Literal::NumLit(2))),
                BinOpcode::Mult,
                Box::new(Exp::Lit(Literal::NumLit(3))),
            ))
        )
    );

    check_expression("(!1 + 2) * 3 == null / this || 4",
        Exp::BinOp(
            Box::new(Exp::BinOp(
                Box::new(Exp::BinOp(
                    Box::new(Exp::BinOp(
                        Box::new(Exp::UnOp(
                            UnOpcode::Negation,
                            Box::new(Exp::Lit(Literal::NumLit(1))),
                        )),
                        BinOpcode::Plus,
                        Box::new(Exp::Lit(Literal::NumLit(2)))
                    )),
                    BinOpcode::Mult,
                    Box::new(Exp::Lit(Literal::NumLit(3)))
                )),
                BinOpcode::Equals,
                Box::new(Exp::BinOp(
                    Box::new(Exp::Lit(Literal::NullLit)),
                    BinOpcode::Div,
                    Box::new(Exp::Lit(Literal::ThisLit)),
                )),
            )),
            BinOpcode::Or,
            Box::new(Exp::Lit(Literal::NumLit(4)))
        )
    );
}


fn check_expression(string : &str, e : Exp) {
    match lopl::ExpParser::new().parse(string) {
        Ok(lit) => {
            if *lit != e {
                panic!("The string '{}' is not equal to:\n'{:?}'\nreceived:\n'{:?}'", string, e, lit);
            }
        },
        Err(e) => panic!("{}", e)
    };
}
