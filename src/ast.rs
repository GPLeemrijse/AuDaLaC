use crate::lopl::ExpParser;
use crate::lopl::StatParser;
use lalrpop_util::ParseError::User;
use std::fmt::{Debug, Error, Formatter};

#[derive(Eq, PartialEq)]
pub enum Exp {
    BinOp(Box<Exp>, BinOpcode, Box<Exp>),
    UnOp(UnOpcode, Box<Exp>),
    Constructor(String, Vec<Box<Exp>>),
    Var(Vec<String>),
    Lit(Literal),
}

#[derive(Eq, PartialEq)]
pub enum Stat {
    IfThen(Box<Exp>, Vec<Box<Stat>>),
    Declaration(Type, String, Box<Exp>),
    Assignment(Vec<String>, Box<Exp>),
}

#[derive(Eq, PartialEq)]
pub enum Type {
    NamedType(String),
    StringType,
    NatType,
    IntType,
    BoolType,
}

#[derive(Eq, PartialEq)]
pub enum Literal {
    NumLit(i64),
    BoolLit(bool),
    StringLit(String),
    NullLit,
    ThisLit,
}

#[derive(Eq, PartialEq)]
pub enum BinOpcode {
    Equals,
    NotEquals,
    LessThanEquals,
    GreaterThanEquals,
    LessThan,
    GreaterThan,
    Mult,
    Div,
    Mod,
    Plus,
    Minus,
    And,
    Or,
}

#[derive(Eq, PartialEq)]
pub enum UnOpcode {
    Negation,
}

impl Debug for Exp {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Exp::*;
        match &*self {
            BinOp(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            UnOp(op, ref l) => write!(fmt, "({:?}{:?})", op, l),
            Constructor(s, v) => write!(fmt, "{}({:?})", s, v),
            Var(v) => write!(fmt, "_Var({:?})", v),
            Lit(l) => write!(fmt, "_Lit({:?})", l),
        }
    }
}

impl Debug for Stat {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Stat::*;
        match &*self {
            IfThen(cond, stmts) => write!(fmt, "if({:?}){{{:?}}}", cond, stmts),
            Declaration(typ, id, exp) => write!(fmt, "{:?} {} := {:?};", typ, id, exp),
            Assignment(ids, exp) => write!(fmt, "{:?} := {:?};", ids, exp),
        }
    }
}

impl Debug for Type {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Type::*;
        match &*self {
            NamedType(s) => write!(fmt, "T({})", s),
            StringType => write!(fmt, "String"),
            NatType => write!(fmt, "Nat"),
            IntType => write!(fmt, "Int"),
            BoolType => write!(fmt, "Bool"),
        }
    }
}

impl Debug for BinOpcode {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::BinOpcode::*;
        match *self {
            Equals => write!(fmt, "=="),
            NotEquals => write!(fmt, "!="),
            LessThanEquals => write!(fmt, "<="),
            GreaterThanEquals => write!(fmt, ">="),
            LessThan => write!(fmt, "<"),
            GreaterThan => write!(fmt, ">"),
            Mult => write!(fmt, "*"),
            Div => write!(fmt, "\\"),
            Mod => write!(fmt, "%"),
            Plus => write!(fmt, "+"),
            Minus => write!(fmt, "-"),
            And => write!(fmt, "&&"),
            Or => write!(fmt, "||"),
        }
    }
}

impl Debug for UnOpcode {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::UnOpcode::*;
        match *self {
            Negation => write!(fmt, "!"),
        }
    }
}

impl Debug for Literal {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Literal::*;
        match &*self {
            NumLit(n) => write!(fmt, "NumLit({})", n),
            BoolLit(b) => write!(fmt, "BoolLit({})", b),
            StringLit(s) => write!(fmt, "StrLit({})", s),
            NullLit => write!(fmt, "NullLit"),
            ThisLit => write!(fmt, "ThisLit"),
        }
    }
}

#[test]
fn test_statements() {
    //Covers: IfThen, Assignment, Declaration
    check_statement(
        "if true then { id.id2 := null; String b := \"a\";}",
        Stat::IfThen(
            Box::new(Exp::Lit(Literal::BoolLit(true))),
            vec![
                Box::new(Stat::Assignment(
                    vec!["id".to_string(), "id2".to_string()],
                    Box::new(Exp::Lit(Literal::NullLit)),
                )),
                Box::new(Stat::Declaration(
                    Type::StringType,
                    "b".to_string(),
                    Box::new(Exp::Lit(Literal::StringLit("a".to_string()))),
                )),
            ],
        ),
    );
}

#[test]
fn test_vars() {
    check_expression("somethingelse", Exp::Var(vec!["somethingelse".to_string()]));
    check_expression(
        "some.thing.else",
        Exp::Var(vec![
            "some".to_string(),
            "thing".to_string(),
            "else".to_string(),
        ]),
    );
}

#[test]
fn test_constructors() {
    check_expression(
        "somethingelse(a.b, \"test\")",
        Exp::Constructor(
            "somethingelse".to_string(),
            vec![
                Box::new(Exp::Var(vec!["a".to_string(), "b".to_string()])),
                Box::new(Exp::Lit(Literal::StringLit("test".to_string()))),
            ],
        ),
    );
}

#[test]
fn test_no_keywords_or_lits_as_constructors() {
    let _r = ExpParser::new().parse("this(test)").is_err();
    let _r = ExpParser::new().parse("null(test)").is_err();
    let _r = ExpParser::new().parse("0(test)").is_err();
    let _r = ExpParser::new().parse("\"abc\"(test)").is_err();
}

#[test]
fn test_constructors_fail_with_dots() {
    let _r = ExpParser::new().parse("some.thing(else)");
    assert!(
        _r == Err(User {
            error: "Expected Simple ID (= without '.')."
        })
    );
}

#[test]
fn test_literals() {
    check_expression("this", Exp::Lit(Literal::ThisLit));
    check_expression("null", Exp::Lit(Literal::NullLit));
    check_expression("123", Exp::Lit(Literal::NumLit(123)));
    check_expression("true", Exp::Lit(Literal::BoolLit(true)));
    check_expression("\"true\"", Exp::Lit(Literal::StringLit("true".to_string())));
}

#[test]
fn test_binops() {
    check_expression(
        "1 + 2",
        Exp::BinOp(
            Box::new(Exp::Lit(Literal::NumLit(1))),
            BinOpcode::Plus,
            Box::new(Exp::Lit(Literal::NumLit(2))),
        ),
    );

    check_expression(
        "1 + 2 * 3",
        Exp::BinOp(
            Box::new(Exp::Lit(Literal::NumLit(1))),
            BinOpcode::Plus,
            Box::new(Exp::BinOp(
                Box::new(Exp::Lit(Literal::NumLit(2))),
                BinOpcode::Mult,
                Box::new(Exp::Lit(Literal::NumLit(3))),
            )),
        ),
    );

    check_expression(
        "(!1 + 2) * 3 == null / this || 4",
        Exp::BinOp(
            Box::new(Exp::BinOp(
                Box::new(Exp::BinOp(
                    Box::new(Exp::BinOp(
                        Box::new(Exp::UnOp(
                            UnOpcode::Negation,
                            Box::new(Exp::Lit(Literal::NumLit(1))),
                        )),
                        BinOpcode::Plus,
                        Box::new(Exp::Lit(Literal::NumLit(2))),
                    )),
                    BinOpcode::Mult,
                    Box::new(Exp::Lit(Literal::NumLit(3))),
                )),
                BinOpcode::Equals,
                Box::new(Exp::BinOp(
                    Box::new(Exp::Lit(Literal::NullLit)),
                    BinOpcode::Div,
                    Box::new(Exp::Lit(Literal::ThisLit)),
                )),
            )),
            BinOpcode::Or,
            Box::new(Exp::Lit(Literal::NumLit(4))),
        ),
    );
}

fn check_expression(string: &str, e: Exp) {
    match ExpParser::new().parse(string) {
        Ok(exp) => {
            if *exp != e {
                panic!(
                    "The string '{}' is not equal to:\n'{:?}'\nreceived:\n'{:?}'",
                    string, e, exp
                );
            }
        }
        Err(e) => panic!("{}", e),
    };
}

fn check_statement(string: &str, s: Stat) {
    match StatParser::new().parse(string) {
        Ok(stat) => {
            if *stat != s {
                panic!(
                    "The string '{}' is not equal to:\n'{:?}'\nreceived:\n'{:?}'",
                    string, s, stat
                );
            }
        }
        Err(e) => panic!("{}", e),
    };
}
