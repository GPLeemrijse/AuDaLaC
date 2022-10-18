pub type Loc = (usize, usize);

use std::fmt;
use std::fmt::Display;

#[derive(Eq, PartialEq, Debug)]
pub struct Program {
    pub structs: Vec<LoplStruct>,
    pub schedule: Box<Schedule>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Schedule {
    StepCall(String, Loc),
    TypedStepCall(String, String, Loc),
    Sequential(Box<Schedule>, Box<Schedule>, Loc),
    Fixpoint(Box<Schedule>, Loc),
}

#[derive(Eq, PartialEq, Debug)]
pub struct Step {
    pub name: String,
    pub statements: Vec<Stat>,
    pub loc: Loc,
}

#[derive(Eq, PartialEq, Debug)]
pub struct LoplStruct {
    pub name: String,
    pub parameters: Vec<(String, Type, Loc)>,
    pub steps: Vec<Step>,
    pub loc: Loc,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Exp {
    BinOp(Box<Exp>, BinOpcode, Box<Exp>, Loc),
    UnOp(UnOpcode, Box<Exp>, Loc),
    Constructor(String, Vec<Exp>, Loc),
    Var(Vec<String>, Loc),
    Lit(Literal, Loc),
}

#[derive(Eq, PartialEq, Debug)]
pub enum Stat {
    IfThen(Box<Exp>, Vec<Stat>, Loc),
    Declaration(Type, String, Box<Exp>, Loc),
    Assignment(Vec<String>, Box<Exp>, Loc),
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Type {
    Named(String),
    String,
    Nat,
    Int,
    Bool,
}

impl Type {
    pub fn can_be_coerced_to_type(&self, t: &Type) -> bool {
        self == t || (*t == Type::Int && *self == Type::Nat)
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::ast::Type::*;
        match self {
            Named(s) => write!(f, "{}", s),
            String => write!(f, "String"),
            Nat => write!(f, "Nat"),
            Int => write!(f, "Int"),
            Bool => write!(f, "Bool"),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum Literal {
    NatLit(u32),
    IntLit(i32),
    BoolLit(bool),
    StringLit(String),
    NullLit,
    ThisLit,
}

#[derive(Eq, PartialEq, Debug, Clone)]
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

impl Display for BinOpcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::ast::BinOpcode::*;
        match self {
            Equals => write!(f, "=="),
            NotEquals => write!(f, "!="),
            LessThanEquals => write!(f, "<="),
            GreaterThanEquals => write!(f, ">="),
            LessThan => write!(f, "<"),
            GreaterThan => write!(f, ">"),
            Mult => write!(f, "*"),
            Div => write!(f, "/"),
            Mod => write!(f, "%"),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            And => write!(f, "&&"),
            Or => write!(f, "||"),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum UnOpcode {
    Negation,
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::lopl::ExpParser;
    use crate::lopl::ScheduleParser;
    use crate::lopl::StatParser;
    use crate::lopl::StepsParser;
    use crate::lopl::StructsParser;
    use lalrpop_util::ParseError::User;

    #[test]
    fn test_precedence_of_sequential() {
        check_schedule_str(
            "A < B < C",
            "Sequential(StepCall(\"A\", (0, 1)), Sequential(StepCall(\"B\", (4, 5)), StepCall(\"C\", (8, 9)), (4, 9)), (0, 9))",
        );
    }

    #[test]
    fn test_nested_fixpoints() {
        check_schedule_str(
	        "A < Fix(B < Fix(C.D))",
	        "Sequential(StepCall(\"A\", (0, 1)), Fixpoint(Sequential(StepCall(\"B\", (8, 9)), Fixpoint(TypedStepCall(\"C\", \"D\", (16, 19)), (12, 20)), (8, 20)), (4, 21)), (0, 21))",
	    );
    }

    #[test]
    fn test_structs() {
        let result = StructsParser::new().parse(
            "struct  ABC(param1 : Nat, param2: ABC) { step1{ param1 := 2;} } struct DEF () {}",
        );
        assert!(result.is_ok());
        if let Ok(structs) = result {
            assert_eq!(structs.len(), 2);
            assert!(structs[0].name == "ABC".to_string());
            assert_eq!(
                structs[0].parameters,
                vec![
                    ("param1".to_string(), Type::Nat, (12, 24)),
                    (
                        "param2".to_string(),
                        Type::Named("ABC".to_string()),
                        (26, 37)
                    )
                ]
            );
            assert!(structs[0].steps[0].name == "step1".to_string());
            assert!(structs[1].name == "DEF".to_string());
            assert!(structs[1].parameters.len() == 0);
            assert!(structs[1].steps.len() == 0);
        }
    }

    #[test]
    fn test_steps() {
        let result = StepsParser::new().parse("init { a := 2; abc def := 3;} update {}");
        assert!(result.is_ok());
        if let Ok(steps) = result {
            assert_eq!(steps.len(), 2);
            assert!(steps[0].name == "init".to_string());
            assert!(matches!(steps[0].statements[0], Stat::Assignment { .. }));
            assert!(matches!(steps[0].statements[1], Stat::Declaration { .. }));
            assert!(steps[1].name == "update".to_string());
            assert!(steps[1].statements.len() == 0);
        }
    }

    #[test]
    fn test_statements() {
        //Covers: IfThen, Assignment, Declaration
        check_statement(
            "if true then { id.id2 := null; String b := \"a\";}",
            Stat::IfThen(
                Box::new(Exp::Lit(Literal::BoolLit(true), (3, 7))),
                vec![
                    Stat::Assignment(
                        vec!["id".to_string(), "id2".to_string()],
                        Box::new(Exp::Lit(Literal::NullLit, (25, 29))),
                        (15, 30),
                    ),
                    Stat::Declaration(
                        Type::String,
                        "b".to_string(),
                        Box::new(Exp::Lit(Literal::StringLit("a".to_string()), (43, 46))),
                        (31, 47),
                    ),
                ],
                (3, 7),
            ),
        );
    }

    #[test]
    fn test_vars() {
        check_expression(
            "somethingelse",
            Exp::Var(vec!["somethingelse".to_string()], (0, 13)),
        );
        check_expression(
            "some.thing.else",
            Exp::Var(
                vec!["some".to_string(), "thing".to_string(), "else".to_string()],
                (0, 15),
            ),
        );
    }

    #[test]
    fn test_constructors() {
        check_expression(
            "somethingelse(a.b, \"test\")",
            Exp::Constructor(
                "somethingelse".to_string(),
                vec![
                    Exp::Var(vec!["a".to_string(), "b".to_string()], (14, 17)),
                    Exp::Lit(Literal::StringLit("test".to_string()), (19, 25)),
                ],
                (0, 26),
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
        check_expression("this", Exp::Lit(Literal::ThisLit, (0, 4)));
        check_expression("null", Exp::Lit(Literal::NullLit, (0, 4)));
        check_expression("123", Exp::Lit(Literal::NatLit(123), (0, 3)));
        check_expression("-123", Exp::Lit(Literal::IntLit(-123), (0, 4)));
        check_expression("true", Exp::Lit(Literal::BoolLit(true), (0, 4)));
        check_expression(
            "\"true\"",
            Exp::Lit(Literal::StringLit("true".to_string()), (0, 6)),
        );
    }

    #[test]
    fn test_binops() {
        check_expression(
            "1 + 2",
            Exp::BinOp(
                Box::new(Exp::Lit(Literal::NatLit(1), (0, 1))),
                BinOpcode::Plus,
                Box::new(Exp::Lit(Literal::NatLit(2), (4, 5))),
                (0, 5),
            ),
        );

        check_expression(
            "-1 + 2 * 3",
            Exp::BinOp(
                Box::new(Exp::Lit(Literal::IntLit(-1), (0, 2))),
                BinOpcode::Plus,
                Box::new(Exp::BinOp(
                    Box::new(Exp::Lit(Literal::NatLit(2), (5, 6))),
                    BinOpcode::Mult,
                    Box::new(Exp::Lit(Literal::NatLit(3), (9, 10))),
                    (5, 10),
                )),
                (0, 10),
            ),
        );

        check_expression(
            "(!1 + 2) * 3 == null / this || -4",
            Exp::BinOp(
                Box::new(Exp::BinOp(
                    Box::new(Exp::BinOp(
                        Box::new(Exp::BinOp(
                            Box::new(Exp::UnOp(
                                UnOpcode::Negation,
                                Box::new(Exp::Lit(Literal::NatLit(1), (2, 3))),
                                (1, 3),
                            )),
                            BinOpcode::Plus,
                            Box::new(Exp::Lit(Literal::NatLit(2), (6, 7))),
                            (1, 7),
                        )),
                        BinOpcode::Mult,
                        Box::new(Exp::Lit(Literal::NatLit(3), (11, 12))),
                        (0, 12),
                    )),
                    BinOpcode::Equals,
                    Box::new(Exp::BinOp(
                        Box::new(Exp::Lit(Literal::NullLit, (16, 20))),
                        BinOpcode::Div,
                        Box::new(Exp::Lit(Literal::ThisLit, (23, 27))),
                        (16, 27),
                    )),
                    (0, 27),
                )),
                BinOpcode::Or,
                Box::new(Exp::Lit(Literal::IntLit(-4), (31, 33))),
                (0, 33),
            ),
        );
    }

    fn check_schedule_str(string: &str, s: &str) {
        match ScheduleParser::new().parse(string) {
            Ok(sched) => {
                let r = format!("{:?}", sched);
                if s != r {
                    panic!(
                        "The string '{}' does not parse to:\n'{}',\nreceived:\n'{}'",
                        string, s, r
                    );
                }
            }
            Err(e) => panic!("{}", e),
        };
    }

    fn check_expression(string: &str, e: Exp) {
        match ExpParser::new().parse(string) {
            Ok(exp) => {
                if exp != e {
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
                if stat != s {
                    panic!(
                        "The string '{}' is not equal to:\n'{:?}'\nreceived:\n'{:?}'",
                        string, s, stat
                    );
                }
            }
            Err(e) => panic!("{}", e),
        };
    }
}
