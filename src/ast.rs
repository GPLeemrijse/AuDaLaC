pub type Loc = (usize, usize);

#[derive(Eq, PartialEq, Debug)]
pub struct Program {
    pub structs: Vec<Box<LoplStruct>>,
    pub schedule: Box<Schedule>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Schedule {
    StepCall(String),
    TypedStepCall(String, String),
    Sequential(Box<Schedule>, Box<Schedule>),
    Fixpoint(Box<Schedule>),
}

#[derive(Eq, PartialEq, Debug)]
pub struct Step {
    pub name: String,
    pub statements: Vec<Box<Stat>>,
    pub loc: Loc,
}

#[derive(Eq, PartialEq, Debug)]
pub struct LoplStruct {
    pub name: String,
    pub parameters: Vec<(String, Type, Loc)>,
    pub steps: Vec<Box<Step>>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Exp {
    BinOp(Box<Exp>, BinOpcode, Box<Exp>),
    UnOp(UnOpcode, Box<Exp>),
    Constructor(String, Vec<Box<Exp>>),
    Var(Vec<String>),
    Lit(Literal),
}

#[derive(Eq, PartialEq, Debug)]
pub enum Stat {
    IfThen(Box<Exp>, Vec<Box<Stat>>),
    Declaration(Type, String, Box<Exp>, Loc),
    Assignment(Vec<String>, Box<Exp>, Loc),
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Type {
    NamedType(String),
    StringType,
    NatType,
    IntType,
    BoolType,
}

#[derive(Eq, PartialEq, Debug)]
pub enum Literal {
    NumLit(i64),
    BoolLit(bool),
    StringLit(String),
    NullLit,
    ThisLit,
}

#[derive(Eq, PartialEq, Debug)]
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
            "Sequential(StepCall(\"A\"), Sequential(StepCall(\"B\"), StepCall(\"C\")))",
        );
    }

    #[test]
    fn test_nested_fixpoints() {
        check_schedule_str(
	        "A < Fix(B < Fix(C.D))",
	        "Sequential(StepCall(\"A\"), Fixpoint(Sequential(StepCall(\"B\"), Fixpoint(TypedStepCall(\"C\", \"D\")))))",
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
            assert!((*structs[0]).name == "ABC".to_string());
            assert_eq!(
                (*structs[0]).parameters,
                vec![
                    ("param1".to_string(), Type::NatType, (12, 24)),
                    (
                        "param2".to_string(),
                        Type::NamedType("ABC".to_string()),
                        (26, 37)
                    )
                ]
            );
            assert!(*structs[0].steps[0].name == "step1".to_string());
            assert!((*structs[1]).name == "DEF".to_string());
            assert!((*structs[1]).parameters.len() == 0);
            assert!((*structs[1]).steps.len() == 0);
        }
    }

    #[test]
    fn test_steps() {
        let result = StepsParser::new().parse("init { a := 2; abc def := 3;} update {}");
        assert!(result.is_ok());
        if let Ok(steps) = result {
            assert_eq!(steps.len(), 2);
            assert!((*steps[0]).name == "init".to_string());
            assert!(matches!(*steps[0].statements[0], Stat::Assignment { .. }));
            assert!(matches!(*steps[0].statements[1], Stat::Declaration { .. }));
            assert!(*steps[1].name == "update".to_string());
            assert!((*steps[1]).statements.len() == 0);
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
                        (15, 30),
                    )),
                    Box::new(Stat::Declaration(
                        Type::StringType,
                        "b".to_string(),
                        Box::new(Exp::Lit(Literal::StringLit("a".to_string()))),
                        (31, 47),
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
}
