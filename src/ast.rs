pub type Loc = (usize, usize);

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;

#[derive(Eq, PartialEq, Debug)]
pub struct Program {
    pub inline_global_cpp: Vec<String>,
    pub structs: Vec<ADLStruct>,
    pub schedule: Box<Schedule>,
}

impl Program {
    pub fn struct_by_name(&self, name: &String) -> Option<&ADLStruct> {
        self.structs.iter().find(|s| &s.name == name)
    }

    pub fn step_by_name(&self, struct_name: &String, step_name: &String) -> Option<&Step> {
        self.structs
            .iter()
            .find(|s| &s.name == struct_name)?
            .step_by_name(step_name)
    }

    pub fn has_any_step_by_name(&self, step_name: &String) -> bool {
        self.structs
            .iter()
            .any(|strct| strct.step_by_name(step_name).is_some())
    }

    pub fn get_step_to_structs(&self) -> HashMap<&String, Vec<&ADLStruct>> {
        let mut s2s: HashMap<&String, Vec<&ADLStruct>> = HashMap::new();

        for strct in &self.structs {
            for step in &strct.steps {
                s2s.entry(&step.name)
                    .and_modify(|v| v.push(strct))
                    .or_insert(vec![strct]);
            }
        }
        return s2s;
    }

    pub fn executors(&self) -> HashSet<&String> {
        let s2s = self.get_step_to_structs();
        let mut calls = Vec::new();
        self.schedule.step_calls(&mut calls);

        let mut result = HashSet::new();
        for c in calls {
            match c {
                (None, step) => s2s.get(step).unwrap().iter().for_each(|strct| {
                    result.insert(&strct.name);
                }),
                (Some(strct), _) => {
                    result.insert(strct);
                }
            }
        }
        result
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum Schedule {
    StepCall(String, Loc),
    TypedStepCall(String, String, Loc),
    Sequential(Box<Schedule>, Box<Schedule>, Loc),
    Fixpoint(Box<Schedule>, Loc),
}

impl Schedule {
    pub fn fixpoint_depth(&self) -> usize {
        use crate::ast::Schedule::*;
        match self {
            Sequential(s1, s2, _) => std::cmp::max(s1.fixpoint_depth(), s2.fixpoint_depth()),
            Fixpoint(s, _) => s.fixpoint_depth() + 1,
            _ => 0,
        }
    }

    pub fn is_fixpoint(&self) -> bool {
        return matches!(self, crate::ast::Schedule::Fixpoint(..));
    }

    /*  Unfolds Sequential steps and returns the first non-Sequential (sub)schedule.
        Does return itself if non-Sequential.
    */
    pub fn earliest_subschedule(&self) -> &Schedule {
        use crate::ast::Schedule::*;
        match self {
            Sequential(s, _, _) => s.earliest_subschedule(),
            StepCall(..) | TypedStepCall(..) | Fixpoint(..) => &self,
        }
    }

    pub fn step_calls<'a>(&'a self, result: &mut Vec<(Option<&'a String>, &'a String)>) {
        use crate::ast::Schedule::*;

        match self {
            StepCall(step, _) => result.push((None, step)),
            TypedStepCall(strct, step, _) => result.push((Some(strct), step)),
            Sequential(s1, s2, _) => {
                s1.step_calls(result);
                s2.step_calls(result);
            }
            Fixpoint(s, _) => s.step_calls(result),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Step {
    pub name: String,
    pub statements: Vec<Stat>,
    pub loc: Loc,
}

fn remove_duplicates<T: Ord>(vec: &mut Vec<T>) {
    vec.sort();
    vec.dedup();
}

impl Step {
    /* Performs 'f_expr' on all expressions and 'f_stmt' on all statements.
       Collects all unique results.
    */
    pub fn visit<'a, T: 'a + Ord>(
        &'a self,
        f_stmt: fn(&'a Stat) -> Vec<T>,
        f_exp: fn(&'a Exp) -> Vec<T>,
    ) -> Vec<T> {
        let mut results = self
            .statements
            .iter()
            .map(|s| s.visit(f_stmt, f_exp))
            .flatten()
            .collect::<Vec<T>>();

        remove_duplicates(&mut results);
        return results;
    }

    // Returns a unique vector of possibly constructed struct types
    pub fn constructors(&self) -> Vec<&String> {
        use Exp::*;
        self.visit::<&String>(
            |_| Vec::new(),
            |e| {
                if let Constructor(s, _, _) = e {
                    vec![&s]
                } else {
                    Vec::new()
                }
            },
        )
    }

    pub fn declarations(&self) -> Vec<(&String, &Type)> {
        use Stat::*;
        self.visit::<(&String, &Type)>(
            |stat| {
                if let Declaration(t, s, _, _) = stat {
                    vec![(&s, &t)]
                } else {
                    Vec::new()
                }
            },
            |_| Vec::new(),
        )
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct ADLStruct {
    pub name: String,
    pub parameters: Vec<(String, Type, Loc)>,
    pub steps: Vec<Step>,
    pub loc: Loc,
}

impl ADLStruct {
    pub fn step_by_name(&self, name: &String) -> Option<&Step> {
        self.steps.iter().find(|s| &s.name == name)
    }

    pub fn parameter_by_name(&self, name: &String) -> Option<&(String, Type, Loc)> {
        self.parameters.iter().find(|p| &p.0 == name)
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Exp {
    BinOp(Box<Exp>, BinOpcode, Box<Exp>, Loc),
    UnOp(UnOpcode, Box<Exp>, Loc),
    Constructor(String, Vec<Exp>, Loc),
    Var(Vec<String>, Loc),
    Lit(Literal, Loc),
}

impl Exp {
    pub fn visit<'a, T: 'a + Ord>(&'a self, f_exp: fn(&'a Exp) -> Vec<T>) -> Vec<T> {
        use Exp::*;
        let mut result = f_exp(self);
        match self {
            BinOp(e1, _, e2, _) => {
                result.append(&mut e1.visit(f_exp));
                result.append(&mut e2.visit(f_exp));
            }
            UnOp(_, e, _) => {
                result.append(&mut e.visit(f_exp));
            }
            Constructor(_, exps, _) => {
                result.append(&mut exps.iter().map(|e| e.visit(f_exp)).flatten().collect());
            }
            _ => (),
        }
        result
    }

    pub fn get_parts(&self) -> &Vec<String> {
        match self {
            Exp::Var(parts, _) => parts,
            _ => panic!(),
        }
    }
}

impl Display for Exp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::ast::Exp::*;
        match self {
            BinOp(l, o, r, _) => write!(f, "{l} {o} {r}"),
            UnOp(o, e, _) => write!(f, "{o}{e}"),
            Constructor(n, exps, _) => write!(
                f,
                "{n}({})",
                exps.iter()
                    .map(|e| format!("{e}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Var(parts, _) => write!(f, "{}", parts.join(".")),
            Lit(l, _) => write!(f, "{l}"),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Stat {
    IfThen(Box<Exp>, Vec<Stat>, Vec<Stat>, Loc),
    Declaration(Type, String, Box<Exp>, Loc),
    Assignment(Box<Exp>, Box<Exp>, Loc),
    InlineCpp(String),
}

impl Stat {
    pub fn visit<'a, T: 'a + Ord>(
        &'a self,
        f_stmt: fn(&'a Stat) -> Vec<T>,
        f_exp: fn(&'a Exp) -> Vec<T>,
    ) -> Vec<T> {
        use Stat::*;
        let mut result = f_stmt(self);

        match self {
            IfThen(cond, stmts1, strmts2, _) => {
                result.append(
                    &mut stmts1
                        .iter()
                        .chain(strmts2.iter())
                        .map(|s| s.visit(f_stmt, f_exp))
                        .flatten()
                        .chain(cond.visit(f_exp))
                        .collect::<Vec<T>>(),
                );
            }
            Declaration(_, _, e, _) | Assignment(_, e, _) => {
                result.append(&mut e.visit(f_exp));
            }
            _ => (),
        }
        result
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Ord, PartialOrd)]
pub enum Type {
    Named(String),
    String,
    Nat,
    Int,
    Bool,
    Null,
}

impl Type {
    pub fn can_be_coerced_to_type(&self, t: &Type) -> bool {
        // Types match
        self == t ||
        // Nat can be changed to Int
        (*self == Type::Nat && *t == Type::Int) ||
        // NullType can be changed to named
        (*self == Type::Null && matches!(*t, Type::Named(..)))
    }

    pub fn name(&self) -> Option<&String> {
        match self {
            Type::Named(s) => Some(s),
            _ => None,
        }
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
            Null => write!(f, "NullType"),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Literal {
    NatLit(u32),
    IntLit(i32),
    BoolLit(bool),
    StringLit(String),
    NullLit,
    ThisLit,
}

impl Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::ast::Literal::*;
        match self {
            NatLit(v) => write!(f, "{v}"),
            IntLit(v) => write!(f, "{v}"),
            BoolLit(v) => write!(f, "{v}"),
            StringLit(v) => write!(f, "\"{v}\""),
            NullLit => write!(f, "null"),
            ThisLit => write!(f, "this"),
        }
    }
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

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum UnOpcode {
    Negation,
}

impl Display for UnOpcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::ast::UnOpcode::*;
        match self {
            Negation => write!(f, "!"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::adl::ExpParser;
    use crate::adl::ScheduleParser;
    use crate::adl::StatParser;
    use crate::adl::StepsParser;
    use crate::adl::StructsParser;
    use crate::ast::*;
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
            assert!(structs[0].name == *"ABC");
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
            assert!(structs[0].steps[0].name == *"step1");
            assert!(structs[1].name == *"DEF");
            assert!(structs[1].parameters.is_empty());
            assert!(structs[1].steps.is_empty());
        }
    }

    #[test]
    fn test_steps() {
        let result = StepsParser::new().parse("init { a := 2; abc def := 3;} update {}");
        assert!(result.is_ok());
        if let Ok(steps) = result {
            assert_eq!(steps.len(), 2);
            assert!(steps[0].name == *"init");
            assert!(matches!(steps[0].statements[0], Stat::Assignment { .. }));
            assert!(matches!(steps[0].statements[1], Stat::Declaration { .. }));
            assert!(steps[1].name == *"update");
            assert!(steps[1].statements.is_empty());
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
                        Box::new(Exp::Var(
                            vec!["id".to_string(), "id2".to_string()],
                            (15, 21),
                        )),
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
                vec![],
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
