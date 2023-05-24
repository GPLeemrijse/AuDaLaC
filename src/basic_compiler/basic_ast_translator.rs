use crate::parser::ast::Exp;
use crate::parser::ast::Literal;
use crate::parser::ast::Program;
use crate::parser::ast::Type;

pub fn as_c_type(t: &Type) -> String {
    use Type::*;
    match t {
        Named(s) => format!("{s}*").to_string(),
        String => "string".to_string(),
        Nat => "unsigned int".to_string(),
        Int => "int".to_string(),
        Bool => "bool".to_string(),
        Null => todo!(),
    }
}

pub fn as_printf(t: &Type) -> String {
    use Type::*;
    match t {
        Named(_) => "%p",
        String => "%s",
        Nat => "%u",
        Int => "%d",
        Bool => "%u",
        Null => todo!(),
    }
    .to_string()
}

pub fn as_c_default(t: &Type) -> String {
    use Type::*;
    match t {
        Named(_) => "NULL",
        String => "\"\"",
        Nat => "0",
        Int => "0",
        Bool => "false",
        Null => todo!(),
    }
    .to_string()
}

pub fn as_c_literal(l: &Literal, type_name: Option<&String>) -> String {
    use Literal::*;
    match l {
        NatLit(n) => format!("{}", n),
        IntLit(i) => format!("{}", i),
        BoolLit(b) => format!("{}", if *b { "true" } else { "false" }),
        StringLit(s) => format!("\"{}\"", s),
        NullLit => {
            if let Some(name) = type_name {
                format!("({}*){}_manager.structs", name, name)
            } else {
                panic!("Need type information to express this in c.");
            }
        }
        ThisLit => "self".to_string(),
    }
}

pub fn as_c_expression<F: Fn(&String) -> bool>(
    e: &Exp,
    program: &Program,
    is_param: &F,
    type_name: Option<&String>,
) -> String {
    use Exp::*;
    println!("{:?}", e);
    match e {
        BinOp(e1, c, e2, _) => {
            let e1_comp = as_c_expression(e1, program, is_param, type_name);
            let e2_comp = as_c_expression(e2, program, is_param, type_name);

            format!("({e1_comp} {c} {e2_comp})")
        }
        UnOp(c, e, _) => {
            let e_comp = as_c_expression(e, program, is_param, type_name);
            format!("({c}{e_comp})")
        }
        Constructor(n, args, _) => {
            let n_struct = program.structs.iter().find(|s| s.name == *n).unwrap();
            let arg_types: Vec<Option<&String>> = n_struct
                .parameters
                .iter()
                .map(|(_, t, _)| {
                    if let Type::Named(t_name) = t {
                        Some(t_name)
                    } else {
                        None
                    }
                })
                .collect();
            let args_comp = args
                .iter()
                .enumerate()
                .map(|(idx, e)| as_c_expression(e, program, is_param, arg_types[idx]))
                .reduce(|acc: String, nxt| acc + ", " + &nxt)
                .unwrap();

            format!("create_{n}(&{n}_manager, {args_comp})")
        }
        Var(parts, _) => {
            let is_p = if is_param(&parts[0]) { "self->" } else { "" };
            let p = parts.join("->");

            format!("({is_p}{p})")
        }
        Lit(l, _) => as_c_literal(l, type_name),
    }
}
