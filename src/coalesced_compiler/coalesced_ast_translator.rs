use crate::ast::Program;
use crate::ast::Type;
use crate::ast::Literal;
use crate::ast::Exp;

pub fn as_c_type(t : &Type) -> String {
    use Type::*;
    match t {
        Named(..) => "RefType".to_string(),
        String => "StringType".to_string(),
        Nat => "NatType".to_string(),
        Int => "IntType".to_string(),
        Bool => "BoolType".to_string(),
        Null => "RefType".to_string(),
    }
}

pub fn as_type_enum(t : &Type) -> String {
    use Type::*;
    match t {
        Named(..) => "Ref".to_string(),
        String => "String".to_string(),
        Nat => "Nat".to_string(),
        Int => "Int".to_string(),
        Bool => "Bool".to_string(),
        Null => "Ref".to_string(),
    }
}

pub fn as_printf(t : &Type) -> String {
    use Type::*;
    match t {
        Named(_) => "%u",
        String => "%s",
        Nat => "%u",
        Int => "%d",
        Bool => "%s",
        Null => "%u",
    }.to_string()
}

pub fn as_c_default(t : &Type) -> String {
    use Type::*;
    match t {
        Named(_) => "0",
        String => "\"\"",
        Nat => "0",
        Int => "0",
        Bool => "false",
        Null => "0",
    }.to_string()
}

pub fn as_c_literal(l : &Literal) -> String {
    use Literal::*;
    match l {
        NatLit(n) => format!("{}", n),
        IntLit(i) => format!("{}", i),
        BoolLit(b) => format!("{}", if *b {"true"} else {"false"}),
        StringLit(s) => format!("\"{}\"", s),
        NullLit => "0".to_string(),// index based!
        ThisLit => "self".to_string(),
    }
}

pub fn as_c_expression<F: Fn(&String) -> bool>(e : &Exp, program : &Program, is_param: &F, type_name : Option<&String>) -> String {
    use Exp::*;
    match e {
        BinOp(e1, c, e2, _) => {
            let e1_comp = as_c_expression(e1, program, is_param, type_name);
            let e2_comp = as_c_expression(e2, program, is_param, type_name);

            format!("({e1_comp} {c} {e2_comp})")
        },
        UnOp(c, e, _) => {
            let e_comp = as_c_expression(e, program, is_param, type_name);
            format!("({c}{e_comp})")
        },
        Constructor(n, args, _) => {
            let n_struct = program.structs.iter().find(|s| s.name == *n).unwrap();
            let arg_types : Vec<Option<&String>> = n_struct.parameters.iter()
                                                                      .map(|(_, t, _)|
                                                                            if let Type::Named(t_name) = t {
                                                                                Some(t_name)
                                                                            } else {
                                                                                None
                                                                            })
                                                                       .collect();
            let args_comp = args.iter()
                    .enumerate()
                    .map(|(idx, e)| as_c_expression(e, program, is_param, arg_types[idx]))
                    .reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();

            format!("create_{n}(&{n}_manager, {args_comp})")
        },
        Var(parts, _) => {
            let is_p = if is_param(&parts[0]) {"self->"} else {""};
            let p = parts.join("->");

            format!("({is_p}{p})")
        },
        Lit(l, _) => as_c_literal(l),
    }
}