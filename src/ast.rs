use std::fmt::{Debug, Error, Formatter};


#[derive(Eq, PartialEq)]
pub enum Exp {
	BinOp(Box<Exp>, BinOpcode, Box<Exp>),
	UnOp(UnOpcode, Box<Exp>),
	Constructor(String, Vec<Exp>),
	Lit(Literal),
}


#[derive(Eq, PartialEq)]
pub enum Literal {
	NumLit(i64),
	BoolLit(bool),
	StringLit(String),
	NullLit,
	ThisLit
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
	Or
}

#[derive(Eq, PartialEq)]
pub enum UnOpcode {
	Negation
}



impl Debug for Exp {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Exp::*;
        match &*self {
            BinOp(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            UnOp(op, ref l) => write!(fmt, "({:?}{:?})", op, l),
            Constructor(s, v) => write!(fmt, "{:?}({:?})", s, v),
            Lit(l) => write!(fmt, "{:?}", l),
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
			Or => write!(fmt, "||")
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
            NumLit(n) => write!(fmt, "{:?}", n),
			BoolLit(b) => write!(fmt, "{:?}", b),
			StringLit(s) => write!(fmt, "{:?}", s),
			NullLit => write!(fmt, "null"),
			ThisLit => write!(fmt, "this"),
        }
    }
}
