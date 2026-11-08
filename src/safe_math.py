"""Safely evaluate basic arithmetic without Python eval(), exec(), or shell commands."""
import ast
import operator

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}
LIMIT = 1e12


def calculate(expression: str) -> float | int:
    if not expression or len(expression) > 150:
        raise ValueError('Ekspresi kosong atau terlalu panjang.')
    tree = ast.parse(expression, mode='eval')

    def walk(node, depth=0):
        if depth > 20:
            raise ValueError('Ekspresi terlalu kompleks.')
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = walk(node.left, depth+1), walk(node.right, depth+1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1e6):
                raise ValueError('Eksponen/basis terlalu besar.')
            value = OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            value = OPERATORS[type(node.op)](walk(node.operand, depth+1))
        else:
            raise ValueError('Hanya aritmetika sederhana (+,-,*,/,//,%,**) yang diizinkan.')
        if isinstance(value, complex) or abs(value) > LIMIT:
            raise ValueError('Hasil di luar batas aman.')
        return value

    return walk(tree.body)
