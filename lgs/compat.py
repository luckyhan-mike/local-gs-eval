"""Prepare an experiment-local Python 3.9 copy of AnySplat; never edit upstream."""
import ast
import shutil
import sys
from pathlib import Path


class TypeUnions(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.BitOr):
            return ast.copy_location(ast.Subscript(
                value=ast.Attribute(value=ast.Call(func=ast.Name(id="__import__", ctx=ast.Load()),
                    args=[ast.Constant("typing")], keywords=[]), attr="Union", ctx=ast.Load()),
                slice=ast.Tuple(elts=[node.left, node.right], ctx=ast.Load()), ctx=ast.Load()), node)
        return node


class Annotations(ast.NodeTransformer):
    def visit_arg(self, node):
        if node.annotation:
            node.annotation = TypeUnions().visit(node.annotation)
        return node

    def visit_AnnAssign(self, node):
        node.annotation = TypeUnions().visit(node.annotation)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.returns:
            node.returns = TypeUnions().visit(node.returns)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        aliases = {"AnyExample", "AnyViews", "DatasetCfgWrapper", "LossCfgWrapper"}
        if any(isinstance(t, ast.Name) and (t.id in aliases or t.id.endswith("Cfg")) for t in node.targets):
            node.value = TypeUnions().visit(node.value)
        return node


def anysplat_source(source, work):
    if sys.version_info >= (3, 10):
        return source
    dest = Path(work) / "anysplat_py39"
    shutil.copytree(Path(source) / "src", dest / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in (dest / "src").rglob("*.py"):
        tree = Annotations().visit(ast.parse(path.read_text(encoding="utf-8")))
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
    print(f"Python 3.9 annotation compatibility copy: {dest}")
    return str(dest)
