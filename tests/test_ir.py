import ast
import unittest

from data.ir_tasks import build_tasks
from ir import compile_ir, tokenize_ir, validate_ir
from ir_model import IRLanguageModel


class IRTests(unittest.TestCase):
    def test_compiler_turns_ir_into_valid_python(self):
        tokens = tokenize_ir("SET VAR:x NUM:7\nMUL VAR:x NUM:4\nPRINT VAR:x")
        valid, reason = validate_ir(tokens)
        self.assertTrue(valid, reason)
        code = compile_ir(tokens)
        ast.parse(code)
        self.assertIn("x = x * 4", code)
        self.assertIn("print(x)", code)

    def test_invalid_ir_is_rejected(self):
        valid, reason = validate_ir(["<BOS>", "PRINT", "BROKEN", "<END>"])
        self.assertFalse(valid)
        self.assertTrue(reason)

    def test_autoregressive_model_has_unseen_validation_tasks(self):
        tasks = build_tasks(seed=3, count=50)
        train, novel = tasks[:40], tasks[40:]
        model = IRLanguageModel(order=4).fit(train, epochs=2)
        self.assertNotIn(novel[0]["prompt"], [item["prompt"] for item in train])
        generated = model.generate_ir(novel[0]["prompt"], temperature=0)
        self.assertTrue(generated)
        self.assertGreater(len(generated), 1)
        valid, reason = validate_ir(generated)
        self.assertTrue(valid, reason)
        ast.parse(compile_ir(generated))
        self.assertTrue(model.cross_entropy(novel) >= 0)


if __name__ == "__main__":
    unittest.main()
