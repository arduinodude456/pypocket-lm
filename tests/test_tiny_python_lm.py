import json
import tempfile
import unittest
from pathlib import Path

from tiny_python_lm import TinyPythonLM, detokenize, tokenize_python


class TinyPythonLMTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            'def greet(name):\n    print("Hello", name)\n',
            'def add(a, b):\n    return a + b\n',
        ]
        self.model = TinyPythonLM().fit(self.examples, epochs=2)

    def test_tokenizer_keeps_python_structure(self):
        tokens = tokenize_python(self.examples[0])
        self.assertIn("KW:def", tokens)
        self.assertIn("<INDENT>", tokens)
        self.assertIn("OP:(", tokens)
        self.assertIn("LIT:<str>", tokens)

    def test_model_generates_known_continuation(self):
        generated = self.model.generate("def greet(name):\n    ", max_tokens=12)
        self.assertTrue(generated)
        self.assertIn("FN:print", generated)

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = TinyPythonLM.load(path)
            self.assertEqual(self.model.to_dict(), loaded.to_dict())
            json.loads(path.read_text())

    def test_detokenize_is_readable(self):
        rendered = detokenize(["KW:def", "NAME:<id>", "OP:(", "NAME:<id>", "OP:)", "OP::", "<NEWLINE>"])
        self.assertIn("def", rendered)
        self.assertIn("value", rendered)


if __name__ == "__main__":
    unittest.main()
