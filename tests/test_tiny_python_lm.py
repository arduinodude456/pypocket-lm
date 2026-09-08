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
        self.prompts = [
            {"intent": "hello_world", "prompt": "write a hello world program", "template": "print(\"Hello, World!\")"},
            {"intent": "hello_world", "prompt": "schreibe ein hallo welt programm", "template": "print(\"Hallo Welt\")"},
            {"intent": "sum_list", "prompt": "sum a list of numbers", "template": "def sum_numbers(numbers):\n    return sum(numbers)"},
        ]
        self.model = TinyPythonLM().fit(self.examples, epochs=2).fit_prompts(self.prompts)

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

    def test_natural_language_hello_world_is_not_exact_prompt_memorization(self):
        result = self.model.generate_for_prompt("please create a simple hello world script in Python")
        self.assertEqual(result["intent"], "hello_world")
        self.assertEqual(result["code"], 'print("Hello, World!")')

    def test_german_variant_uses_learned_intent(self):
        result = self.model.generate_for_prompt("Kannst du mir ein Hallo Welt Beispiel schreiben?")
        self.assertEqual(result["intent"], "hello_world")
        self.assertEqual(result["code"], 'print("Hallo Welt")')

    def test_new_prompt_can_select_a_parameterized_pattern(self):
        result = self.model.generate_for_prompt("generate code that calculates the sum of a list")
        self.assertEqual(result["intent"], "sum_list")
        self.assertIn("return sum(numbers)", result["code"] or "")

    def test_unknown_prompt_has_code_fallback(self):
        result = self.model.generate_for_prompt("def greet(name):\n    ")
        self.assertIsNone(result["intent"])
        self.assertTrue(result["code"])

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
