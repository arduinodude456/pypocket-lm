TRAINING_EXAMPLES = [
    '''def greet(name):\n    print("Hello", name)\n''',
    '''def add(a, b):\n    return a + b\n''',
    '''def is_even(number):\n    return number % 2 == 0\n''',
    '''for item in items:\n    print(item)\n''',
    '''total = 0\nfor value in values:\n    total = total + value\nprint(total)\n''',
    '''if score >= 10:\n    print("passed")\nelse:\n    print("try again")\n''',
    '''def find_user(users, name):\n    for user in users:\n        if user == name:\n            return user\n    return None\n''',
    '''class Counter:\n    def __init__(self):\n        self.value = 0\n\n    def increment(self):\n        self.value += 1\n''',
    '''try:\n    result = int(value)\nexcept ValueError:\n    result = 0\n''',
    '''def make_message(name):\n    return f"Hello {name}"\n''',
]

# Natural-language supervision. The model learns keyword patterns, not exact prompt strings.
PROMPT_EXAMPLES = [
    {"intent": "hello_world", "prompt": "write a hello world program in Python", "template": "print(\"Hello, World!\")"},
    {"intent": "hello_world", "prompt": "schreibe ein Hallo Welt Programm", "template": "print(\"Hallo Welt\")"},
    {"intent": "hello_world", "prompt": "make Python output hello world", "template": "print(\"Hello, World!\")"},
    {"intent": "greeting_function", "prompt": "create a Python function that greets a person", "template": "def greet(name):\n    print(f\"Hello, {name}!\")"},
    {"intent": "greeting_function", "prompt": "baue eine Funktion die einen Namen begrüßt", "template": "def begruessen(name):\n    print(f\"Hallo, {name}!\")"},
    {"intent": "sum_list", "prompt": "write Python code to sum a list of numbers", "template": "def sum_numbers(numbers):\n    return sum(numbers)"},
    {"intent": "sum_list", "prompt": "berechne die Summe einer Liste", "template": "def summe(werte):\n    return sum(werte)"},
    {"intent": "range_loop", "prompt": "loop over numbers from one to ten in Python", "template": "for number in range(1, 11):\n    print(number)"},
    {"intent": "range_loop", "prompt": "schreibe eine Schleife über Zahlen", "template": "for nummer in range(1, 11):\n    print(nummer)"},
    {"intent": "fibonacci", "prompt": "create a Python Fibonacci function", "template": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    {"intent": "fibonacci", "prompt": "erzeuge die Fibonacci Folge in Python", "template": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
]
