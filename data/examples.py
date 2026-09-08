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
