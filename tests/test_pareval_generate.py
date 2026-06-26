from benchmaker.pareval.generate import extract_code


def test_extract_fenced_with_lang():
    reply = "Sure:\n```cpp\nint f() { return 1; }\n```\nDone."
    assert extract_code(reply) == "int f() { return 1; }"

def test_extract_fenced_no_lang():
    reply = "```\nint f() {}\n```"
    assert extract_code(reply) == "int f() {}"

def test_extract_first_of_multiple_blocks():
    reply = "```cpp\nA\n```\nthen\n```cpp\nB\n```"
    assert extract_code(reply) == "A"

def test_extract_unfenced_returns_stripped():
    reply = "   int f() {}\n"
    assert extract_code(reply) == "int f() {}"
