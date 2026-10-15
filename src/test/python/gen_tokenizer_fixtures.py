#!/usr/bin/env python3
"""Generates src/test/resources/tokenizers/<name>.json: reference token ids (Hugging Face `tokenizers`) and
reference chat renders (the official Jinja chat_template via jinja2) for every tokenizer family the Java
Tokenizer / ChatTemplate support.

Usage:  python3 -m venv /tmp/tokv && /tmp/tokv/bin/pip install tokenizers jinja2
        /tmp/tokv/bin/python3 gen_tokenizer_fixtures.py /tmp/jllm-tokenizers <out dir> [random string count]

<tokenizer dir>/<name>/ holds tokenizer.json, tokenizer_config.json (downloaded from the Hugging Face repos
listed in docs/04-tokenizer.md); no model weights are needed.
"""
import json
import os
import random
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

NAMES = ["qwen3", "qwen2", "llama3", "mistral", "gemma3", "gemma2", "phi3", "phi4", "glmedge", "glm4", "dsr1", "mamba"]

HAND = [
    "", " ", "  ", "   ", "\n", "\n\n", "\n\n\n", "\t", " \n ", "Hello world", "Hello, world!", " leading space",
    "trailing space ", "multiple   spaces   inside", "  two leading", "tabs\tand\t tabs", "line1\nline2\nline3",
    "trailing newline\n", "\n\nleading newlines", "windows\r\nline endings",
    "The quick brown fox jumps over the lazy dog.", "1234567890", "3.14159", "1,000,000", "12345678901234567890",
    "x = 42; y = 0xFF",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n",
    "public static void main(String[] args) {\n    System.out.println(\"hi\");\n}",
    "SELECT * FROM users WHERE id = 1;", "<html><body><p>Hi</p></body></html>",
    "https://example.com/path?q=1&r=2#frag", "foo@example.com", "I'm don't we've you'll they'd it's",
    "naïve café résumé", "Привет, мир!", "こんにちは世界", "你好，世界！", "안녕하세요", "مرحبا بالعالم", "שלום עולם",
    "สวัสดีชาวโลก", "😀😃😄", "Rocket 🚀 launch", "👨‍👩‍👧‍👦 family", "🇯🇵 flag", "é combining",
    "ｆｕｌｌｗｉｄｔｈ", "ﬁ ligature", "€100 ≈ $110", "a b nbsp", "zero​width", "\x00 nul", "\x01\x02",
    "tab\x0bvt", "ALL CAPS TEXT", "camelCaseIdentifier snake_case_name", "[1, 2, 3]", "{\"key\": \"value\", \"n\": 1}",
    "a-b_c.d/e\\f", "...", "!!!???", "-- comment", "# heading\n\n* bullet\n* bullet",
    "special <|endoftext|> inside", "<|im_start|>user", "<s>text</s>", "[INST] hi [/INST]",
    "<start_of_turn>user\nhi<end_of_turn>\n", "<|user|>\nhi<|end|>\n", "<|begin_of_text|>hello", "[gMASK]<sop>hi",
    "<unk>", "x" * 300, " " * 50, "ab" * 100, "Hello 你好 world 🌍 123",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
    "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "  indented\n    more\n\ttabbed", "end with spaces   ", "a  b   c    d", "price: $5.99 (50% off!)",
    "<|end|> <|user|> <|assistant|>", "<｜User｜>hi<｜Assistant｜>", "<bos><eos><pad>", "</s><unk>",
]

POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") * 3 + list(" \n\t\r") * 4 + list(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + list("éñüßçøåÆŒΩπДжЖ日本語中文한국어العربيةעבריתไทय—€£¥«»…") + [
    "😀", "🚀", "👍", "́", "​", " ", "\x01", "\x7f", "ǅ", "ß", "İ"]


def random_strings(n, seed):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append("".join(rnd.choice(POOL) for _ in range(rnd.randint(1, 60))))
    return out


CONVS = {
    "single": [{"role": "user", "content": "Hello, who are you?"}],
    "system": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And times 3?"},
    ],
    "nosystem": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "  The answer is 4.\n"},
        {"role": "user", "content": "Thanks!  "},
    ],
    "unicode": [
        {"role": "user", "content": "你好 😀 café"},
        {"role": "assistant", "content": "Ça va très bien, merci"},
        {"role": "user", "content": "Écris 3 lignes\nde code:\n    print(1)"},
    ],
    "leadingws": [
        {"role": "user", "content": " leading space\nand newline"},
        {"role": "assistant", "content": "\n\nreply"},
        {"role": "user", "content": "  x"},
    ],
    "sysonly_user": [
        {"role": "system", "content": "  Be brief.  \n"},
        {"role": "user", "content": "Hi"},
    ],
}


def cfg_token(cfg, key):
    v = cfg.get(key)
    if isinstance(v, dict):
        return v.get("content")
    return v


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env


def id_sequences(tk, n, seed):
    """Random id sequences (decode-only cases). Byte-fallback tokens above 0x7F are left out: the reference
    decoder turns lone high bytes into one U+FFFD each, which no consumer relies on."""
    import re
    rnd = random.Random(seed)
    size = tk.get_vocab_size(with_added_tokens=True)
    out = []
    while len(out) < n:
        seq = []
        for _ in range(rnd.randint(1, 12)):
            for _try in range(50):
                i = rnd.randrange(size)
                t = tk.id_to_token(i)
                if t is not None and not re.fullmatch(r"<0x[89A-F][0-9A-F]>", t):
                    seq.append(i)
                    break
        out.append(seq)
    return out


def main(src, dst, n_random=60):
    os.makedirs(dst, exist_ok=True)
    env = make_env()
    rnd_strings = random_strings(n_random, 20240601)
    for name in NAMES:
        d = os.path.join(src, name)
        tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        cfg = json.load(open(os.path.join(d, "tokenizer_config.json")))
        fixture = {"name": name, "cases": [], "chat": []}
        for text in HAND + rnd_strings:
            ids = tk.encode(text, add_special_tokens=False).ids
            dec = tk.decode(ids, skip_special_tokens=False)
            fixture["cases"].append({"text": text, "ids": ids, "decoded": dec})
        fixture["idseqs"] = [{"ids": q, "decoded": tk.decode(q, skip_special_tokens=False)}
                             for q in id_sequences(tk, 40, 7)]
        template = cfg.get("chat_template")
        if isinstance(template, list):
            template = next(t["template"] for t in template if t.get("name") == "default")
        if template:
            tpl = env.from_string(template)
            think_variants = [None]
            if name == "qwen3":
                think_variants = [True, False]
            for cname, msgs in CONVS.items():
                for gen in (True, False):
                    for think in think_variants:
                        kw = dict(messages=msgs, add_generation_prompt=gen, bos_token=cfg_token(cfg, "bos_token"),
                                  eos_token=cfg_token(cfg, "eos_token"))
                        if think is not None:
                            kw["enable_thinking"] = think
                        try:
                            rendered = tpl.render(**kw)
                        except jinja2.exceptions.TemplateError as e:
                            print(f"{name}/{cname}: template refused ({e}); skipped")
                            continue
                        ids = tk.encode(rendered, add_special_tokens=False).ids
                        fixture["chat"].append({"conv": cname, "messages": msgs, "gen": gen, "think": think,
                                                "rendered": rendered, "ids": ids})
        with open(os.path.join(dst, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
        print(name, len(fixture["cases"]), "cases", len(fixture["chat"]), "chat renders")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
