# 4. The tokenizer: byte-level BPE from scratch

Code: `tokenizer/Tokenizer.java`, `tokenizer/ChatTemplate.java`, `tokenizer/StreamDecoder.java`

## 4.1 Why tokens at all?

A neural network consumes numbers. The model's first operation is a table lookup — "give me the vector for
token number 9707" — so text must first become a list of integers. There are three obvious ways to do this and
each is bad in its own way:

| Unit | Vocabulary | Problem |
|---|---|---|
| characters / bytes | 256 | sequences are very long (one step per letter); the model must learn spelling before meaning |
| words | millions | rare words, typos, other languages, code identifiers → "unknown"; the output layer becomes enormous |
| **subwords (BPE)** | ~150 000 | **the compromise**: common words are a single token, rare words become a few pieces, anything at all can still be expressed because single bytes are always in the vocabulary |

Byte-pair encoding learns its vocabulary from data by repeatedly merging the most frequent adjacent pair of
symbols. We do not train here — the learned vocabulary and merge rules come in `tokenizer.json` — but the
*encoding* side (applying the rules to new text) is what `Tokenizer.java` implements.

Some real encodings with Qwen's vocabulary:

```
"Hello, world!"         →  [9707, 11, 1879, 0]          "Hello"  ","  " world"  "!"
"unbelievable"          →  [359, 31798]                   "un"  "believable"
"1234"                  →  [16, 17, 18, 19]               "1" "2" "3" "4"   (digits are always split)
"naïve café"            →  [12572, 115, 586, 18781]       "na" + 2 bytes of "ï" ... (bytes of multi-byte chars)
"<|im_start|>"          →  [151644]                       one special token, never split
```

Note the leading space in `" world"`: in GPT-style tokenizers the space *belongs to the following word*. That
is why `"world"` and `" world"` are different tokens, and why decoding is a straight concatenation with no
spaces inserted.

## 4.2 What is inside `tokenizer.json`

The file (11 MB) describes a four-stage pipeline. The parts we use:

```json
{
  "normalizer":    {"type": "NFC"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated"},
                      {"type": "ByteLevel", "add_prefix_space": false}]},
  "model": {"type": "BPE", "ignore_merges": true,
            "vocab":  {"!": 0, "\"": 1, "#": 2, ..., "Ġt": 259, "Ġa": 264, ..., "Ġworld": 1879, ...},
            "merges": [["Ġ", "t"], ["Ġ", "a"], ["i", "n"], ["h", "e"], ...]},
  "added_tokens": [{"id": 151643, "content": "<|endoftext|>", "special": true},
                   {"id": 151644, "content": "<|im_start|>",  "special": true},
                   {"id": 151645, "content": "<|im_end|>",    "special": true}, ...],
  "decoder": {"type": "ByteLevel"}
}
```

- **vocab** — 151 643 entries, token string → id. The strings look odd (`Ġworld`) because of the byte-level
  alphabet explained next.
- **merges** — 151 387 ordered pairs. Position in the list is the pair's **rank**: earlier = merge first.
  `["Ġ","t"]` means "when the symbols `Ġ` and `t` are adjacent, fuse them into `Ġt`".
- **added_tokens** — control tokens that are matched literally before anything else, so `<|im_start|>` can
  never be split into `<`, `|`, `im`, … The `special` ones are the chat delimiters and end-of-text.
- The 26 remaining ids (151 643–151 668) are these added tokens; ids 151 669–151 935 exist only as padding
  rows in the model's embedding matrix and are never produced ([09](09-sampling.md)).

## 4.3 Stage 0 — the byte-level alphabet

GPT-2-style tokenizers operate on **UTF-8 bytes**, not on characters. Advantage: every possible input,
including emoji, binary junk or languages the vocabulary never saw, is representable — in the worst case as one
token per byte. But the vocabulary must be stored as JSON *strings*, and many byte values (0x00, the space 0x20,
0x7F, 0x80–0xFF) are unprintable or invalid on their own. So each of the 256 byte values is mapped to a
printable Unicode character:

```
bytes '!'..'~' (0x21–0x7E), 0xA1–0xAC, 0xAE–0xFF   →  themselves            (188 "nice" bytes)
the other 68 bytes (0x00–0x20, 0x7F–0xA0, 0xAD)       →  U+0100, U+0101, ...  in increasing order
```

Hence a space (0x20) becomes `Ġ` (U+0120 = 0x100 + 0x20), a newline (0x0A) becomes `Ċ` (U+010A). The strange
`Ġworld` is simply `" world"`. `initByteMapping()` builds the two tables `byteToChar[256]` and `charToByte[]`;
`decodeByteLevel()` turns each vocab string back into the real `byte[]` once, at load time, into `idToBytes` —
so decoding at runtime never touches the alphabet again.

## 4.4 Stage 1 — normalization

`NFC` (Unicode Normalization Form C) composes characters: `e` + U+0301 (combining acute) becomes the single
code point `é`. Without it, visually identical text could tokenize differently. `java.text.Normalizer` does it
in one line; `containsType()` discovers from the JSON whether the tokenizer wants it.

## 4.5 Stage 2 — pre-tokenization (the regex)

BPE must never merge across word boundaries ("of the" must not become one token, or the vocabulary would fill
with phrases). So before BPE the text is cut into chunks by a regular expression, and BPE runs *inside* each
chunk. Qwen's pattern, piece by piece:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)        English contractions as their own chunk      "don't" → "don" "'t"
| [^\r\n\p{L}\p{N}]?\p{L}+           an optional leading symbol + a run of letters  " world", "Hello", "(foo"
| \p{N}                              a single digit                               "1234" → "1" "2" "3" "4"
|  ?[^\s\p{L}\p{N}]+[\r\n]*          punctuation/symbol runs, optional leading space, trailing newlines
| \s*[\r\n]+                         newline runs (with preceding whitespace)
| \s+(?!\S)                          whitespace that is NOT followed by non-space (keeps one space for the next word)
| \s+                                any other whitespace
```

Digits one-per-token is a deliberate Qwen choice that makes arithmetic more regular for the model.

`java.util.regex` with `Pattern.UNICODE_CHARACTER_CLASS` interprets `\p{L}` (any letter in any script) and
`\p{N}` (any numeral) the same way the Rust `tokenizers` crate does, which is why our ids match Hugging Face
exactly. `encodeOrdinary()` runs the matcher; the "Isolated" behaviour means unmatched gaps are kept too.

## 4.6 Stage 3 — applying the merges (the actual BPE)

Take one chunk, `" world"`. Its UTF-8 bytes are `20 77 6F 72 6C 64`, which the alphabet turns into the symbols
`Ġ w o r l d`. Each symbol is a vocabulary entry (all 256 single bytes are), so we start with six token ids and
repeatedly merge:

```
step  symbols                best adjacent pair (lowest rank)      result
 0    Ġ  w  o  r  l  d       "or" → "or"   (rank 20)              Ġ  w  or  l  d
 1    Ġ  w  or  l  d         "Ġw" → "Ġw"  (rank 112)             Ġw  or  l  d
 2    Ġw  or  l  d           "ld" → "ld"  (rank 406)             Ġw  or  ld
 3    Ġw  or  ld             "Ġwor"       (rank 1 522)           Ġwor  ld
 4    Ġwor  ld               "Ġworld"     (rank 1 673)           Ġworld            ← id 1879
 5    Ġworld                 no mergeable pair                    done
```

(Ranks are illustrative; the order is what matters.) Conceptually the algorithm is:

```
symbols = ids of the single-byte tokens
loop:
    among all adjacent pairs, take the one with the smallest rank (leftmost on ties)
    if no pair is in the merge table → stop
    replace that pair with the merged token id
```

Done naively — rescan every pair after every merge — this is O(n²) in the chunk length, and the pre-tokenizer
can produce arbitrarily long chunks (a run of 1 MB of spaces is one chunk), which an API client could exploit.
`bpeUncached()` therefore uses the standard O(n log n) formulation: the symbols live in a **doubly linked list**
over their original positions, and every mergeable adjacent pair is a candidate in a **min-heap** keyed by
`(rank, position)`. Each step pops the best candidate, checks that the pair is still present at that position
(entries go stale when a neighbour was merged — ranks are unique per pair, so a matching rank means a matching
pair), links the merged symbol in, and pushes the two new pairs formed with its neighbours. The result is
identical to the naive loop (the smoke test proves this against a verbatim copy of the old implementation on
every vocabulary string, thousands of random strings and 20 000-character runs, which now encode in
milliseconds).

Three more implementation details keep this fast:

1. **No strings.** Symbols are token ids; the merge table is `HashMap<Long, Long>` keyed by
   `(leftId << 32) | rightId` with value `(rank << 32) | mergedId`. One hash lookup per pair, no concatenation.
2. **`ignore_merges`** (true for Qwen): if the whole chunk is already a vocabulary entry, return it directly.
   Common words skip the loop entirely.
3. **Memoization.** The result for each chunk is cached in a `ConcurrentHashMap` (capped at 100 000 entries,
   and only for chunks of at most 64 bytes so that unusual input cannot fill memory). Chunks like `" the"`,
   `","`, `"\n"` recur constantly; a chat prompt is mostly cache hits.

## 4.7 Added and special tokens

Before any of the above, the text is split on a regex that matches every added token *literally* (longest
first, each `Pattern.quote`d so `|` is not an alternation). Matches map straight to their id; the text between
them goes through stages 1–3:

```
"<|im_start|>user\nhi<|im_end|>\n"
 → [151644]  encode("user\nhi")=[872, 198, 6023]  [151645]  encode("\n")=[198]
 → [151644, 872, 198, 6023, 151645, 198]
```

`<|im_end|>` (151645) and `<|endoftext|>` (151643) are the **end-of-sequence** ids: when the model produces one,
generation stops. `Main.eosIds` collects them from `generation_config.json` and the tokenizer.

## 4.8 Decoding, and the streaming problem

Decoding is concatenation: `bytes = idToBytes[id₀] ++ idToBytes[id₁] ++ …`, then interpret as UTF-8. There is
one trap when **streaming** token by token: a token can contain *part* of a multi-byte character. `🚀` is four
bytes `F0 9F 9A 80`, and the tokenizer may well produce it as two tokens `F0 9F` + `9A 80`. Printing the first
token alone would show `��`.

`StreamDecoder.push(bytes)` therefore buffers bytes and returns only the text up to the last *complete*
character. It walks back from the end over continuation bytes (`10xxxxxx`), looks at the lead byte it finds, and
knows how long that character must be:

```
0xxxxxxx → 1 byte      110xxxxx → 2 bytes      1110xxxx → 3 bytes      11110xxx → 4 bytes
```

If fewer bytes are present than the lead byte announces, they stay in the buffer until the next token.
`flush()` at the end of generation emits whatever is left (an invalid trailing sequence becomes U+FFFD).

## 4.9 The chat template

A base model continues text. A *chat* model was fine-tuned on conversations written in one specific layout, and
it only behaves if the prompt uses that exact layout. Qwen3 uses **ChatML**:

```
<|im_start|>system
You are terse.<|im_end|>
<|im_start|>user
Name three planets.<|im_end|>
<|im_start|>assistant
```

The prompt ends right after `assistant\n`; the model writes the reply and finishes with `<|im_end|>`.
`ChatTemplate.render(messages, addGenerationPrompt, enableThinking)` produces this string (the official version
is a Jinja template inside `tokenizer_config.json`; ours is the same logic in Java, minus tool-calling).

**Thinking mode.** Qwen3 can reason before answering, wrapped in `<think> … </think>`. When thinking is
*disabled* the template appends an empty block, `<think>\n\n</think>\n\n`, so the model sees that the thinking
phase is already over and answers directly. When enabled, the model emits its own `<think>` block first — the
API returns it inline in `content`. Previous assistant turns have their `<think>` blocks stripped before being
fed back (as the official template does) so the context is not flooded with old reasoning.

## 4.10 Verifying correctness

`SmokeTest` compares our ids with those produced by the reference `tokenizers` library for several strings,
checks `decode(encode(s)) == s` on Unicode, emoji, code and whitespace-heavy samples, checks the special-token
ids, the rendered template, and the stream decoder's handling of a split emoji. Matching ids matter: the model
was trained on the reference tokenization, and a slightly different split (e.g. `" wor" "ld"`) would still
decode to the same text but would be out-of-distribution for the model and hurt quality.
