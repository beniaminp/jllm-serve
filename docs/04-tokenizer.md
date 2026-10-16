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

Before any of the above, the text is scanned for added tokens with a **leftmost-longest** match
(`AddedTokenTrie`: a trie over the token strings, indexed by first character, so a character that starts no
added token costs one array read; at a candidate position the deepest terminal node wins, so `<|im_start|>` beats
any shorter token with the same prefix). Matches map straight to their id; the text between them goes through
stages 1–3:

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

## 4.11 Other tokenizer families

The same `Tokenizer` class loads the tokenizers of Llama 3, Mistral, Gemma 2/3, Phi-3/Phi-4, GLM-4 / GLM-Edge,
DeepSeek-R1-Distill and Mamba (GPT-NeoX). All of them are `"model": {"type": "BPE"}`; none uses the Unigram
model, so no Unigram/Viterbi segmentation is implemented (such a file is rejected with "only BPE tokenizers are
supported"). What differs is the pipeline around the merge loop, which is read from `tokenizer.json`:

| Family | normalizer | pre-tokenizer | model flags | decoder |
|---|---|---|---|---|
| Qwen2/3, DeepSeek-R1-Distill | NFC | Split(regex, Isolated) + ByteLevel | `ignore_merges` (Qwen3) | ByteLevel |
| Llama 3, GLM-4, GLM-Edge | none | Split(regex with `\p{N}{1,3}`, Isolated) + ByteLevel | `ignore_merges` | ByteLevel |
| Phi-4-mini | none | Split(case-aware regex, **Removed + invert**) + ByteLevel | | ByteLevel |
| Mamba (GPT-NeoX) | NFC | ByteLevel with `use_regex` (GPT-2 regex) | 13 byte symbols (0xC0, 0xC1, 0xF5..0xFF, never valid UTF-8) absent from the vocabulary; such bytes are dropped | ByteLevel |
| Mistral v0.3 | none | **Metaspace** (`prepend_scheme: first`, `split: false`) | `byte_fallback`, `fuse_unk` | Replace ▁→space, ByteFallback, Fuse, **Strip(1)** |
| Gemma 2/3 | **Replace " " → ▁** | Split(" ", MergedWithPrevious), a no-op once spaces are gone | `byte_fallback`, `fuse_unk` | Replace, ByteFallback, Fuse |
| Phi-3 | **Prepend ▁, Replace " " → ▁** | none | `byte_fallback`, `fuse_unk` | Replace, ByteFallback, Fuse, Strip(1) |

### Byte-level versus SentencePiece style

The byte-level family (Qwen, Llama 3, Phi-4, GLM, DeepSeek, Mamba) works as described in 4.3: bytes are mapped to
the printable GPT-2 alphabet and a space is part of the *following* word (`Ġworld`). The SentencePiece-style
family (Mistral, Gemma, Phi-3) never touches bytes in the vocabulary strings. Instead:

- **Spaces become `▁` (U+2581)**, again attached to the following word (`▁world`). Gemma and Phi-3 do this with a
  `Replace` normalizer; Mistral with the `Metaspace` pre-tokenizer. Phi-3's `Prepend ▁` and Mistral's
  `prepend_scheme: first` put a `▁` in front of the text, so `"hello"` and `" hello"` start alike. `first` means
  *only at offset 0 of the input*: the text after `[INST]` or after `<s>` gets no extra `▁`. `Tokenizer`
  carries that fact through `encode` and through the chat-template encoder (`newSequence()`).
- **The merge loop starts from Unicode characters**, not bytes: each code point is looked up in the vocabulary.
- **Byte fallback.** A character absent from the vocabulary (a rare CJK character, a control byte, an emoji the
  vocabulary never saw) is replaced by the tokens `<0xE6>`, `<0x97>`, … of its UTF-8 bytes; these single-byte
  tokens exist in the vocabulary for exactly this purpose. If even those are missing the character becomes the
  `unk_token`, and with `fuse_unk` a run of unknowns collapses into one `<unk>`.
- **Decoding** turns `▁` back into a space and `<0xXX>` tokens into the raw byte (`tokenBytes(id)` returns that
  single byte, so `StreamDecoder` reassembles multi-byte characters as before). Mistral's and Phi-3's decoder
  ends with `Strip(start=1)`, which removes the first space of the whole text (the one the prefix added);
  `Tokenizer.decode` does the same and `new StreamDecoder(tokenizer)` does it for streams. Consequently
  `decode(encode(" x"))` is `"x"` for those two, exactly like the reference. `clean_up_tokenization_spaces` is
  never applied.

Gemma deserves one remark: after the normalizer there are no spaces left, so its `Split` does nothing and the
**whole text is a single BPE chunk** (merges can cross punctuation and newlines; `\n\n` is one token). The merge
loop is the same O(n log n) heap, so this stays fast, but such chunks are too long for the piece cache.

### Other constructs now understood

- **Split behaviors**: `Isolated`, `Removed`, `MergedWithPrevious`, `MergedWithNext`, `Contiguous`, with
  `invert`. Phi-4's `Removed` + `invert: true` keeps the regex matches and drops what lies between them.
- **ByteLevel** with `add_prefix_space` and `use_regex`; **Digits**, **WhitespaceSplit**, **Whitespace**.
- **Normalizers** `Sequence`, `NFC/NFD/NFKC/NFKD`, `Prepend`, `Replace` (string or regex), `Strip`, `Lowercase`.
- **Added tokens with `lstrip` / `rstrip`.** Phi-3 and Phi-4 declare `<|user|>`, `<|end|>`, `<|assistant|>`, … with
  `rstrip: true`: the tokenizer swallows the whitespace after such a token, so `<|user|>\nhi` encodes as
  `<|user|>` `hi` and the newline never reaches the model. The reference does this and so do we, also across the
  pieces of a chat prompt.
- **Post-processor / BOS.** `bosId()`, `eosId()`, `unkId()` and `addBosByDefault()` come from the
  `TemplateProcessing` post-processor and, when present, the sibling `tokenizer_config.json`
  (`add_bos_token`, `bos_token`, `eos_token`; DeepSeek has no template but sets `add_bos_token`). `encode` never
  prepends BOS; the caller decides. Chat-template output already contains BOS for the families whose template
  starts with it.

Unknown normalizer or pre-tokenizer types throw `IllegalArgumentException` instead of silently tokenizing
differently. Not supported: the Unigram model, `Precompiled` normalizers (SentencePiece `.model` blobs), `BPE`
dropout, `continuing_subword_prefix` / `end_of_word_suffix` (WordPiece-style BPE), `single_word` added tokens,
the `normalized: true` added-token flag (those tokens are matched on the raw text, which is identical for
Mamba's whitespace tokens), `clean_up_tokenization_spaces`, and tool-calling parts of the chat templates.

### Chat templates per family

`ChatTemplate.forModelType(modelType)` (or `forModelType(modelType, tokenizer)`) returns a `ChatTemplate.Format`
with `render(...)`, `renderTokens(tok, ...)` and `stopIds(tok)`. Pass the tokenizer when you can: `phi3` is the
`model_type` of both Phi-3 (SentencePiece vocabulary, newline-delimited roles) and Phi-4-mini (byte-level, no
newlines), and DeepSeek-R1-Distill reports `qwen2` or `llama` although its template is different; the tokenizer
tells them apart. Message content is always encoded with special tokens disabled.

| `model_type` | Prompt layout (generation prompt in bold) | Stop ids (tokens) |
|---|---|---|
| `qwen3` | `<\|im_start\|>user\n…<\|im_end\|>\n` … **`<\|im_start\|>assistant\n`** (+ empty `<think>` block when thinking is off) | `<\|im_end\|>`, `<\|endoftext\|>` |
| `qwen2` | same ChatML, default system prompt "You are Qwen, created by Alibaba Cloud…", no think block | `<\|im_end\|>`, `<\|endoftext\|>` |
| `qwen2`/`llama` + R1 vocabulary | `<｜begin▁of▁sentence｜>{system}<｜User｜>…<｜Assistant｜>…<｜end▁of▁sentence｜>` … **`<｜Assistant｜><think>\n`** | `<｜end▁of▁sentence｜>` |
| `llama` (3.x) | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date…Today Date: …\n\n{system}<\|eot_id\|>` then `<\|start_header_id\|>user<\|end_header_id\|>\n\n…<\|eot_id\|>` … **`<\|start_header_id\|>assistant<\|end_header_id\|>\n\n`**; contents trimmed | `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|eom_id\|>` |
| `mistral` | `<s>[INST] …[/INST] reply</s>[INST] …[/INST]`; the system prompt is prefixed (`sys\n\n`) to the last user turn | `</s>` |
| `gemma2`, `gemma3` | `<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>\n` … **`<start_of_turn>model\n`**; system text is prefixed to the first user turn; contents trimmed | `<end_of_turn>`, `<eos>` |
| `phi3` + SentencePiece vocab | `<\|system\|>\n…<\|end\|>\n<\|user\|>\n…<\|end\|>\n` … **`<\|assistant\|>\n`**; `<\|endoftext\|>` instead when not prompting | `<\|end\|>`, `<\|endoftext\|>`, `<\|assistant\|>` |
| `phi3` + byte-level vocab (Phi-4-mini) | `<\|system\|>…<\|end\|><\|user\|>…<\|end\|>` … **`<\|assistant\|>`** | `<\|end\|>`, `<\|endoftext\|>` |
| `glm4` | `[gMASK]<sop><\|system\|>\n…<\|user\|>\n…<\|assistant\|>\n…` … **`<\|assistant\|>`** (no newline) | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` |
| `glm` (GLM-Edge) | `<\|user\|>\n…<\|assistant\|>\n…` … **`<\|assistant\|>\n`**, no BOS | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` |
| `mamba` | no template: message contents joined by `\n` | `<\|endoftext\|>` |

The Gemma 2 template refuses system messages; we fold them into the first turn as Gemma 3 does. Llama 3.1/3.2
print a date in the system header; the template's own fallback `26 Jul 2024` is used unless you build the
format with `ChatTemplate.llama3("…")`.

### Verification

`TokenizerFamiliesTest` (plain `main`, like `SmokeTest`) compares, for each family, `encode` with the ids of the
Python `tokenizers` package on 145 strings (English, code, numbers, CJK/emoji/RTL, leading/trailing/multiple
spaces, newlines, special-token look-alikes, the empty string, random mixes), `decode` and token-by-token
`StreamDecoder` output with the reference decode, `decode` of random id sequences, and `render` / `renderTokens`
of 3–4 message conversations (with and without system prompt, odd whitespace, Unicode) with the official Jinja
`chat_template` rendered by `jinja2` and encoded by `tokenizers`. The expected values live in
`src/test/resources/tokenizers/*.json`, produced by `src/test/python/gen_tokenizer_fixtures.py`; the tokenizer
files themselves are downloaded (no weights) into a directory passed as the first argument:

```
python3 -m venv /tmp/tokv && /tmp/tokv/bin/pip install tokenizers jinja2
java -cp target/classes:target/test-classes org.robomiri.jllm.TokenizerFamiliesTest /tmp/jllm-tokenizers
```

Families whose directory is missing are skipped. `-Dtokenizer.fixtures=<dir>` points the test at a larger corpus
generated with `gen_tokenizer_fixtures.py … <dir> 4000`.
