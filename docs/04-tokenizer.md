# 4. The tokenizer: byte-level BPE from scratch

Code: `tokenizer/Tokenizer.java`, `tokenizer/ChatTemplate.java`, `tokenizer/StreamDecoder.java`, `tokenizer/AddedTokenTrie.java`,
`tokenizer/LongLongMap.java`, `tokenizer/Normalizers.java`, `tokenizer/PreTokenizers.java`, `tokenizer/Unigram.java`

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

1. **No strings.** Symbols are token ids; the merge table is `LongLongMap`, an open-addressed primitive map
   keyed by `(leftId << 32) | rightId` with value `(rank << 32) | mergedId`. One hash lookup per pair, no
   concatenation, no boxing. (A `HashMap<Long, Long>` would fold the key to `leftId ^ rightId` — every pair
   with the same xor lands in one bin — so the slot comes from a 64-bit multiplicative mix of the whole key.)
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
stages 1–3. An earlier version joined every added token into one regex alternation, which costs time proportional
to the number of tokens at every position (Gemma 3 has 6 415 of them): encoding a 2 MB text took 50 s for Gemma 3
and is now 0.24 s, with Mistral about 10 times and Llama 3 about 17 times faster.

```
"<|im_start|>user\nhi<|im_end|>\n"
 → [151644]  encode("user\nhi")=[872, 198, 6023]  [151645]  encode("\n")=[198]
 → [151644, 872, 198, 6023, 151645, 198]
```

`<|im_end|>` (151645) and `<|endoftext|>` (151643) are the **end-of-sequence** ids: when the model produces one,
generation stops. `Main.eosIds` collects them from `config.json`'s `eos_token_id`, `generation_config.json` and a list of
family stop tokens the vocabulary has (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|end_of_text|>`, `</s>`); the
engine adds the chat template's own end-of-turn ids on top ([10](10-engine-and-server.md)).

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
`ChatTemplate.forModelType("qwen3").render(messages, addGenerationPrompt, enableThinking)` produces this string
(the official version is a Jinja template inside `tokenizer_config.json`; ours is the same logic in Java, minus
tool-calling). There are no static `render` / `renderTokens` helpers any more: every family is a
`ChatTemplate.Format` obtained from `forModelType` (section 4.11).

**Thinking mode.** Qwen3 can reason before answering, wrapped in `<think> … </think>`. When thinking is
*disabled* the template appends an empty block, `<think>\n\n</think>\n\n`, so the model sees that the thinking
phase is already over and answers directly. When enabled, the model emits its own `<think>` block first — the
API returns it inline in `content`. Assistant turns before the last user query have their `<think>` blocks
stripped before being fed back (as the official template does) so the context is not flooded with old
reasoning; an assistant turn *after* the last query (a conversation that ends with the assistant, rendered
without a generation prompt) keeps its reasoning as `<think>\n…\n</think>\n\n` — the template's
`last_query_index` rule. Only newlines are stripped around the pieces (`lstrip('\n')`), never other
whitespace. Messages with role `tool` become `<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>`,
consecutive ones sharing a single user turn.

## 4.10 Verifying correctness

`SmokeTest` compares our ids with those produced by the reference `tokenizers` library for several Qwen strings,
checks `decode(encode(s)) == s` on Unicode, emoji, code and whitespace-heavy samples, checks the special-token
ids and the rendered Qwen3 template, that malformed `tokenizer.json` files fail with an
`IllegalArgumentException` naming the problem, that the heap merge loop equals a verbatim copy of the old
rescan-every-pair implementation, and the piece-cache limits. The cross-family comparison against the reference
is `TokenizerFamiliesTest` (4.11). Matching ids matter: the model
was trained on the reference tokenization, and a slightly different split (e.g. `" wor" "ld"`) would still
decode to the same text but would be out-of-distribution for the model and hurt quality.

## 4.11 Other tokenizer families

The same `Tokenizer` class loads the tokenizers of Llama 3, Mistral, Gemma 2/3, Phi-3 / Phi-4-mini / phi-4,
GLM-4 / GLM-Edge, DeepSeek-V2 / DeepSeek-R1-Distill and Mamba (GPT-NeoX). All of them are
`"model": {"type": "BPE"}`. (The Unigram and WordPiece models are implemented too, for tokenizers outside these
families; see 4.12. A GGUF file carries its tokenizer as lists in its metadata instead of a `tokenizer.json`; see 4.13.) What differs is the pipeline around the merge loop, which is read from `tokenizer.json`:

| Family | normalizer | pre-tokenizer | model flags | decoder |
|---|---|---|---|---|
| Qwen2/3, DeepSeek-R1-Distill | NFC | Split(regex, Isolated) + ByteLevel | `ignore_merges` (Qwen3) | ByteLevel |
| Llama 3, GLM-4, GLM-Edge | none | Split(regex with `\p{N}{1,3}`, Isolated) + ByteLevel | `ignore_merges` | ByteLevel |
| Phi-4-mini | none | Split(case-aware regex, **Removed + invert**) + ByteLevel | | ByteLevel |
| phi-4 (14B) | none | Split(Llama-3 regex, Removed + invert) + ByteLevel; all 96 control tokens `lstrip` + `rstrip` | | ByteLevel |
| DeepSeek-V2 | empty Sequence | Split on `[\r\n]`, a long Latin/CJK-range regex, digits, punctuation (Isolated) + ByteLevel | | ByteLevel |
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
  `Tokenizer.decode` does the same and `new StreamDecoder(tokenizer)` does it for streams (feed it ids with
  `push(int)`). Consequently `decode(encode(" x"))` is `"x"` for those two, exactly like the reference. A
  `Metaspace` *decoder* (older Llama-2-style conversions) is different: the reference drops every `▁` of the
  **first token** — `decode(["▁▁a", "▁b"])` is `"a b"` — which `tokenBytes(id, first)` reproduces.
  `clean_up_tokenization_spaces` is never applied.

Gemma deserves one remark: after the normalizer there are no spaces left, so its `Split` does nothing and the
**whole text is a single BPE chunk** (merges can cross punctuation and newlines; `\n\n` is one token). The merge
loop is the same O(n log n) heap, so this stays fast, but such chunks are too long for the piece cache.

### Other constructs now understood

- **Split behaviors**: `Isolated`, `Removed`, `MergedWithPrevious`, `MergedWithNext`, `Contiguous`, with
  `invert`. Phi-4's `Removed` + `invert: true` keeps the regex matches and drops what lies between them.
- **ByteLevel** with `add_prefix_space` and `use_regex`; **Digits**, **WhitespaceSplit**, **Whitespace**.
- **Normalizers** `Sequence`, `NFC/NFD/NFKC/NFKD`, `Prepend`, `Replace` (string or regex; the regex is compiled
  with `UNICODE_CHARACTER_CLASS` like the pre-tokenizer patterns, so `\s` matches NBSP as in Rust), `Strip`
  (Rust's `char::is_whitespace`, i.e. exactly the Unicode `White_Space` set — the same predicate decides what
  `lstrip` / `rstrip` added tokens swallow), `Lowercase` (code point by code point, like Rust: a final `Σ`
  becomes `σ`, not the `ς` of `String.toLowerCase`), and for the models of 4.12 `StripAccents`, `Nmt`,
  `Precompiled` and `BertNormalizer`.
- **BPE `dropout`** is ignored with a one-line note on stderr. The reference applies it when encoding (each merge
  is skipped with that probability, so the same text gets different ids on every call); that is a regularization
  for training, and an inference engine wants reproducible ids, i.e. dropout 0.
- **Added tokens with `lstrip` / `rstrip`.** Phi-3 and Phi-4 declare `<|user|>`, `<|end|>`, `<|assistant|>`, … with
  `rstrip: true`: the tokenizer swallows the whitespace after such a token, so `<|user|>\nhi` encodes as
  `<|user|>` `hi` and the newline never reaches the model. The reference does this and so do we, also across the
  pieces of a chat prompt.
- **Post-processor / BOS.** `bosId()`, `eosId()`, `unkId()` and `addBosByDefault()` come from the
  `TemplateProcessing` post-processor and, when present, the sibling `tokenizer_config.json`
  (`add_bos_token`, `bos_token`, `eos_token`; DeepSeek has no template but sets `add_bos_token`). `encode` never
  prepends BOS; the caller decides, and `promptBos()` (= `bosId()` when `addBosByDefault()`, else -1) is the
  single answer the engine should use for raw completions instead of guessing from `model_type`
  (DeepSeek-R1-Distill-Qwen is `qwen2` with `add_bos_token: true`). Chat-template output already contains BOS
  for the families whose template starts with it.

Unknown normalizer, pre-tokenizer, decoder or post-processor types throw `IllegalArgumentException` instead of
silently tokenizing differently, as do a non-empty `continuing_subword_prefix` / `end_of_word_suffix` on a BPE
model (BPE with WordPiece-style affixes), a `Replace` decoder other than `▁ → " "`, a `Strip` decoder that is
not "leading spaces of the fused text", and a `tokenizer_config.json` that is not valid JSON. Not supported (the
`Tokenizer` javadoc keeps the list): the WordLevel model, `single_word` added tokens, the `normalized: true`
added-token flag (those tokens are matched on the raw text, which is identical for Mamba's whitespace tokens),
`clean_up_tokenization_spaces`, and tool-calling parts of the chat templates.

### Chat templates per family

`ChatTemplate.forModelType(modelType)` (or `forModelType(modelType, tokenizer)`) returns a `ChatTemplate.Format`
with `render(...)`, `renderTokens(tok, ...)` and `stopIds(tok)`. Pass the tokenizer when you can: `phi3` is the
`model_type` of both Phi-3 (SentencePiece vocabulary, newline-delimited roles) and Phi-4-mini (byte-level, no
newlines), and DeepSeek-R1-Distill reports `qwen2` or `llama` although its template is different; the tokenizer
tells them apart. Message content is always encoded with special tokens disabled: `encode(text, false)` and the
content pieces of a template leave every token flagged `special` as plain text, but still match the *non-special*
added tokens (`<think>`, `<tool_call>`, Gemma's `<unused0>`, Mamba's whitespace-run tokens) exactly as the
reference does with `encode_special_tokens`. A format whose control markers are not flagged special lists them
in `Format.protectedTokens()` (DeepSeek-R1's `<｜User｜>` / `<｜Assistant｜>`), and content cannot produce those
either.

| `model_type` | Prompt layout (generation prompt in bold) | Stop ids (tokens) |
|---|---|---|
| `qwen3` | `<\|im_start\|>user\n…<\|im_end\|>\n` … **`<\|im_start\|>assistant\n`** (+ empty `<think>` block when thinking is off); tool results as `<tool_response>` blocks in a user turn | `<\|im_end\|>`, `<\|endoftext\|>` |
| `qwen2` | same ChatML, default system prompt "You are Qwen, created by Alibaba Cloud…", no think block, same tool-result turns | `<\|im_end\|>`, `<\|endoftext\|>` |
| `qwen2`/`llama` + R1 vocabulary | `<｜begin▁of▁sentence｜>{system}<｜User｜>…<｜Assistant｜>…<｜end▁of▁sentence｜>` … **`<｜Assistant｜><think>\n`** | `<｜end▁of▁sentence｜>` |
| `llama` (3.x) | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date…Today Date: …\n\n{system}<\|eot_id\|>` then `<\|start_header_id\|>user<\|end_header_id\|>\n\n…<\|eot_id\|>` … **`<\|start_header_id\|>assistant<\|end_header_id\|>\n\n`**; contents trimmed | `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|eom_id\|>` |
| `mistral` | `<s>[INST] …[/INST] reply</s>[INST] …[/INST]`; the system prompt is prefixed (`sys\n\n`) to the last user turn | `</s>` |
| `gemma2`, `gemma3` | `<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>\n` … **`<start_of_turn>model\n`**; system text is prefixed to the first user turn; contents trimmed | `<end_of_turn>`, `<eos>` |
| `phi3` + SentencePiece vocab | `<\|system\|>\n…<\|end\|>\n<\|user\|>\n…<\|end\|>\n` … **`<\|assistant\|>\n`**; `<\|endoftext\|>` instead when not prompting | `<\|end\|>`, `<\|endoftext\|>`, `<\|assistant\|>` |
| `phi3` + byte-level vocab (Phi-4-mini) | `<\|system\|>…<\|end\|><\|user\|>…<\|end\|>` … **`<\|assistant\|>`** | `<\|end\|>`, `<\|endoftext\|>` |
| `phi3` + `<\|im_sep\|>` vocab (phi-4 14B) | `<\|im_start\|>user<\|im_sep\|>…<\|im_end\|>` … **`<\|im_start\|>assistant<\|im_sep\|>`**, no newlines | `<\|im_end\|>`, `<\|endoftext\|>` |
| `deepseek_v2` | `<｜begin▁of▁sentence｜>{system}\n\nUser: …\n\nAssistant: …<｜end▁of▁sentence｜>` … **`Assistant:`** (plain-text roles) | `<｜end▁of▁sentence｜>` |
| `glm4` | `[gMASK]<sop><\|system\|>\n…<\|user\|>\n…<\|assistant\|>\n…` … **`<\|assistant\|>`** (no newline) | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` |
| `glm` (GLM-Edge) | `<\|user\|>\n…<\|assistant\|>\n…` … **`<\|assistant\|>\n`**, no BOS | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` |
| `mamba` | no template: message contents joined by `\n` | `<\|endoftext\|>` |

The Gemma 2 template refuses system messages; we fold them into the first turn as Gemma 3 does. Mistral and
Gemma insist, like their Jinja (`raise_exception`), that the turns after the optional system message alternate
user/assistant/user/… and throw `IllegalArgumentException` otherwise. Llama 3.1/3.2 print a date in the system
header: `forModelType("llama")` uses the date of the render (`%d %b %Y`, what transformers' `strftime_now`
produces); `ChatTemplate.llama3("26 Jul 2024")` pins it, which the fixture test does because jinja2 without
`strftime_now` renders the template's fallback date. `phimoe` (Phi-3.5-MoE) renders with the Phi-3 template.

### Verification

`TokenizerFamiliesTest` (plain `main`, like `SmokeTest`) compares, for each of the 14 families and the five
Unigram / WordPiece tokenizers of 4.12 (4 239 checks when all tokenizer files are present), `encode` with the
ids of the Python `tokenizers` package on 150 strings (English, code, numbers, CJK/emoji/RTL,
leading/trailing/multiple spaces, newlines, special and non-special added tokens, indented code, the empty
string, random mixes) — both with special tokens enabled and in the content mode (`encode_special_tokens`) —
`decode` and token-by-token `StreamDecoder` output with the reference decode, `decode` of random id sequences,
and `render` / `renderTokens` of conversations (with and without system prompt, odd whitespace, Unicode,
`<think>` in content, ending with an assistant turn, tool results) with the official Jinja `chat_template`
rendered by `jinja2` and encoded by `tokenizers`. The expected values live in
`src/test/resources/tokenizers/*.json`, produced by `src/test/python/gen_tokenizer_fixtures.py`; the tokenizer
files themselves (`tokenizer.json`, `tokenizer_config.json`, no weights) are downloaded into
`<dir>/<family>/` from these Hugging Face repos (the tokenizer is the same for every size of a family; the
unsloth entries are ungated mirrors of the Google / Mistral files) and the directory is passed as the first
argument:

| directory | repo | | directory | repo |
|---|---|---|---|---|
| `qwen3` | Qwen/Qwen3-0.6B | | `phi4` | microsoft/Phi-4-mini-instruct |
| `qwen2` | Qwen/Qwen2.5-0.5B-Instruct | | `phi4_14b` | microsoft/phi-4 |
| `llama3` | meta-llama/Llama-3.2-1B-Instruct | | `glmedge` | THUDM/glm-edge-1.5b-chat |
| `mistral` | unsloth/Mistral-7B-Instruct-v0.3 | | `glm4` | THUDM/GLM-4-9B-0414 |
| `gemma3` | google/gemma-3-1b-it | | `dsr1` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| `gemma2` | unsloth/gemma-2-2b-it | | `dsv2` | deepseek-ai/DeepSeek-V2-Lite |
| `phi3` | microsoft/Phi-3-mini-4k-instruct | | `mamba` | state-spaces/mamba-130m-hf |
| `t5` | google-t5/t5-small | | `bert` | google-bert/bert-base-uncased |
| `albert` | albert/albert-base-v2 | | `mbert` | google-bert/bert-base-multilingual-cased |
| `xlmr` | FacebookAI/xlm-roberta-base | | | |

```
python3 -m venv /tmp/tokv && /tmp/tokv/bin/pip install tokenizers jinja2
java -cp target/classes:target/test-classes org.robomiri.jllm.TokenizerFamiliesTest /tmp/jllm-tokenizers
```

Families whose directory is missing are skipped. `-Dtokenizer.fixtures=<dir>` points the test at a larger corpus
generated with `gen_tokenizer_fixtures.py … <dir> 4000` (a fourth argument `families` or `other` limits the run to
the 14 families or to the tokenizers of 4.12; those also get 37 extra strings aimed at their normalizers: accents
and case, compatibility forms, control and format characters, CJK, punctuation runs, long words). With 4 000
random strings each, the five tokenizers of 4.12 and the GGUF `t5` path match the reference on all 20 835 cases.

## 4.12 Other model types: Unigram and WordPiece

None of the model families the engine runs needs this section: they all tokenize with BPE, byte-level or
SentencePiece-style (4.11). `tokenizer.json` knows two more model types, though, used by encoder and embedding
models (T5, ALBERT, XLM-R, BERT and its many descendants), and `Tokenizer` reads them so that such tokenizers
load and give exactly the reference ids — for future families, embedding models, or anyone who wants to count
tokens for them. No model weights of these kinds are run.

### Unigram (SentencePiece unigram)

A BPE vocabulary is a list of merges; a **unigram** vocabulary is a list of pieces with a score each:

```json
"model": {"type": "Unigram", "unk_id": 2, "byte_fallback": false,
          "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["▁", -2.01], [".", -3.06], ["▁the", -3.42], ...]}
```

The score is the natural log of the piece's probability under a model in which a text is a sequence of pieces
drawn independently. The probability of one segmentation `x₁ x₂ … xₙ` of a word is the product of its pieces'
probabilities, so its log is the **sum of the scores**, and the tokenizer picks the segmentation with the largest
sum — the most probable one:

```
best(word) = argmax over all segmentations x₁…xₙ of word   Σᵢ log p(xᵢ)
```

There are exponentially many segmentations, but the best one is found by dynamic programming over positions
(the Viterbi algorithm): the best segmentation of the first `e` characters ends in some piece `s[b, e)`, and what
precedes that piece must itself be the best segmentation of the first `b` characters. So

```
best[0] = 0
best[e] = max over pieces s[b, e) in the vocabulary of   best[b] + score(s[b, e))
```

and the answer is read backwards from `best[length]` through the remembered start of each last piece. Example
with T5's vocabulary: `▁unbelievable` is itself a piece (score −12.95); `▁un` + `be` + `lie` + `v` + `able` sums
to −41.17 and the thirteen single characters to −73.25, so the whole word wins. `▁tokenization` is not a piece;
its best segmentation is `▁token` + `ization` (−12.08 − 9.97 = −22.05). Every additional piece adds another
negative score, so long pieces win unless they are rare.

The exact algorithm (`Unigram.java`, the reference's `encode_optimized`):

```
pieces are stored in a trie (UTF-16 characters, child links in a LongLongMap)
best[0] = 0; from[e] = none for all e > 0
for each character position b, in increasing order:
    walk the trie along s[b..]: for every piece s[b, e) found (shorter pieces first)
        cand = score(piece) + best[b]
        if from[e] is none or cand > best[e]:  best[e] = cand, from[e] = b, id[e] = piece
    if no piece is exactly the single character at b:      (the character is unknown)
        e = b + 1 character; cand = (min score of the vocabulary − 10) + best[b]; same update with id = unk
backtrack from e = length; adjacent unknown characters are fused into one run
each run of unknowns: the vocabulary id of the run's text if it happens to be a piece, else its <0xXX> byte
tokens when byte_fallback is on and all exist, else unk_id
```

The strict `>` decides ties: the first candidate found wins, i.e. the one whose last piece starts earliest (at
equal scores `ab` beats `a` + `b`). The unknown score — 10 below the worst piece — makes an unknown character
always the last resort, yet still allows a path through any text. The work is O(length × longest piece), and
results are memoized per pre-tokenized word like BPE chunks.

Unigram tokenizers usually come with:

- **`Precompiled` normalizer**: the `precompiled_charsmap` of the SentencePiece `.model` file, base64-encoded.
  It is a double-array trie ("darts-clone": each 32-bit unit packs a label byte, a has-leaf bit and an XOR
  offset) over UTF-8 byte strings, whose leaf values point into a blob of NUL-terminated replacement strings.
  For the usual `nmt_nfkc` rule set the table encodes NFKC plus SentencePiece's own clean-up (control characters
  removed, odd spaces made plain). `Normalizers.Precompiled` reads the table and applies it like the reference:
  the text is cut into grapheme clusters (`\X`); a cluster shorter than 6 bytes is replaced as a whole when a
  prefix of it is in the trie (the shortest such prefix decides), otherwise each code point is looked up alone.
  Interpreting the real table rather than calling `java.text.Normalizer` makes the result exact; only a file
  whose `precompiled_charsmap` is `null` (which the reference refuses to load) falls back to plain NFKC, an
  approximation that differs on control characters and unusual spaces.
- **`WhitespaceSplit` + `Metaspace`** pre-tokenizer: split on whitespace, then put `▁` in front of each word —
  the dummy prefix, so that a word at the start of the text and after a space are the same piece.
- **`Metaspace` decoder**: `▁` becomes a space, except in the first token (4.11).

ALBERT adds two `Replace` normalizers (two backticks or two apostrophes become `"`), `NFKD`, `StripAccents` (drops all combining marks) and `Lowercase` in
front of its charsmap. XLM-R's vocabulary has 250 002 pieces and loads in about a second.

**From GGUF.** A GGUF file with `tokenizer.ggml.model = t5` stores the pieces in `tokenizer.ggml.tokens`, the
scores in `tokenizer.ggml.scores` and the charsmap in `tokenizer.ggml.precompiled_charsmap` (raw bytes).
`GgufTokenizer` turns them into the same Unigram `tokenizer.json` (`remove_extra_whitespaces` becomes the
`WhitespaceSplit`, `add_space_prefix` the Metaspace prefix, control / unknown tokens become special added
tokens). The test rebuilds T5's tokenizer this way from its pieces, scores and charsmap and gets the reference
ids for every fixture string.

### WordPiece (BERT)

```json
"normalizer":    {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                  "strip_accents": null, "lowercase": true},
"pre_tokenizer": {"type": "BertPreTokenizer"},
"model":         {"unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                  "vocab": {"[PAD]": 0, ..., "the": 1996, ..., "##s": 2015, ...}},
"decoder":       {"type": "WordPiece", "prefix": "##", "cleanup": true}
```

(Older files such as this one have no `"type"` in the model; like the reference, `Tokenizer` recognizes the
model by its fields: a vocabulary *list* is Unigram, a `continuing_subword_prefix` without merges is WordPiece.)

WordPiece has no scores and no merges, only a vocabulary in which a piece that continues a word is written
with the prefix `##`. Encoding a word is **greedy longest-match-first**: take the longest vocabulary entry that
the word starts with, then the longest `##` entry at the position after it, and so on:

```
"tokenization"  →  "token"  then "##ization"             (ids 19204, 3989)
"unaffable"     →  "una"  then "##ffa"  then "##ble"      (ids 14477, 20961, 3468)
```

The second line shows the greediness: `una` is the longest prefix in the vocabulary, so the split is `una|ffa|ble`
rather than the linguistically nicer `un|aff|able`.

Exactly (`Tokenizer.wordPiece`):

```
if the word has more than max_input_chars_per_word characters: return [unk]
start = 0
while start < length:
    for end = length down to start + 1 (one character at a time):
        candidate = word[start, end), with "##" in front unless start = 0
        if candidate is in the vocabulary: emit it, start = end, continue the outer loop
    no candidate at all: return [unk]          (the whole word, not just the rest)
```

Greedy is not optimal in any probabilistic sense — it is simply what BERT was trained with — and a word whose
remainder cannot be spelled at some point becomes a single `[UNK]`, losing the pieces already found.

The pipeline around it:

- **`BertNormalizer`**, in this order: *clean text* (drop NUL, U+FFFD and characters of category Cc / Cf / Co
  except tab, newline and carriage return; every whitespace character becomes a space), *handle Chinese
  characters* (a space on both sides of every CJK ideograph, so each becomes its own word), *strip accents*
  (NFD, then drop nonspacing marks; `strip_accents: null` means "when lowercasing"), *lowercase* (per code
  point).
- **`BertPreTokenizer`**: split on whitespace (removed), then make every punctuation character its own piece:
  ASCII punctuation including symbols such as `$ + < = > ^ | ~`, and Unicode category P. `"(hello) €5"` is `(`,
  `hello`, `)`, `€5`: the euro sign (a currency symbol, category Sc) is not split off, and WordPiece then makes
  it `€` `##5`.
- **`WordPiece` decoder**: the first token as is; every other token loses its `##` and is glued to the previous
  one, or gets a space in front if it has no `##`. With `cleanup`, each token then has the space removed before
  `. ? ! ,` and before English contractions (`n't`, `'m`, `'s`, `'ve`, `'re`). This is done per token, so
  `tokenBytes(id, first)` and `StreamDecoder` reproduce it. Without any decoder the reference joins the token
  strings with spaces, which a Unigram or WordPiece tokenizer here does too.

The BOS of these tokenizers comes from their `TemplateProcessing` post-processor as before: `[CLS]` for BERT and
ALBERT, `<s>` for XLM-R, none for T5; `encode` never adds it (nor the trailing `[SEP]` / `</s>`).

## 4.13 How a GGUF file carries its tokenizer

🧸 **A box with a word list and a name tag.** A GGUF file ([chapter 16](16-gguf.md)) is one box that holds everything about a model. The tokenizer travels in the box as plain lists: all the puzzle pieces of the vocabulary, a score or a "which pieces glue first" list for each, and a label on every piece saying whether it is an ordinary piece, a
special signal such as "the turn ends here", or a raw byte. What the box does **not** hold is the instruction for *cutting the sentence into words before gluing*: it holds only a name tag for that instruction (`llama-bpe`, `qwen2`, `deepseek-llm` ...), and the instruction itself is written out inside llama.cpp. Each family cuts differently:
Llama 3 chops a long number into groups of three digits, Qwen takes digits one at a time, GPT-2 keeps a whole number together. If jllm cut even one place differently from llama.cpp, the model would be fed pieces it was never taught and answer a little worse without any error. So jllm has the same instruction for every name tag, and refuses a tag it does not know.

📐 **What the file stores.** All keys start with `tokenizer.ggml.`:

| key | holds | used for |
|---|---|---|
| `model` | `gpt2` (byte-level BPE), `llama` (SentencePiece), `t5` (SentencePiece unigram; see 4.12) | which kind of tokenizer to build |
| `tokens` | the vocabulary, a string for each id | the vocabulary |
| `token_type` | per id: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte | special tokens, byte fallback |
| `merges` | `"a b"` strings in merge order (byte-level BPE only) | the merge loop of 4.6 |
| `scores` | a log-probability per piece (`llama` and `t5`) | SentencePiece segmentation |
| `pre` | the name tag of the cutting instruction | the pre-tokenizer of 4.5 |
| `bos_token_id`, `eos_token_id`, `unknown_token_id`, `add_bos_token`, `add_eos_token` | ids and flags | the BOS rule (below) |
| `add_space_prefix`, `remove_extra_whitespaces`, `precompiled_charsmap` | SentencePiece normalization | the `▁` prefix and, for `t5`, the normalization table |
| `tokenizer.chat_template` (no `ggml.`), `general.name` | the Jinja chat template; the model's name | the chat template of 4.9; Phi-3's whitespace rule |

**How jllm uses it.** `GgufTokenizer.build` does not contain a second tokenizer. It writes, in memory, the `tokenizer.json` that the same model has in the Hugging Face world and loads it with the `Tokenizer` of this chapter:

```
 GGUF metadata                                         tokenizer.json built in memory                          the Tokenizer of 4.2-4.9
 tokens, merges, token_type (normal / user / control) ► model {type BPE, vocab, merges, ignore_merges}   ┐
 pre = "qwen2"  ───────────────────────────────────►  pre_tokenizer Sequence [ Split(regex 1, Isolated),   │  the same merge loop,
                                                                              Split(regex 2, Isolated) …,  ├  the same added-token scan,
                                                                              ByteLevel ]                  │  the same decoder
 control / user-defined tokens ───────────────────►  added_tokens (special: true / false, rstrip for Phi-3) ┘
 model = llama: tokens, scores ───────────────────►  model {type LlamaSpm, spm_tokens, spm_scores, byte_fallback}, normalizer Prepend ▁ + Replace " " → ▁
```

**The pre-tokenizer is a list of regular expressions** applied one after the other; each cuts the pieces the previous one left, keeping the matches *and* the text between them (a Hugging Face `Sequence` of `Split` with `Isolated`), and then GPT-2's byte spelling follows. Two flags come with the name tag: `ignore_merges` (a piece that is already a token is not merged further) and the default for the BOS. The same sentence cut by five name tags (the regexes of `GgufPre` applied
in order; a leading space belongs to the word after it, written here with `␣`):

```
text:  "I have 12345 cats, don't panic!"
gpt-2 (1 regex, digits stay together):     I | ␣have | ␣12345 | ␣cats | , | ␣don | 't | ␣panic | !
llama-bpe (1 regex, digits in threes):     I | ␣have | ␣ | 123 | 45 | ␣cats | , | ␣don | 't | ␣panic | !        ignore_merges, BOS added
qwen2 (1 regex, one digit at a time):      I | ␣have | ␣ | 1 | 2 | 3 | 4 | 5 | ␣cats | , | ␣don | 't | ␣panic | !
deepseek-llm (6 regexes, digits together): I | ␣have | ␣ | 12345 | ␣cats | , | ␣don | ' | t | ␣panic | !
default / key absent (4 regexes):          I | ␣have | ␣ | 123 | 45 | ␣cats | , | ␣don | ' | t | ␣panic | !         llama.cpp's default, not GPT-2's
```

(`llama-bpe` and `qwen2` end each digit group alone after a lone space piece because their expression for numbers does not take the leading space.) After cutting, each piece goes through the merge loop of 4.6 on its own. A missing key or `default` is llama.cpp's own default (the last line), **not** GPT-2's; a value jllm does not know is refused with its name, and the loader then looks for the original model's `tokenizer.json` named in the file's `general.base_model.0.repo_url` / `general.source.url`
([16.7](16-gguf.md#167-the-tokenizer-in-the-file)). [16.12](16-gguf.md#1612-tokenizer-pre-tokenizers-which-values-are-supported-and-how-they-were-checked) has the table of all name tags and how each was checked.

**SentencePiece is not BPE.** For `model = llama` (Llama 2, Mistral 7B, Gemma, Phi-3) the file has scores, not merges, and llama.cpp segments greedily: the characters are the first symbols; then, over and over, the adjacent pair whose concatenation is a token of the vocabulary and has the **best score** is merged (equal scores: the leftmost pair); a character that is no token becomes its `<0xXX>` byte tokens (`SpmGreedy`). A toy vocabulary
(scores invented; the code was run on it) for `▁hello`:

```
symbols  ▁ h e l l o            pairs that are tokens (score):  ▁h −3.0, he −2.0, el −5.0, ll −1.5, lo −4.0         → merge ll (best, −1.5)
symbols  ▁ h e ll o            new pairs: ell −6.0, llo −2.5; he −2.0 is still the best                                → merge he
symbols  ▁ he ll o             new pairs: ▁he −3.5, hell −2.2; llo −2.5                                                → merge hell
symbols  ▁ hell o              new pairs: ▁hell −4.0, hello −1.0                                                       → merge hello
symbols  ▁ hello               ▁hello −0.5                                                                             → merge ▁hello       result: one token
"▁helloé": the é is no token and becomes its UTF-8 bytes <0xC3> <0xA9>      →  ▁hello <0xC3> <0xA9>
```

Turning the same scores into BPE merge ranks (as `transformers` does) gives a different result as soon as two scores are equal, which is why jllm follows llama.cpp here: Gemma's scores are all 0, so every pair ties and the leftmost wins.

**Special tokens and BOS.** Tokens of type control (`<|im_start|>`, `<|endoftext|>`) and user-defined (`<think>`, `<tool_call>`) are found in the text before anything else, as the *added tokens* of 4.7; llama.cpp also treats some tokens as control by their **name** whatever the file says (`<|im_end|>`, `</s>`, the fill-in-the-middle markers), and so does jllm. Control tokens are recognised in the text when special tokens are parsed (a chat prompt); user-defined ones are recognised even in untrusted text (`encode(text, false)`), as in llama.cpp. Phi-3 files (`general.name`) make special tokens swallow the whitespace that follows them. A BOS is added when `add_bos_token` says so; when the key is absent: yes for SentencePiece, `llama-bpe` and `tekken`, no otherwise. When the file's pre-tokenizer is unknown and the original repository's `tokenizer.json` is used instead (Gemma 4), that repository's `tokenizer_config.json` has no `add_bos_token` and its post-processor adds no BOS, so a raw prompt (`run --raw`, `/v1/completions`) would reach the model without `<bos>` and degenerate ("0.111..."); `ModelLoader.ggufTokenizer` therefore overrides `add_bos_token` with the GGUF's own key. Chat templates that already start with `<bos>` are not given a second one (`Engine.preparePrompt` is idempotent).

**Why this design.** The alternative to rebuilding a Hugging Face tokenizer is a second, llama.cpp-shaped encoder next to the first; this way there is one merge loop to keep fast and correct and only the *translation* is new. The costs: the translation must reproduce llama.cpp's behaviour, which differs from the Hugging Face tokenizer of the same model in a few places (no NFC normalization, `</s>` a control token: on Qwen3, 5 of the 150 fixture strings), and
every name tag needs its regexes copied from llama.cpp (four small changes so `java.util.regex` acts like C++; see [16.12](16-gguf.md#1612-tokenizer-pre-tokenizers-which-values-are-supported-and-how-they-were-checked)).
**Settings.** None: the tokenizer follows the file. If a name tag is unknown the load fails with the tag in the message, or falls back to the original `tokenizer.json` when the file names its source repository.
**Checked.** Against `llama-tokenize` on 14 871 strings per name tag, llama.cpp's own `ggml-vocab-*.gguf` files (47 strings each), and Qwen3-0.6B's GGUF against its `tokenizer.json` (145 of 145 strings): [16.7](16-gguf.md#167-the-tokenizer-in-the-file) and [16.12](16-gguf.md#1612-tokenizer-pre-tokenizers-which-values-are-supported-and-how-they-were-checked) have the numbers and the commands (`scripts/gguf-tokenizer-check.sh`, `GgufTokenizerTest`).
**Limits.** WordPiece (`bert`), `add_eos_token`, and the name tags that are not regex lists are refused; characters assigned in Unicode 14-17 differ from llama.cpp's older tables; two overlapping special tokens resolve leftmost here and longest in llama.cpp ([16.12](16-gguf.md#1612-tokenizer-pre-tokenizers-which-values-are-supported-and-how-they-were-checked) lists every deliberate difference).
