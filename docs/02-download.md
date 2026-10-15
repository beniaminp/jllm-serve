# 2. Downloading a model from the Hugging Face Hub

Code: `hub/HuggingFaceHub.java`, `model/ShardProvider.java`, `Main.load`

Before anything can run, ~1.5 GB (Qwen3-0.6B) or 61 GB (Qwen3-30B-A3B) of weights must be on disk. This
chapter explains the few networking concepts involved, then walks through exactly what the downloader does —
including how a 61 GB model is fetched on a laptop with 50 GB free.

---

## 2.0 The toolbox: HTTP in five minutes

### A request and a response

HTTP is a text protocol. The client sends a request line, headers, a blank line; the server answers with a
status line, headers, a blank line, and the body. Here is what `HuggingFaceHub` sends to fetch a file:

```
GET /Qwen/Qwen3-0.6B/resolve/main/config.json HTTP/1.1
Host: huggingface.co
User-Agent: jllm-serve/1.0 (java 26.0.2.1)
Authorization: Bearer hf_xxxxxxxx            ← only if a token is configured, only to the Hub host

HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 726

{ "architectures": ["Qwen3ForCausalLM"], ... }
```

The **status code** says what happened: `200` success, `206` partial content (see below), `301/302/307/308`
"go to this other URL", `401/403` not allowed, `404` no such thing, `416` the requested byte range is invalid.
`java.net.http.HttpClient` (part of the JDK) builds requests and parses responses; we never touch raw sockets
here.

### Redirects

Large files are not served by `huggingface.co` itself but by a content delivery network (CDN). The Hub answers
with a redirect:

```
GET /Qwen/Qwen3-0.6B/resolve/main/model.safetensors HTTP/1.1
Host: huggingface.co

HTTP/1.1 302 Found
Location: https://cdn-lfs-us-1.hf.co/repos/.../f47f7117...?Expires=1760600000&Signature=...
```

The client must then issue a *new* GET to the `Location` URL. `HttpClient` can do this automatically, but we
deliberately switch that off (`HttpClient.Redirect.NEVER`) and loop ourselves in `send()`, for the reason in
2.2.

### Byte ranges (resuming)

A client can ask for only part of a file:

```
GET https://cdn-lfs-us-1.hf.co/...  HTTP/1.1
Range: bytes=1000000000-                     ← "from byte 1 000 000 000 to the end"

HTTP/1.1 206 Partial Content
Content-Range: bytes 1000000000-1503300327/1503300328
Content-Length: 503300328
```

`206` means "here is the part you asked for". If the server does not support ranges it answers `200` with the
whole file; if the range starts at or past the end it answers `416`. All three cases are handled in `transfer()`.

### Hashes

A **cryptographic hash** (we use SHA-256) is a function that turns any amount of data into a fixed 64-hex-digit
fingerprint, such that changing even one bit of the input changes the fingerprint completely and it is
infeasible to find two inputs with the same fingerprint. The Hub publishes the SHA-256 of every large file; if
our computed hash matches, the bytes are exactly what was uploaded — no corruption, no truncation, no tampering.

---

## 2.1 What is on the Hub

A model repository is a git repository whose large files are stored through git-LFS. Two endpoints matter:

**1. Metadata** — `GET {endpoint}/api/models/{repo}/revision/{rev}?blobs=true` returns JSON describing every file.
The fields `listFiles()` reads:

```json
{
  "sha": "c1899de289a04d12100db370d81485cdf75e47ca",        ← the commit; not used
  "siblings": [
    {"rfilename": "config.json", "size": 726},                                        ← small file: no "lfs"
    {"rfilename": "model.safetensors", "size": 1503300328,
     "lfs": {"sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
             "size": 1503300328}},                                                     ← LFS file: has a hash
    {"rfilename": "tokenizer.json", "size": 11422654, "lfs": {"sha256": "aeb1...", ...}},
    ...
  ]
}
```

Each sibling becomes a `RepoFile(name, size, sha256)`; `sha256` is `null` for non-LFS files (they are small
and simply size-checked). The listing is cached per `repo@revision` so the converter's repeated "give me shard
N" calls do not hit the API again.

**2. Content** — `GET {endpoint}/{repo}/resolve/{rev}/{file}` → redirect → bytes.

`{endpoint}` is `https://huggingface.co` unless `hub.endpoint` (settings) or `$HF_ENDPOINT` names a mirror.
`{rev}` is a branch, tag or commit hash (`model.revision`, default `main`); it is URL-encoded segment by segment
(`encodePath`) so a revision like `refs/pr/3` works.

### Which files we take

`isInferenceFile(name)` keeps only what inference needs:

| file | why |
|---|---|
| `config.json` | the architecture's dimensions ([05](05-transformer-math.md)) |
| `generation_config.json` | end-of-sequence ids and default sampling values |
| `tokenizer.json` | vocabulary + merges ([04](04-tokenizer.md)) |
| `tokenizer_config.json` | chat template (kept for reference) |
| `*.safetensors` | the weights ([03](03-safetensors.md)) |
| `model.safetensors.index.json` | tensor → shard map for sharded checkpoints |

Skipped: `README.md`, `LICENSE`, `vocab.json`/`merges.txt` (duplicates of `tokenizer.json`), `.gitattributes`.

---

## 2.2 Why redirects are followed by hand

Gated models (Llama, Gemma, …) need an access token sent as `Authorization: Bearer hf_…`. The CDN that
actually serves the bytes is a different host (`cdn-lfs-us-1.hf.co`, or a mirror's storage). If the HTTP client
followed the redirect automatically it would re-send *all* headers — including the token — to that other host.
A misconfigured mirror or a malicious `Location` would then receive your credential.

`send()` therefore loops over redirects itself (at most `MAX_REDIRECTS = 10`), rebuilding the request each time
and attaching the token **only when the target host equals the Hub endpoint's host**:

```java
if (token != null && hubHost.equalsIgnoreCase(current.getHost())) {
    b.header("Authorization", "Bearer " + token);
}
```

The token comes from, in order: the `hub.token` setting, `$HF_TOKEN`, `$HUGGING_FACE_HUB_TOKEN`, or the file
`~/.cache/huggingface/token` written by `huggingface-cli login`. Public models such as Qwen3 need none.

---

## 2.3 One file, start to finish (`downloadFile`)

```
1.  target = dir/<name>, normalized; refuse if it escapes dir            (path traversal, 2.6)
2.  if target exists with the expected size → "up to date", return       (idempotent re-runs)
3.  part   = dir/<name>.incomplete
    have   = size of part if it exists (0 otherwise; reset to 0 if larger than expected)
    sha    = new SHA-256; feed it the existing `have` bytes of part       (hashExisting)
4.  up to 5 attempts:
        transfer(uri, part, have, …)       ← GET with Range when have > 0, stream to disk, hash as we go
        on IOException: wait attempt × 1 s, re-read `have` from disk, re-hash, try again
5.  final size must equal the Hub's size; final hash must equal the Hub's sha256 (else delete part, fail)
6.  atomic rename part → target
```

Why each step matters:

- **`.incomplete` + atomic rename.** A file with its final name is *always* complete and verified. A crash,
  Ctrl-C or network failure leaves only a `.incomplete` file, which the next run resumes. `ATOMIC_MOVE` means no
  reader can ever see a half-written `model.safetensors`.
- **Hashing while writing.** Hashing 1.5 GB after the download would mean reading it back from disk — a second
  pass. Instead every 1 MiB buffer goes to `out.write` *and* `sha.update` in the same loop, so verification is
  free.
- **Resume re-hashes the partial file.** The hash state must cover every byte of the final file in order. After
  a restart (or a failed attempt, whose stream state is unknown) we rebuild it by reading the bytes already on
  disk once (`hashExisting`), then continue hashing the new bytes as they arrive.

A worked resume. The Hub says `model.safetensors` is 1 503 300 328 bytes with hash `f47f…`. The first run died
after 1 000 000 000 bytes:

```
on disk: model.safetensors.incomplete   1 000 000 000 bytes
have = 1 000 000 000;  sha ← hash of those bytes (≈1 s of reading)
GET … Range: bytes=1000000000-
  ← 206, 503 300 328 bytes: appended to the file, fed to sha      (progress bar shows the resumed rate)
size 1 503 300 328 == expected ✓    sha.digest() == f47f… ✓
rename → model.safetensors
```

Had the server answered `200` instead (no range support), `transfer()` truncates the file, resets the hash and
starts over; had it answered `416` with `have == size`, the file was already complete and is just finalized.

### Retries

Network errors (`IOException`: reset connection, timeout, DNS blip) trigger up to 5 attempts with a linear
back-off of 1, 2, 3, 4 seconds. Each retry resumes from whatever reached the disk, so a flaky connection
costs time, not progress. An HTTP error status (404, 403) is also raised as an `IOException` and therefore
retried the same way — harmless but slow; see the troubleshooting table.

### The progress line

`Progress` prints at most 4 times a second (`\r`-overwritten): bytes, percentage and the rate *since this
attempt started* (so a resumed download does not show a misleading average).

---

## 2.4 Three ways to download

The same `downloadFile` serves three entry points, used in different situations:

| method | what it fetches | used by |
|---|---|---|
| `download(repo, rev, dir)` | every inference file, in listing order | `./jllm download` |
| `downloadMetadata(repo, rev, dir)` | only the non-`.safetensors` inference files | `Main.load` when `config.json` is missing |
| `ensure(repo, rev, dir, name)` | exactly one named file, if not already present | `ShardProvider.ensure` → `ModelConverter`, one shard at a time |

`download` first checks free space: it sums the sizes of the files that are *not* already complete and compares
with `Files.getFileStore(dir).getUsableSpace()`. If the model does not fit it fails immediately with a message
pointing at `--low-disk`, rather than filling the disk and dying at shard 13.

The on-demand path is what makes large models practical. `Main.load` builds a `ShardProvider`:

```java
ensure(name)  → dir/name if present, else hub.ensure(model, revision, dir, name)
release(name) → if model.low-disk: delete dir/name
```

and hands it to `ModelWeights.load` → `ModelConverter`, which asks for shards only when it needs a tensor in them
and releases each shard once every tensor in it has been consumed ([03 §3.5](03-safetensors.md#35-from-safetensors-to-jllms-quantized-cache)).

---

## 2.5 On disk, step by step: a 16-shard model with `--low-disk`

Qwen3-30B-A3B-Thinking-2507: 16 shards of ~3.8 GB (61 GB total), 48 layers, ~3 layers per shard, with some
layers straddling two shards. Machine: 50 GB free. Command:

```sh
./jllm run --model Qwen/Qwen3-30B-A3B-Thinking-2507 --low-disk --prompt "..."
```

| moment | files in `models/Qwen_Qwen3-30B-A3B-Thinking-2507/` | free disk |
|---|---|---|
| start | — | 50 GB |
| `downloadMetadata` | `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json` (1.7 MB: the tensor → shard map) | 50 GB |
| converter needs `model.embed_tokens.weight` → `ensure(shard 1)` | + `model-00001-of-00016.safetensors.incomplete` growing… then renamed `…safetensors` (3.8 GB) | 46 GB |
| embedding + layers 0–2 quantized | + `jllm-q4_0.bin` growing (0.35 GB embedding, then ~0.4 GB per layer) | 45 GB |
| layer 3 straddles shards 1 and 2 → `ensure(shard 2)` | shard 1 + shard 2 + bin | 41 GB |
| last tensor of shard 1 consumed → `release(shard 1)` → **deleted** | shard 2 + bin | 45 GB |
| … repeats: at any moment 1–2 shards (≤ 7.6 GB) + the growing `.bin` | | |
| layer 47 done, final norm, `lm_head.weight` from shard 16 | shard 16 + `jllm-q4_0.bin` (19.7 GB) | 26 GB |
| `release(shard 16)`, `jllm-q4_0.json` written (**now** the cache is valid) | `jllm-q4_0.bin` + `jllm-q4_0.json` + the five small files | 30 GB |
| every later start | the same; shards are never needed again | |

Peak usage is about 2 shards + the finished cache ≈ 27 GB. Without `--low-disk` the shards are kept
(standard behaviour, lets you reconvert with `--quant q8` without downloading again), which needs 61 + 20 GB.

If the process is interrupted mid-way, the next run finds a `.bin` without a `.json`, discards it and starts
the conversion over; shards still on disk are reused, deleted ones are downloaded again. (Resuming the
conversion itself is a possible improvement, see [11](11-performance.md).)

---

## 2.6 Path traversal protection

File names come from the network (`rfilename` in the API response). A hostile or buggy listing could contain
`../../.zshrc`. `downloadFile` resolves the target, normalizes it and refuses anything outside the model
directory:

```java
Path target = dir.resolve(f.name()).normalize();
if (!target.startsWith(dir.normalize())) {
    throw new IOException("refusing to write outside model dir: " + f.name());
}
```

Sub-directories inside the repo (`onnx/model.onnx`) are allowed and created with `createDirectories`.

---

## 2.7 Troubleshooting

| symptom | meaning | fix |
|---|---|---|
| `Hub API … returned HTTP 401/403 (gated/private repo? set HF_TOKEN)` | the model requires accepting a license and authenticating | accept the license on the model page, then `export HF_TOKEN=hf_…` or `huggingface-cli login`, or set `hub.token` |
| `Hub API … returned HTTP 404` | repo name or `model.revision` is wrong | check the spelling (`Qwen/Qwen3-0.6B`, case-sensitive) and the revision |
| `need 61.1 GB … but only 50.2 GB are free` | `./jllm download` refuses to start a download that cannot fit | run `run`/`serve` with `--low-disk` instead, or free space |
| `…: sha256 mismatch` | the bytes on disk differ from what the Hub published (corruption in transit or on disk); the `.incomplete` file was deleted | just re-run; if it repeats, check the disk and any proxy |
| `…: size mismatch` | the server closed the connection early 5 times in a row | re-run (it resumes); check the network |
| repeated `retrying (n/5)` with HTTP 404 in the message | one file of the listing does not resolve at this revision | report the repo; the retries are harmless |
| `refusing to write outside model dir` | a file name in the listing contains `..` | do not use that repo |
| very slow (< 5 MB/s) | single connection, far from the CDN | set `HF_ENDPOINT` to a nearby mirror; parallel shard downloads are a possible improvement |

---

## 2.8 The whole thing in 12 lines

```
listing  = GET /api/models/{repo}/revision/{rev}?blobs=true          # names, sizes, sha256
for each file we need (or the one shard the converter asked for):
    if dir/name exists with the right size: skip
    have = bytes already in dir/name.incomplete; sha = hash(those bytes)
    retry up to 5×:
        GET /{repo}/resolve/{rev}/{name}  [Range: bytes=have-]       # follow redirects by hand; token only to the Hub
        append the stream to dir/name.incomplete, hashing as we go
    check size == listing.size and sha == listing.sha256
    rename dir/name.incomplete → dir/name                             # atomic: a final name means "complete and verified"
--low-disk: the converter deletes each shard as soon as all of its tensors are quantized
```
