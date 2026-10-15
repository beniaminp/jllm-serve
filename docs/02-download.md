# 2. Downloading a model from the Hugging Face Hub

Code: `hub/HuggingFaceHub.java`

## What is on the Hub

A model repository is a git repository with large files stored through git-LFS. Two HTTP endpoints matter:

1. **Metadata:** `GET https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true` returns JSON listing every
   file (`siblings`), with size and — for LFS files — the SHA-256 of the content:
   ```json
   {"siblings":[{"rfilename":"model.safetensors","size":1503300328,
                 "lfs":{"sha256":"f47f7117...","size":1503300328}}, ...]}
   ```
2. **Content:** `GET https://huggingface.co/{repo}/resolve/{rev}/{file}` answers with a redirect to a CDN URL that
   serves the bytes.

We only need six files for inference (`isInferenceFile`): `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, any `*.safetensors` and, for sharded models, `model.safetensors.index.json`.
We skip `vocab.json`/`merges.txt` (the same information is inside `tokenizer.json`) and the README.

## The HTTP client

`java.net.http.HttpClient` is part of the JDK. Three details make the downloader robust:

- **Redirects are followed manually** (`HttpClient.Redirect.NEVER`). The Hub answers `302` with a `Location` on a
  different host (a CDN). The `Authorization: Bearer <HF_TOKEN>` header must only be sent to `huggingface.co`,
  never to the CDN, so we loop ourselves (`send()`), re-attaching the token only when the host matches.
- **Resumable downloads.** Data streams into `file.incomplete`. If the process is interrupted, the next run
  sends `Range: bytes=<have>-` and the server replies `206 Partial Content` with the rest. A `200` reply means the
  server ignored the range, so we start over. The already-downloaded prefix is re-hashed first so the SHA-256
  stays correct.
- **Verification.** Every chunk is fed through `MessageDigest("SHA-256")` while it is being written. At the end
  the digest must equal the Hub's `lfs.sha256`, otherwise the file is deleted and an error is thrown. Only then is
  `file.incomplete` atomically renamed to its final name, so a complete file on disk is always a correct file.

Retries: up to 5 attempts per file with linear back-off, resuming from whatever reached the disk.

## Why a token is sometimes needed

Public models such as Qwen3-0.6B need none. Gated models (Llama, Gemma) require you to accept a license and
then authenticate; the downloader reads `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `~/.cache/huggingface/token`.
`HF_ENDPOINT` switches to a mirror.

## Path safety

`rfilename` comes from the network. `downloadFile` normalizes `dir.resolve(name)` and refuses anything that
escapes the model directory (e.g. `../../.zshrc`).

## Usage

```sh
./jllm download                          # Qwen/Qwen3-0.6B → models/Qwen_Qwen3-0.6B
./jllm download --model Qwen/Qwen3-1.7B  # any Qwen3 dense model works (same architecture)
```
`run`, `chat`, `serve` and `bench` call the downloader automatically when `config.json` is missing.
