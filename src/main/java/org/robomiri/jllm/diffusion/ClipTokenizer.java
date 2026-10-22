package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The CLIP byte-pair-encoding tokenizer of Stable Diffusion's text encoders ({@code vocab.json} + {@code merges.txt}):
 * the text is NFC-normalised, white space collapsed, lower-cased and cut into words by a fixed pattern; each word's
 * bytes are mapped to printable characters, the last one gets the end-of-word marker {@code </w>}, and the merges are
 * applied in rank order. The sequence is {@code <|startoftext|> ids <|endoftext|>}, truncated to the model's length and
 * padded with the pad token (the end-of-text token for CLIP ViT-L, {@code "!"} for OpenCLIP).
 */
public final class ClipTokenizer {

    private static final Pattern WORDS = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE);

    private final Map<String, Integer> vocab;
    private final Map<String, Integer> ranks = new HashMap<>();
    private final String[] byteChar = new String[256];
    private final Map<String, int[]> cache = new HashMap<>();
    /** Special tokens that are matched in the raw text before anything else (e.g. {@code "!"} in OpenCLIP's tokenizer). */
    private final Map<String, Integer> literal = new HashMap<>();
    private final Pattern literalPattern;
    public final int bos;
    public final int eos;
    public final int pad;
    public final int maxLength;

    public ClipTokenizer(Path dir) throws IOException {
        Map<String, Object> v = Json.parseObject(Files.readString(dir.resolve("vocab.json"), StandardCharsets.UTF_8));
        vocab = new HashMap<>();
        for (Map.Entry<String, Object> e : v.entrySet()) {
            vocab.put(e.getKey(), ((Number) e.getValue()).intValue());
        }
        List<String> lines = Files.readAllLines(dir.resolve("merges.txt"), StandardCharsets.UTF_8);
        int rank = 0;
        for (String line : lines) {
            if (line.isEmpty() || line.startsWith("#version")) {
                continue;
            }
            ranks.put(line, rank++);
        }
        // GPT-2 / CLIP byte <-> printable character table
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            byteChar[b] = String.valueOf((char) (printable ? b : 256 + n++));
        }
        Map<String, Object> cfg = Map.of();
        Path cfgFile = dir.resolve("tokenizer_config.json");
        if (Files.isRegularFile(cfgFile)) {
            cfg = Json.parseObject(Files.readString(cfgFile, StandardCharsets.UTF_8));
        }
        Integer b = vocab.get("<|startoftext|>");
        Integer e = vocab.get("<|endoftext|>");
        if (b == null || e == null) {
            throw new IOException(dir + ": vocab.json lacks <|startoftext|> / <|endoftext|>");
        }
        bos = b;
        eos = e;
        String padToken = Json.str(cfg, "pad_token", "<|endoftext|>");
        Integer p = vocab.get(padToken);
        pad = p == null ? eos : p;
        Map<String, Object> added = Json.map(cfg, "added_tokens_decoder");
        if (added != null) {
            for (Map.Entry<String, Object> en : added.entrySet()) {
                String content = Json.str(en.getValue(), "content", null);
                if (content != null && !Json.bool(en.getValue(), "normalized", true) && !content.isEmpty()) {
                    literal.put(content, Integer.parseInt(en.getKey()));
                }
            }
        }
        if (literal.isEmpty()) {
            literalPattern = null;
        } else {
            List<String> toks = new ArrayList<>(literal.keySet());
            toks.sort((x, y) -> y.length() - x.length());
            StringBuilder re = new StringBuilder();
            for (String t : toks) {
                re.append(re.isEmpty() ? "" : "|").append(Pattern.quote(t));
            }
            literalPattern = Pattern.compile(re.toString());
        }
        long max = Json.num(cfg, "model_max_length", 77);
        maxLength = max > 0 && max < 100_000 ? (int) max : 77;
    }

    /** Token ids of {@code text}: begin-of-text, the words, end-of-text, padded or truncated to {@code length}. */
    public int[] encode(String text, int length) {
        List<Integer> ids = new ArrayList<>();
        if (literalPattern == null) {
            words(text, ids);
        } else {
            Matcher lm = literalPattern.matcher(text);
            int at = 0;
            while (lm.find()) {
                words(text.substring(at, lm.start()), ids);
                ids.add(literal.get(lm.group()));
                at = lm.end();
            }
            words(text.substring(at), ids);
        }
        int[] out = new int[length];
        java.util.Arrays.fill(out, pad);
        out[0] = bos;
        int n = Math.min(ids.size(), length - 2);
        for (int i = 0; i < n; i++) {
            out[1 + i] = ids.get(i);
        }
        out[1 + n] = eos;
        return out;
    }

    public int[] encode(String text) {
        return encode(text, maxLength);
    }

    /** Normalises a stretch of text, cuts it into words and appends the BPE ids of the words. */
    private void words(String text, List<Integer> ids) {
        String norm = Normalizer.normalize(text, Normalizer.Form.NFC).replaceAll("\\s+", " ").toLowerCase(Locale.ROOT).strip();
        Matcher m = WORDS.matcher(norm);
        while (m.find()) {
            String w = m.group();
            if (w.equals("<|startoftext|>")) {
                ids.add(bos);
            } else if (w.equals("<|endoftext|>")) {
                ids.add(eos);
            } else {
                for (int id : bpe(w)) {
                    ids.add(id);
                }
            }
        }
    }

    private int[] bpe(String word) {
        int[] hit = cache.get(word);
        if (hit != null) {
            return hit;
        }
        byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
        List<String> sym = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            sym.add(byteChar[b & 0xFF]);
        }
        sym.set(sym.size() - 1, sym.get(sym.size() - 1) + "</w>");
        while (sym.size() > 1) {
            int best = Integer.MAX_VALUE;
            int at = -1;
            for (int i = 0; i + 1 < sym.size(); i++) {
                Integer r = ranks.get(sym.get(i) + " " + sym.get(i + 1));
                if (r != null && r < best) {
                    best = r;
                    at = i;
                }
            }
            if (at < 0) {
                break;
            }
            // merge every occurrence of the best pair, left to right (as the reference implementation does)
            String a = sym.get(at);
            String b = sym.get(at + 1);
            List<String> next = new ArrayList<>(sym.size());
            for (int i = 0; i < sym.size(); i++) {
                if (i + 1 < sym.size() && sym.get(i).equals(a) && sym.get(i + 1).equals(b)) {
                    next.add(a + b);
                    i++;
                } else {
                    next.add(sym.get(i));
                }
            }
            sym = next;
        }
        int[] ids = new int[sym.size()];
        for (int i = 0; i < ids.length; i++) {
            Integer id = vocab.get(sym.get(i));
            ids[i] = id == null ? eos : id; // cannot happen with a consistent vocabulary
        }
        cache.put(word, ids);
        return ids;
    }
}
