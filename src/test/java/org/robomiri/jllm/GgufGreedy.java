package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Greedy decoding of a Qwen3-Next / Qwen3.5 GGUF straight from the weights (no engine, no sampler), for comparing with
 * llama.cpp: per prompt it prints the prompt ids, the top-5 log-probabilities of the first position, the greedy ids and
 * text, and the prefill and decode speed.
 *
 * <pre>
 * GgufGreedy model.gguf cpu|metal [-n 32] [--ctx 512] [--batch 512] [--kv f16|f32] MODE:prompt.txt ...
 * </pre>
 * MODE {@code ids}: the ids after the colon, comma separated; {@code raw}: the file's text as it is; {@code chat}: one user message rendered with the model's chat template with
 * thinking off; {@code think}: the same with thinking on. The model is loaded once; every prompt starts at position 0.
 */
public final class GgufGreedy {

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args[0]);
        String backend = args[1];
        int n = 32;
        int ctx = 512;
        int batch = 512;
        KvDtype kvd = KvDtype.F16;
        boolean int8 = false;
        boolean mtp = false;
        List<String> prompts = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "-n" -> n = Integer.parseInt(args[++i]);
                case "--ctx" -> ctx = Integer.parseInt(args[++i]);
                case "--batch" -> batch = Integer.parseInt(args[++i]);
                case "--kv" -> kvd = KvDtype.parse(args[++i]);
                case "--mtp" -> mtp = true; // feed the multi-token-prediction head and count how often its guess is the next token
                case "--int8" -> int8 = true; // CPU: int8 activations (Q8_0 x Q8_0 dot products, as ggml does)
                default -> prompts.add(args[i]);
            }
        }
        long t0 = System.nanoTime();
        try (GgufModel g = GgufModel.open(file)) {
            Tokenizer tok;
            try {
                tok = GgufModel.tokenizer(g.file());
            } catch (java.io.IOException e) { // a tiny test file has a vocabulary that is no byte-level BPE: ids only
                System.out.println("no tokenizer: " + e.getMessage());
                tok = null;
            }
            Qwen3NextArch arch = (Qwen3NextArch) g.architecture();
            System.out.println("arch: " + arch);
            try (ModelWeights w = g.weights()) {
                System.out.printf("loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
                try (Backend b = backend.equals("metal")
                        ? new Qwen3NextMetalBackend(w, ctx, batch, System.getProperty("jllm.metal.lib"), kvd)
                        : new Qwen3NextCpuBackend(w, ctx, batch, Runtime.getRuntime().availableProcessors(), kvd, int8)) {
                    System.out.println("backend: " + b.name());
                    if (mtp) {
                        b.enableMtp(true);
                    }
                    for (String spec : prompts) {
                        run(b, tok, arch, spec, n, batch, mtp);
                    }
                }
            }
        }
    }

    private static void run(Backend b, Tokenizer tok, Qwen3NextArch arch, String spec, int n, int batch, boolean mtp)
            throws Exception {
        int colon = spec.indexOf(':');
        String mode = spec.substring(0, colon);
        int[] ids;
        if (mode.equals("ids")) { // the ids as they are: 1,5,17
            ids = Arrays.stream(spec.substring(colon + 1).split(",")).mapToInt(Integer::parseInt).toArray();
        } else if (mode.equals("raw")) {
            ids = tok.encode(Files.readString(Path.of(spec.substring(colon + 1))));
        } else {
            String text = Files.readString(Path.of(spec.substring(colon + 1)));
            ChatTemplate.Format f = ChatTemplate.forModelType(arch.core().modelType(), tok);
            List<ChatTemplate.Message> msgs = List.of(new ChatTemplate.Message("user", text));
            boolean think = mode.equals("think");
            ids = f.renderTokens(tok, msgs, true, think);
            System.out.println("template " + f.name() + " renders:\n" + f.render(msgs, true, think).replace("\n", "\\n"));
        }
        System.out.println("== " + spec + "\nprompt ids (" + ids.length + "): " + Arrays.toString(ids));
        int vocab = arch.core().vocabSize();
        float[] logits = new float[vocab];
        long p0 = System.nanoTime();
        for (int off = 0; off < ids.length; off += batch) {
            int len = Math.min(batch, ids.length - off);
            b.forward(ids, off, len, off, off + len == ids.length ? logits : null);
        }
        double prefill = (System.nanoTime() - p0) / 1e9;
        System.out.printf("prefill %d tokens in %.2fs = %.1f tok/s%n", ids.length, prefill, ids.length / prefill);
        top(logits, tok, 5);
        List<Integer> out = new ArrayList<>();
        int pos = ids.length;
        long d0 = System.nanoTime();
        int drafted = 0;
        int accepted = 0;
        int guess = -1;
        StringBuilder bits = new StringBuilder();
        float[] head = mtp ? new float[vocab] : null;
        for (int i = 0; i < n; i++) {
            int next = argmax(logits);
            if (guess >= 0) {
                drafted++;
                accepted += guess == next ? 1 : 0;
                bits.append(guess == next ? '1' : '0');
            }
            out.add(next);
            if (i + 1 < n) {
                if (mtp) { // the head's guess for the token after next
                    b.mtpDraft(next, head);
                    guess = argmax(head);
                }
                b.forward(new int[]{next}, 0, 1, pos++, logits);
            }
        }
        if (mtp) {
            System.out.println("mtp: " + accepted + " of " + drafted + " guesses were the next token: " + bits);
        }
        double decode = (System.nanoTime() - d0) / 1e9;
        System.out.printf("decode %d tokens in %.2fs = %.2f tok/s%n", n - 1, decode, (n - 1) / decode);
        int[] o = out.stream().mapToInt(Integer::intValue).toArray();
        System.out.println("greedy ids: " + Arrays.toString(o));
        System.out.println("text: " + (tok == null ? "-" : tok.decode(o).replace("\n", "\\n")));
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    /** The {@code k} largest logits as log-probabilities (log-softmax over the whole vocabulary). */
    private static void top(float[] logits, Tokenizer tok, int k) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (float v : logits) {
            sum += Math.exp(v - max);
        }
        double lse = max + Math.log(sum);
        Integer[] idx = new Integer[logits.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Float.compare(logits[y], logits[x]));
        StringBuilder sb = new StringBuilder("top-" + k + " (id logit logprob token):");
        for (int i = 0; i < k; i++) {
            String piece;
            try {
                piece = tok.decode(new int[]{idx[i]}).replace("\n", "\\n");
            } catch (RuntimeException e) {
                piece = "?";
            }
            sb.append(String.format("%n  %d  %.4f  %.4f  %s", idx[i], logits[idx[i]], logits[idx[i]] - lse, piece));
        }
        System.out.println(sb);
    }
}
