package org.robomiri.jllm;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.GemmaMath;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.Scaling;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.robomiri.jllm.SmokeTest.check;
import static org.robomiri.jllm.SmokeTest.decoder;
import static org.robomiri.jllm.SmokeTest.gqa;
import static org.robomiri.jllm.SmokeTest.rope;

/**
 * Gemma 2 / 3 checks that need no model: the scalar math against hand-computed values and the config parsing
 * against the real Hub configs (unsloth/gemma-2-2b-it, unsloth/gemma-3-1b-it, unsloth/gemma-3-4b-it).
 */
final class GemmaChecks {

    private GemmaChecks() {
    }

    static void run() throws Exception {
        math();
        configs();
    }

    static void math() {
        // gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). By hand for x = 1:
        // u = 0.7978845608 * 1.044715 = 0.8335696, tanh(u) = 0.6823837, so 0.5 * 1.6823837 = 0.8411919.
        check(Math.abs(GemmaMath.geluTanh(1f) - 0.8411919f) < 1e-5f, "gelu_tanh(1) = " + GemmaMath.geluTanh(1f));
        check(Math.abs(GemmaMath.geluTanh(-1f) + 0.1588081f) < 1e-5f, "gelu_tanh(-1) = " + GemmaMath.geluTanh(-1f));
        check(Math.abs(GemmaMath.geluTanh(3f) - 2.9963627f) < 1e-5f, "gelu_tanh(3) = " + GemmaMath.geluTanh(3f));
        check(GemmaMath.geluTanh(0f) == 0f && Math.abs(GemmaMath.geluTanh(10f) - 10f) < 1e-6f, "gelu_tanh limits");

        // soft-capping: 50 * tanh(100 / 50) = 50 * 0.96402758 = 48.201379; small inputs pass through; cap 0 = off
        check(Math.abs(GemmaMath.softcap(100f, 50f) - 48.201379f) < 1e-4f, "softcap(100, 50)");
        check(Math.abs(GemmaMath.softcap(0.5f, 50f) - 0.49998333f) < 1e-6f && GemmaMath.softcap(7f, 0f) == 7f,
                "softcap small / off");

        // RMSNorm with gain (1 + w): x = [1 2 3 4], w = [0 0.5 -0.5 1], eps = 0.
        // mean(x^2) = 30/4 = 7.5, rsqrt = 0.36514837, gain = [1 1.5 0.5 2]
        float[] x = {1, 2, 3, 4};
        float[] w = {0, 0.5f, -0.5f, 1};
        float[] y = GemmaMath.rmsnorm(x, GemmaMath.foldNormGain(w), 0f);
        float[] expect = {0.36514837f, 1.0954451f, 0.5477226f, 2.9211870f};
        boolean ok = true;
        for (int i = 0; i < 4; i++) {
            ok &= Math.abs(y[i] - expect[i]) < 1e-6f;
        }
        check(ok, "gemma rmsnorm (1 + w) against hand values");
        check(GemmaMath.foldNormGain(new float[]{0, -1, 0.25f})[1] == 0f, "folded gain 1 + w");
    }

    static void configs() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-gemma-cfg");
        try {
            // unsloth/gemma-2-2b-it
            Architecture g2 = cfg(tmp, """
                    {"architectures":["Gemma2ForCausalLM"],"attention_bias":false,"attn_logit_softcapping":50.0,
                     "bos_token_id":2,"eos_token_id":[1,107],"final_logit_softcapping":30.0,"head_dim":256,
                     "hidden_act":"gelu_pytorch_tanh","hidden_activation":"gelu_pytorch_tanh","hidden_size":2304,
                     "intermediate_size":9216,"max_position_embeddings":8192,"model_type":"gemma2",
                     "num_attention_heads":8,"num_hidden_layers":26,"num_key_value_heads":4,
                     "query_pre_attn_scalar":256,"rms_norm_eps":1e-06,"rope_theta":10000.0,"sliding_window":4096,
                     "torch_dtype":"bfloat16","vocab_size":256000}""");
            LayerSpec l0 = decoder(g2).layers().getFirst();
            check(g2.core().modelType().equals("gemma2") && !gqa(g2, 0).qkNorm() && l0.norms() == LayerSpec.Norms.SANDWICH_GEMMA
                    && l0.mlp() instanceof Mlp.Dense d0 && d0.act() == Mlp.Activation.GELU_TANH && g2.core().normPlusOne(),
                    "gemma2 basics");
            check(g2.core().tiedEmbeddings(), "gemma2 ties embeddings although the config omits the key");
            check(gqa(g2, 0).softcap() == 50f && g2.core().finalSoftcap() == 30f && gqa(g2, 0).scale() == 0.0625f,
                    "gemma2 caps and scale 1/16");
            check(gqa(g2, 0).window() == 4096 && gqa(g2, 1).window() == 0 && gqa(g2, 24).window() == 4096 && gqa(g2, 25).window() == 0,
                    "gemma2 even layers are local");
            check(decoder(g2).ropeTables().size() == 1 && gqa(g2, 0).ropeTable() == 0 && g2.core().embedScale() == 48f,
                    "gemma2 embed scale sqrt(2304) = 48, bos 2, one rope table");
            check(g2.core().eosTokenIds().equals(List.of(1, 107)) && gqa(g2, 0).headDim() == 256 && g2.core().vocabSize() == 256000,
                    "gemma2 ids and shapes");

            // unsloth/gemma-3-1b-it (transformers 4.51: no layer_types, sliding_window_pattern 6)
            Architecture g3 = cfg(tmp, """
                    {"architectures":["Gemma3ForCausalLM"],"attention_bias":false,"attn_logit_softcapping":null,
                     "bos_token_id":2,"eos_token_id":106,"final_logit_softcapping":null,"head_dim":256,
                     "hidden_activation":"gelu_pytorch_tanh","hidden_size":1152,"intermediate_size":6912,
                     "max_position_embeddings":32768,"model_type":"gemma3_text","num_attention_heads":4,
                     "num_hidden_layers":26,"num_key_value_heads":1,"query_pre_attn_scalar":256,"rms_norm_eps":1e-06,
                     "rope_local_base_freq":10000,"rope_scaling":null,"rope_theta":1000000,"sliding_window":512,
                     "sliding_window_pattern":6,"torch_dtype":"bfloat16","vocab_size":262144}""");
            check(gqa(g3, 0).qkNorm() && gqa(g3, 0).softcap() == 0f && g3.core().finalSoftcap() == 0f
                    && rope(g3).scaling() instanceof Scaling.None && rope(g3).theta() == 1e6 && g3.core().vocabSize() == 262144
                    && decoder(g3).ropeTables().size() == 2 && decoder(g3).ropeTables().get(1).theta() == 1e4, "gemma3 1b basics");
            boolean pat = true;
            for (int l = 0; l < 26; l++) {
                boolean global = l % 6 == 5;
                pat &= (gqa(g3, l).window() == (global ? 0 : 512)) && ((gqa(g3, l).ropeTable() == 1) == !global);
            }
            check(pat, "gemma3 5 local : 1 global (layers 5, 11, 17, 23 global)");
            // sqrt(1152) = 33.941 rounds to 34.0 in bfloat16 (8 significant bits), as transformers' normalizer does
            check(g3.core().embedScale() == 34.0f, "gemma3 embed scale bf16(sqrt(1152)) = 34");

            // unsloth/gemma-3-4b-it: multimodal wrapper, decoder settings in text_config
            Architecture g4 = cfg(tmp, """
                    {"architectures":["Gemma3ForConditionalGeneration"],"boi_token_index":255999,"bos_token_id":2,
                     "eoi_token_index":256000,"eos_token_id":106,"image_token_index":262144,"mm_tokens_per_image":256,
                     "model_type":"gemma3","pad_token_id":0,
                     "text_config":{"attention_bias":false,"attn_logit_softcapping":null,"head_dim":256,
                       "hidden_activation":"gelu_pytorch_tanh","hidden_size":2560,"intermediate_size":10240,
                       "max_position_embeddings":131072,"model_type":"gemma3_text","num_attention_heads":8,
                       "num_hidden_layers":34,"num_key_value_heads":4,"query_pre_attn_scalar":256,"rms_norm_eps":1e-06,
                       "rope_local_base_freq":10000.0,"rope_scaling":{"factor":8.0,"rope_type":"linear"},
                       "rope_theta":1000000.0,"sliding_window":1024,"sliding_window_pattern":6,
                       "torch_dtype":"bfloat16","vocab_size":262208},
                     "torch_dtype":"bfloat16","vision_config":{"hidden_size":1152,"model_type":"siglip_vision_model"}}""");
            check(g4.core().modelType().equals("gemma3_text") && g4.core().dim() == 2560 && g4.core().layers() == 34
                    && g4.core().vocabSize() == 262208
                    && gqa(g4, 0).window() == 1024 && gqa(g4, 5).window() == 0 && gqa(g4, 33).window() == 1024
                    && g4.core().bosTokenId() == 2 && g4.core().eosTokenIds().equals(List.of(106)), "gemma3 4b text_config");
            check(rope(g4).scaling() instanceof Scaling.Linear lin && lin.factor() == 8.0
                    && gqa(g4, 0).ropeTable() == 1 && gqa(g4, 5).ropeTable() == 0
                    && decoder(g4).ropeTables().get(1).scaling() instanceof Scaling.None, "gemma3 4b linear rope x8 global, plain local");

            // transformers >= 5 spelling: layer_types and rope_parameters keyed by layer type
            Architecture g5 = cfg(tmp, """
                    {"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text","hidden_size":64,
                     "intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
                     "head_dim":32,"vocab_size":100,"sliding_window":8,"bos_token_id":2,"eos_token_id":[1,106],
                     "layer_types":["sliding_attention","full_attention","full_attention","sliding_attention"],
                     "rope_parameters":{"full_attention":{"rope_type":"linear","factor":4.0,"rope_theta":1000000.0},
                                        "sliding_attention":{"rope_type":"default","rope_theta":10000.0}}}""");
            check(gqa(g5, 0).window() == 8 && gqa(g5, 1).window() == 0 && gqa(g5, 2).window() == 0 && gqa(g5, 3).window() == 8
                    && rope(g5).theta() == 1e6 && rope(g5).scaling() instanceof Scaling.Linear l5 && l5.factor() == 4.0
                    && gqa(g5, 3).ropeTable() == 1, "gemma3 layer_types / rope_parameters");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    private static Architecture cfg(Path dir, String json) throws java.io.IOException {
        Files.writeString(dir.resolve("config.json"), json);
        return Architecture.load(dir);
    }
}
