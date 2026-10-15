package org.robomiri.jllm.model;

/**
 * The DeepSeek-V2 / V3 expert router ({@code MoEGate} of modeling_deepseek.py), for one token.
 *
 * <ol>
 *   <li>{@code scores = softmax(logits)} (V2) or {@code sigmoid(logits)} (V3)</li>
 *   <li>selection, by {@code topk_method}:
 *     <ul>
 *       <li>{@code greedy}: the {@code topK} highest scores</li>
 *       <li>{@code group_limited_greedy}: the experts are split into {@code nGroup} consecutive groups; a group's
 *           score is the maximum score in it; the {@code topkGroup} best groups are kept and the {@code topK}
 *           highest scores inside them win</li>
 *       <li>{@code noaux_tc}: like the above on {@code scores + e_score_correction_bias}, with the group score
 *           being the sum of its two best corrected scores; the weights are still the uncorrected scores</li>
 *     </ul></li>
 *   <li>if {@code topK > 1 && norm_topk_prob}: weights /= (sum + 1e-20)</li>
 *   <li>times {@code routed_scaling_factor} (V3: always; V2: only when not renormalized)</li>
 * </ol>
 * Ties go to the lower index. The Metal kernel {@code router_ds} implements the same rule.
 */
public final class DeepSeekRouter {

    private DeepSeekRouter() {
    }

    /**
     * @param logits   router logits {@code [nExperts]}; overwritten with the scores
     * @param bias     {@code e_score_correction_bias} {@code [nExperts]}, only read for {@code noaux_tc}
     * @param idx      receives the {@code topK} selected experts
     * @param weight   receives their routing weights (scaling factor already applied)
     * @param scratch  {@code [nExperts]} floats of working space
     */
    public static void select(float[] logits, float[] bias, DeepSeekConfig d, int topK, boolean normTopK,
                              int[] idx, float[] weight, float[] scratch) {
        int e = logits.length;
        if (d.sigmoid()) {
            for (int i = 0; i < e; i++) {
                logits[i] = 1f / (1f + (float) Math.exp(-logits[i]));
            }
        } else {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < e; i++) {
                max = Math.max(max, logits[i]);
            }
            float sum = 0;
            for (int i = 0; i < e; i++) {
                logits[i] = (float) Math.exp(logits[i] - max);
                sum += logits[i];
            }
            for (int i = 0; i < e; i++) {
                logits[i] /= sum;
            }
        }
        float[] scores = logits;
        float[] choice = scratch; // what the selection looks at
        boolean noaux = d.topkMethod().equals("noaux_tc");
        for (int i = 0; i < e; i++) {
            choice[i] = noaux ? scores[i] + bias[i] : scores[i];
        }
        if (d.hasGroups()) {
            int gs = e / d.nGroup();
            // group scores, then keep the topkGroup best groups
            float[] gscore = new float[d.nGroup()];
            for (int g = 0; g < d.nGroup(); g++) {
                int o = g * gs;
                if (noaux) {
                    float b1 = Float.NEGATIVE_INFINITY;
                    float b2 = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < gs; i++) {
                        float v = choice[o + i];
                        if (v > b1) {
                            b2 = b1;
                            b1 = v;
                        } else if (v > b2) {
                            b2 = v;
                        }
                    }
                    gscore[g] = gs > 1 ? b1 + b2 : b1;
                } else {
                    float b1 = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < gs; i++) {
                        b1 = Math.max(b1, choice[o + i]);
                    }
                    gscore[g] = b1;
                }
            }
            boolean[] keep = new boolean[d.nGroup()];
            for (int k = 0; k < d.topkGroup(); k++) {
                int best = -1;
                for (int g = 0; g < d.nGroup(); g++) {
                    if (!keep[g] && (best < 0 || gscore[g] > gscore[best])) {
                        best = g;
                    }
                }
                keep[best] = true;
            }
            for (int i = 0; i < e; i++) {
                if (!keep[i / gs]) {
                    choice[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }
        float selected = 0;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bv = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < e; i++) {
                if (choice[i] > bv) {
                    bv = choice[i];
                    best = i;
                }
            }
            idx[k] = best;
            weight[k] = scores[best];
            selected += weight[k];
            choice[best] = Float.NEGATIVE_INFINITY;
        }
        boolean renorm = topK > 1 && normTopK;
        if (renorm) {
            float denom = selected + 1e-20f;
            for (int k = 0; k < topK; k++) {
                weight[k] /= denom;
            }
        }
        if (renorm ? d.scaleAlways() : true) {
            for (int k = 0; k < topK; k++) {
                weight[k] *= (float) d.routedScale();
            }
        }
    }
}
