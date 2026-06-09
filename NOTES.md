# cohere2_moe DIY port - notes

DIY implementation of `cohere2_moe` (North-Mini-Code-1.0) for llama.cpp,
done blind (no peeking at PR #24260 until Phase 7).

Sources used: model config.json, safetensors index, HF transformers
`modeling_cohere2_moe.py` (the model's reference implementation - fair game;
only the llama.cpp PR is off-limits).

NOTE: llama.cpp's AGENTS.md prohibits AI-generated upstream contributions
(private forks exempt). This branch is a private learning exercise and must
never become an upstream PR.

## Hyperparameter mapping

| config.json | value | GGUF key / handling |
|---|---|---|
| architectures | Cohere2MoeForCausalLM | converter registration |
| hidden_size | 2048 | embedding_length |
| num_hidden_layers | 49 | block_count |
| num_attention_heads | 32 | attention.head_count |
| num_key_value_heads | 4 | attention.head_count_kv (GQA) |
| head_dim | 128 | attention.key_length/value_length (NB: 32*128=4096 != 2048, must be written explicitly) |
| intermediate_size | 768 | expert_feed_forward_length |
| prefix_dense_intermediate_size | 3072 | feed_forward_length (dense lead layer only) |
| first_k_dense_replace | 1 | leading_dense_block_count |
| num_experts | 128 | expert_count |
| num_experts_per_tok | 8 | expert_used_count |
| expert_selection_fn | "sigmoid" | expert_gating_func = SIGMOID |
| norm_topk_prob | false | expert_weights_norm = false |
| num_shared_experts | 0 | no shexp tensors |
| layer_types | full@(il%4==0) | sliding_window_pattern=4, C++ set_swa_pattern(4, dense_first=true) |
| sliding_window | 4096 | attention.sliding_window |
| prefix_dense_sliding_window_pattern | 1 | force RoPE on dense lead layer (see RoPE rule below) |
| rope_theta | 50000 | rope.freq_base |
| rope_scaling | null | rope_scaling_type NONE |
| max_position_embeddings | 500000 | context_length |
| logit_scale | 1.0 | logit_scale (no-op at 1.0 but wired like cohere2) |
| rms_norm_eps | 1e-6 | attention.layer_norm_rms_epsilon - model uses RMSNorm! |
| layer_norm_eps | 1e-5 | UNUSED (rms_norm_eps non-null wins, per modeling code) |
| vocab_size | 262144 | vocab_size |
| attention_bias / use_qk_norm | false | no bias / qk-norm tensors |
| use_parallel_block | true | parallel attn+FFN graph (like cohere2) |
| hidden_act + use_gated_activation | silu + true | SwiGLU (LLM_FFN_SILU, gated) |
| shared_expert_combination_strategy | "average" | irrelevant (0 shared experts) |
| bos/eos/pad | 2 / 255001 / 0 | vocab metadata |

## Key findings from modeling code

1. **RMSNorm, not LayerNorm.** Break from Cohere tradition. Decoder layer:
   single pre-norm (RMS, eps 1e-6), then parallel attn + mlp on the normed
   input, output = residual + attn + mlp. Same block shape as cohere2
   otherwise (which uses LLM_NORM; we use LLM_NORM_RMS).
2. **Router**: plain linear (no bias), top-k FIRST then sigmoid on selected
   logits, norm_topk_prob=false, no routed scaling factor. Sigmoid is
   monotonic, so llama.cpp's sigmoid-then-topk in build_moe_ffn is exactly
   equivalent. Use gating_op=SIGMOID, norm_w=false, w_scale=1 (0/unset).
3. **RoPE rule**: `apply rope iff (sliding layer) OR force_rope`, where
   force_rope = dense-lead layer && prefix_dense_sliding_window_pattern==1.
   So layer 0 is FULL attention WITH RoPE; layers 4,8,...,48 are full NoPE;
   sliding layers RoPE. Our C++ bakes in: rope iff (is_swa(il) || il <
   n_layer_dense_lead). Caveat: assumes prefix pattern==1; fine for this
   checkpoint.
4. **SWA phase differs from cohere2**: North has full attention at il%4==0
   (full first); cohere2/R7B has full at il%4==3. llama.cpp's
   set_swa_pattern has a dense_first flag for exactly this -> pass true.
5. **Tied embeddings** confirmed: no lm_head in checkpoint.
6. Final logits scaled by logit_scale (=1.0 here, harmless).

## Checkpoint tensor inventory (18730 tensors, checks out)

- `model.embed_tokens.weight`, `model.norm.weight`
- per layer: `input_layernorm`, `self_attn.{q,k,v,o}_proj` (no biases)
- layer 0 (dense): `mlp.{gate,up,down}_proj` (n_ff 3072)
- layers 1-48 (MoE): `mlp.gate.weight` [128, 2048] router (-> ffn_gate_inp),
  `mlp.experts.{0..127}.{gate,up,down}_proj` stored INDIVIDUALLY ->
  converter must stack into 3D fused tensors (deepseek.py pattern:
  buffer per layer, torch.stack(dim=0), emit merged name)
- No router bias (no exp_probs_b), no shared experts, q[4096,2048] k/v[512,2048]

## Donor archs

- Attention/graph skeleton: `src/models/cohere2.cpp` (parallel block, rope
  on swa only, tied output, logit scale, build_attn_inp_kv_iswa)
- MoE side: `src/models/glm4-moe.cpp` (sigmoid gating, n_layer_dense_lead
  dense/moe split in both tensor creation and graph; ignore its shexp/nextn/
  post-norm extras). build_moe_ffn(no-bias overload) with exp_probs_b=null.
- Codebase was refactored into per-model classes: new arch = new
  src/models/<name>.cpp with load_arch_hparams/load_arch_tensors/
  build_arch_graph + entry in models.h + dispatch in llama-model.cpp +
  arch tables in llama-arch.{h,cpp}. Converter classes live in conversion/
  (ours goes next to Cohere2Model in conversion/command_r.py).

## Plan

- Phase 2: gguf-py constants (MODEL_ARCH.COHERE2MOE, arch name
  "cohere2moe", tensor list incl. FFN_GATE_INP/FFN_*_EXPS; keys all exist
  already), converter class Cohere2MoeModel (subclass TextModel; set params
  per table; deepseek-style expert stacking; skip-zero-bias guard not
  needed - no biases in checkpoint). Convert with --outtype bf16.
- Phase 3: C++ per plan above.
- Watch out: tokenizer is new (262144 vocab vs R7B's 256000) - likely needs
  a new pre-tokenizer hash entry in get_vocab_base_pre; will surface as a
  hard error during conversion.
- rope_dimension_count = head_dim = 128 (no rotary_pct / partial factor in
  config; transformers default = full head_dim).

## Dead ends / incidents

- First setup attempt crashed the machine: MCE hardware error on CPU 8
  (13900K) during -j30 CUDA build. Rebuilt with -j12, completed fine.
  Throttle all heavy parallel work on voyager.
