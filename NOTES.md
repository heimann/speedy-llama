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

## Phase 2/3 decisions

- Arch string: "cohere2moe" (style of "glm4moe").
- Tokenizer: pre-tokenizer hash was unknown, but the tokenizer.json regexes
  match LLAMA_VOCAB_PRE_TYPE_TINY_AYA exactly (digit thousands-grouping
  split + case-aware word split) - North shares Cohere's new tokenizer
  family. Added North's chkhsh mapping to res="tiny_aya" in base.py.
- Model's tokenizer_config.json declares tokenizer_class
  "TokenizersBackend" (transformers v5 only; llama.cpp pins 4.57.6).
  Workaround: ~/code/north-mini-src/ is a symlink dir over the HF snapshot
  with tokenizer_config.json patched to PreTrainedTokenizerFast.
- RoPE type: HF modeling uses interleaved/GPT-J rotate_half ("different
  from e.g. Llama" comment) -> LLAMA_ROPE_TYPE_NORM group (same as
  cohere2), no q/k permutation in converter.
- Converter swaps intermediate_size (768, experts) out for
  prefix_dense_intermediate_size (3072) so feed_forward_length carries the
  dense width and expert_feed_forward_length carries the expert width.
- build_moe_ffn: exp_probs_b=nullptr, norm_w=false, w_scale=0.0 (default
  hparams.expert_weights_scale; 0 means "no scaling" in build_moe_ffn).
- bf16 GGUF verified: 442 tensors, experts fused to [2048,768,128] (ggml
  dim order), gating_func=2 (sigmoid), all metadata keys correct.
- llama-model-saver.cpp blacklists COHERE2; added COHERE2_MOE alongside.

## Phase 4 result

- bf16 GGUF coherent on FIRST RUN, no debugging needed. CPU (-ngl 0),
  greedy: correct iterative + recursive linked-list reversal with clean
  markdown and docstrings. Prompt 33.3 t/s, generation 8.2 t/s.
- Model emits visible "We need to..." planning prose before the answer -
  North is thinking-trained (<|START_THINKING|> tokens in the template).
  Worth checking later whether llama.cpp's reasoning parser should split
  this into a separate channel; cosmetic, not architectural.
- llama-cli quirk: -no-cnv still entered conversation mode and looped on
  "> " with EOF stdin; use -st for scripted single-turn runs.

## Phase 5/6 results

- Quants: Q4_K_M 18.6GB / Q5_K_M 21.7GB / Q6_K 25GB (~2.5 min each).
- Q4_K_M fully offloaded on the 4090: 230 t/s gen, ~4000 t/s pp.
- Turn termination: stops naturally on <|END_OF_TURN_TOKEN|> (eos
  255001 is auto-EOG; <|END_OF_TURN_TOKEN|> is not in vocab.cpp's
  name-match list, the eos id carries it).
- Needle recall at ~14k tokens (needle at position 0, window 4096):
  exact recall -> global NoPE layers + iSWA pattern verified at range.
- Tool-use prompt: correctly structured JSON call.
- Perplexity, 16x512 chunks of llama.cpp source: bf16 1.7880 +/- 0.047,
  Q4_K_M 1.8109 +/- 0.048 (+1.3%) - normal Q4_K_M degradation.
- At temp 1.0 the model sometimes re-enters analysis and emits literal
  gpt-oss Harmony strings ("<|channel|>analysis<|message|>") - training
  data leakage in the model itself, not a port bug (tokens are spelled
  out as plain text; they don't exist in Cohere's vocab).

## Phase 7 postmortem (vs PR #24260)

### Converged exactly (independently identical decisions)

- Arch string "cohere2moe", enum placement, registry entry in
  conversion/__init__.py -> command_r module, class name Cohere2MoeModel,
  file name src/models/cohere2-moe.cpp, even LLM_TYPE_30B_A3B for 49 layers.
- Graph math is identical for this checkpoint: RMSNorm, parallel
  attn+FFN block, rope iff (is_swa || il < n_layer_dense_lead) - their
  force_rope is literally the same condition - sigmoid build_moe_ffn with
  exp_probs_b=null, no top-k norm, logit_scale on output, tied embeddings.
- Same expert stacking (buffer + torch.stack dim=0 + merged names), same
  rope_dimension_count=head_dim, rope scaling NONE, NORM rope group.
- Default gating to SIGMOID when key absent: same code.
- Their C++ scalar-pattern fallback is set_swa_pattern(swa_period, true) -
  same dense_first call; their loader would load our GGUF correctly.

### What they did that we didn't

1. **Per-layer SWA pattern array**: they write layer_types as a bool array
   (add_sliding_window_pattern([...])) and read it via get_arr into
   is_swa_impl, scalar+dense-first only as fallback. Faithful to arbitrary
   patterns; needed a get_arr<std::array<uint32_t,512>> template
   instantiation in llama-model-loader.cpp. We bake in strict 1:3
   periodicity (validated at conversion, so safe but less general).
2. **MTP/NextN support**: full multi-token-prediction path (graph_mtp,
   trunk-only/mtp-only loading, converter filters). Our checkpoint's
   config has no num_nextn_predict_layers, so we never saw this; either
   another North variant has it or they ported it speculatively.
3. **Shared experts**: supported, including the "average" combination
   implemented as (routed + shared) * 0.5. We raise on
   num_shared_experts != 0 (correct for this checkpoint, less general).
4. **Fused gate_up experts + quant sidecar scales**: they plumb
   ffn_gate_up_exps and all the *_s scale tensors through
   build_ffn/build_moe_ffn (NVFP4-era infrastructure), and even patched
   dense cohere2.cpp to pass _s tensors.
5. **tests/test-llama-archs.cpp**: synthetic arch test registration
   (with moe_mandatory). We didn't know this harness existed - the real
   miss of the exercise; it's the project's standard for new archs.
6. They read LLM_KV_ROPE_FREQ_BASE_SWA (optional) like cohere2 does; we
   dropped that read. No-op for this model, faithful for hypothetical ones.

### Post-postmortem correction (learned the hard way)

Their `_set_vocab_gpt2` override - which I originally filed under "unclear
why" - fixes a real bug we shipped: tokenizer_config.json carries LEGACY
chat templates (R7B-era, using <|START_RESPONSE|> markers that don't exist
in North's vocab), while chat_template.jinja is the trained format (thinking
scaffolded by real <|START_THINKING|> tokens). Our GGUF embedded the legacy
template; the model ran off-distribution: thinking dumped as plain text,
gpt-oss-style "<|channel|>analysis" loops on conversational prompts,
occasional failure to terminate. Symptom did not show on single-shot code
prompts (all our smoke tests) - only in multi-turn chat. Fixed by adopting
their override + patching shipped GGUFs via gguf_new_metadata.py. With the
correct template, llama-cli parses thinking into a proper reasoning channel
(rendered dim) and the loops disappear. Lesson: smoke tests must include
multi-turn conversational prompts, not just one-shot tasks.

### Where our version is arguably better

- Tokenizer: we mapped North's chkhsh to the existing "tiny_aya" pre
  (no C++ change). They bypass hashing with a hardcoded
  get_vocab_base_pre -> "cohere2-moe" + new C++ vocab branch aliasing
  TINY_AYA. Ours is less code; theirs is immune to tokenizer-file drift.
- They hardcode add_expert_gating_func(SIGMOID) ignoring
  expert_selection_fn; we map sigmoid/softmax and raise on unknown.
  A softmax-routed cohere2_moe checkpoint would silently misroute
  with their converter.
- Their base.py change adds prefix_dense_intermediate_size to the global
  find_hparam list for feed_forward_length (affects every arch); we
  swapped hparams locally in our model class only.

### Verdict

Blind implementation was functionally correct and byte-level compatible
with their loader for this checkpoint (their fallback path). We missed
generality (pattern array, shexp, MTP) and the arch test harness, not
correctness. Possible upstream-worthy observations (David's call, in his
own words per repo policy): the hardcoded SIGMOID vs expert_selection_fn,
and the global find_hparam edit.

## Upstream-worthy findings (report in David's own words, per repo policy)

1. Their converter hardcodes SIGMOID gating, ignoring expert_selection_fn -
   a softmax-routed cohere2_moe checkpoint would silently misroute (PR
   #24260 review note).
2. llama-cli never copies chat_params.preserved_tokens into
   sampling.preserved_tokens (the /chat/completions path does, in
   server-task.cpp). Any template whose parser tags are control tokens
   (Cohere <|START_THINKING|>/<|END_THINKING|>, etc.) streams everything as
   reasoning in the CLI: thinking never "ends", answer renders dim with no
   newline. Fixed in this fork (tools/cli/cli.cpp).
3. llama-cli also force-overrides --reasoning-format to DEEPSEEK, and the
   PEG parser hard-crashes (std::runtime_error -> terminate) on truncated
   or grammar-mismatched output instead of degrading gracefully.

## Dead ends / incidents

- First setup attempt crashed the machine: MCE hardware error on CPU 8
  (13900K) during -j30 CUDA build. Rebuilt with -j12, completed fine.
  Throttle all heavy parallel work on voyager.
- The crash also silently truncated 2 safetensors shards and emptied
  model.safetensors.index.json even though `hf download` exited 0. Caught
  by checking every file size against hub metadata. Lesson: after a hard
  crash, size-verify the whole HF cache, don't trust the CLI.
- Incremental build after the 2228-commit master jump left
  libggml-cuda.so missing new flash-attn template instances (stale
  configure state). Required rm -rf build + clean rebuild.
