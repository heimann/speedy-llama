```
                          ▄▄
                      ▄▄ █▀ █▄
   ≡≡≡≡≡≡≡≡      ▄▄▄▄▄▄▄▄█  ▄▀
    ≡≡≡≡≡≡≡≡≡ ▄██████████████
     ≡≡≡≡≡≡≡  ██▀ ▀██▀▀▀▀▀██▀
              ▀▀   ▀▀     ▀▀
              s p e e d y - l l a m a
```

# Speedy Llama

Experimental [llama.cpp](https://github.com/ggml-org/llama.cpp) fork carrying model architectures that haven't merged upstream yet.

Think of this repository as a learning platform designed to see if I can get quantized models running on 4090/3090-tier NVIDIA GPUs. First priority is compatibility, second priority is speed.

Currently adds:

- **cohere2_moe** (Cohere North-Mini-Code-1.0, 30B-A3B MoE) — [Q4_K_M and Q5_K_M quants](https://huggingface.co/heimann/North-Mini-Code-1.0-GGUF) run fully offloaded on a single 24GB card at 200+ tok/s.

Over time the goal is to layer in speed work: draft-model speculation defaults, 24GB-targeted offload presets, and similar.

Build and usage identical to upstream llama.cpp ([upstream README](README-upstream.md)).
