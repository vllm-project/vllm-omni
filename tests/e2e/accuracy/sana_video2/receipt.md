# SANA-Video 2.0 small-shape validation receipt

Executed 2026-10-23 against production source `aede5ecd638001fd3653d0c4c38cd3ba9bae64ad`. The accompanying change adds tests and documentation only; `vllm_omni/` is unchanged.

One NVIDIA H20-3e per GPU run (143771 MiB, driver 580.105.08). Alignment/offline and HTTP validation both completed with exit 0. All 10 focused CPU registry/loading tests passed; the combined registry/loading and video-serving run passed all 189 tests. No runtime compatibility adapter or production source patch was used.

Python 3.12.15; PyTorch 2.13.0+cu130; vLLM 0.31.0; Triton 3.7.1; Diffusers 0.40.0; Transformers 5.14.1. The installed Omni distribution metadata is stale (`0.1.dev3103+g8c3d3407d`) and emitted a version warning; imports used the source SHA above.

Measured runner at `37270e02079697b40ce74af62322378748353163`, SHA256: `4b12c9899704e1026e580bbc9a1a573931018282def76235b4eb260ec3fbefaa`. The current runner omits host and scheduler identifiers from generated receipts; this metadata-only change does not alter inference, and the GPU checks were not rerun. Reference: `NVlabs/Sana@e93c883e10730ee5a4a6edf1cbcf501dc4ef753b`. See the recipe for exact setup and invocation commands.

## Inputs and loading

Both tasks: 64x96, 9 frames, 50 steps, CFG 8, seed 42, motion score 10, flow shift 12, 24 FPS. Prompt: `A small red boat sailing across a calm lake.`; explicit negative prompt: `blur, distortion, abrupt cuts`. TI2V uses the generated 80x120 RGB gradient image, SHA256 `d57b5d2412be7203f838da4d36c7847a199318962fbcede4bbbc3b1c2c2d78cd`.

Transformer and Gemma weights are BF16 at execution; VAE is BF16; sampler latents are FP32. Native self-attention follows the release FP32 path. All downloaded LFS files were checked against the pinned Hub SHA256.

| Component | Repository | Revision |
| --- | --- | --- |
| model | `Efficient-Large-Model/SANA-Video_2.0_5B_720p` | `f2d95fa06400f186fd1b077d12c69c8ac58aba48` |
| text | `Efficient-Large-Model/gemma-2-2b-it` | `569d9809d0c8b6722d4d31b5a77a2ec7a400650a` |
| vae | `Efficient-Large-Model/LTX-2.3-Diffusers` | `362acdf779d42e785fb26910c32254e9458e78c2` |

| Weight file | SHA256 |
| --- | --- |
| model/checkpoints/SANA_Video_2.0_5B_720p.pth | `a4ee62595f26fbed5fc3f66641015508d703f15f95c77952b6eb03b16fce56a1` |
| text/model-00001-of-00002.safetensors | `532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf` |
| text/model-00002-of-00002.safetensors | `6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20` |
| text/tokenizer.json | `3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922` |
| text/tokenizer.model | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |
| vae/vae/diffusion_pytorch_model.safetensors | `425f0dfa227dee5d0ff3d9720563370810409a439c302ca74f0f944057ce55c5` |

Native strict checkpoint load: **795 learned keys, 4,466,980,960 parameters**, FP32 checkpoint source. Only `pos_embed` and `y_embedder.y_embedding` are ignored as documented unused inference buffers. The offline worker RPC confirms the automatically selected class is `SanaVideo2Pipeline` and returns the same complete load report. No `model_class_name` is forced.

## Reference comparison

The transformer and both samplers are imported unchanged from the pinned upstream checkout. Only package initializers and the training builder import are bypassed; the registration decorators receive a real mmcv Registry. Parameters are initialized with Accelerate `init_empty_weights`, preserving upstream non-parameter RoPE caches. Xformers is disabled and FP32 attention enabled according to the release path. The shared pinned Gemma decoder is exercised with independently constructed upstream tokenization/instruction/masks; all four conditioning tensors match exactly. The reference VAE is separately loaded from the same pinned weights using Diffusers, including the upstream decoder-upsample config adaptation. This comparison does not run the full upstream training/inference CLI.

Both tasks have **50/50 byte-identical post-update latent tensors**, each `[1,128,2,2,3]`, and byte-identical decoded tensors `[1,3,9,64,96]`. Every per-step maximum/mean absolute error and decoded error is **0**. Saved NPZ arrays contain both trajectories and both decoded outputs (decoded BF16 values are losslessly represented as FP32). A separate byte comparison of the saved arrays also passed.

| Task | Final latent SHA256 | Decoded FP32 representation SHA256 |
| --- | --- | --- |
| t2v | `d94c4c2aecd3cf01aff1ecf2cf1bfd2a71d75a96aa8bae0b685b347464be04c2` | `4112dda415a90192bda120fbc730b0fbc805212bd969985a83a2d3ec346ca6fb` |
| ti2v | `e9c21ec1e8e2fb78de1eb1b40a97532ad41fc3878b23ce0c7fb92f3753ae3a3c` | `13c8f730414fd0d7f646c7b71bde51a31e79f523fe37739b597b15157dc82434` |

Per-update native/reference hashes are equal; each row below has max/mean absolute error 0 for both tasks.

| Update | T2V latent SHA256 (both implementations) | TI2V latent SHA256 (both implementations) |
| --- | --- | --- |
| 1 | `85ad3c49ed8020d0925141cfe0be651df0473c7242f2c685edc71ea4967b211d` | `a8bc2d5ca7ca3904dc5b8574ec11558f7145570553c861c2a9c47a5ef4a17de0` |
| 2 | `35a9a7b5575777fd7843fc186dfec3f7b5687ac7c3c748539e016058d954a96e` | `8b592baebf1e7cdc95910da817765f4ce6be0a57bc990b7f1cf3ba7c9266f15e` |
| 3 | `9a901f57f55e64025ad36d512dbbcce30d89f6c4bbaabff52bbce68072c4db2a` | `9ed08497511c7c099cf602419fae9f4782042dfafa355b1c7f7fa3f49dde7d4e` |
| 4 | `bb21f4cbde6be21b51d2b62f6461f54fbd9c91c9f3358ffbee5618c17ce43793` | `48a2909685df806fcc5e93a0645435894c11e1dc5099f05ff4c6377fc91dd497` |
| 5 | `5ad9bf1d76a68231e4bb498215455912d586d035b3352b0cec5921433faea2ba` | `e496caea01063ecbc0e8f62519dbf5cc0186d36179aec84864ea37f78c180295` |
| 6 | `b291768d349797a5a2b7b2c38b9a1684df03cf67bedcc8f6b2ddbd3a3eb0c35e` | `ddd54ca303ec9b3a95778c76cb6b3ceab93e27983d6ab1bd0be3e74121d6c7ba` |
| 7 | `c3d48398a9b633b58e223a6a7cfdea30a0ff795ea8f41988473a5eb49e1e02ff` | `453e62e9bed964fba61e0e069d4c111f2b88ecb027902d1da026ae6088629a62` |
| 8 | `97986f7864d6c6ca5f42e909092f8fe693ba8bed0b27fb70c160562bd626f852` | `534050bf58e68ebb0d35a15704e577b70970a0ca6f5c51f468e928c2b92e38a5` |
| 9 | `69c5d6e414d812981989a2e76f3e8c4fe7252ce912ef9c903e095f5a1589b4e6` | `fb8bd541cd243a81a52b34d96df350d917d06e7edab5f298fc4c619c0a7172d5` |
| 10 | `7f7610fd42d40a1a47eddcf14f4655f0e08860877a1b4762f0d5c12e0b2b244a` | `35b29e57a39d64b39ddf39e03964bffdacc1fec8a477b7bbbb80b5fab4104eef` |
| 11 | `cf6523f2c89a9c9ecab3ab2c0ac2f58a87f01a2b2f20856669cdd7dac8066cc4` | `91baea06cc009fc8aa1d4d9610b6872c608218844ede1aa1faf5174cc2292d73` |
| 12 | `6c014c875bbc3955144b6495abfdbf2d4651d0129687817d109d966dd864c808` | `87ec3f4bf4e72dd7a6087dd40e61024944838287ff4d4f5c5177fac009f59182` |
| 13 | `b64488777f9405733563023db49e8d7c756b979952f4a029d330b6330d557f8b` | `7fb12414264fc781b796f23fada6b0495f341850d93e9192b75ab863bd92c5e9` |
| 14 | `abf04603eaf9be44939bfe6591033833723cb3159607a573bb35c5da1b30ca9f` | `6fcc890516842ef6007ee9e1b6ac1ee5a2d6094d410771b8e54aeb08fd7d710e` |
| 15 | `f1cd35d9ab117d57b9a0edb02b72220fca25de182979d61b76c89e29dd41e986` | `677a6bf81f64a3b4f20852f06fb8b7f1329087030bf6fa34251f4cf21dac7c20` |
| 16 | `dabc7133e715d37b97281cecbe4dc38dd355acce5c0191c42a93aa758be37f40` | `cab4ec361b286cce0be4dc6773608f22ffe64973bccd49708f83be697c421a8d` |
| 17 | `16013f1916c8387717bb900ec03d88309cf0c87fc49b4f49279aed227ab2c684` | `cbbd0624248e1b0ddc8e113f596e55c3bc6fa22c60bfe4c22ff47cbe3506f346` |
| 18 | `3bb22f82cdf2d30f659b2ccbe9ce0133e550a51ae3d60ef4d90712d962861edc` | `d94a729b20db1aa938edf8a309f0757648880c28f87633ca0e085bc5d431a665` |
| 19 | `94138218c45c07dc935e6f50050c7062a8318c2b504d5637b652559578e29e87` | `6f72c6157c53bd88df74dedc078e5f25bf215d340d6a3bad657237f321e4bcf4` |
| 20 | `e7af01404a3d9a3d9dd8b2273bb7af94d260d85f9234a4c2104c098b5ba0de2e` | `c14daf8eeed171451c5d109d384183194ba94bf3899669c31e987959664fed49` |
| 21 | `87a8842566f0ce22613105e1b733dff30508d7b7838ece31598e984135dd41bd` | `d966878c69ebe21d1f0281a32179cc85360d5cbb383995845bb537acfeb46e5d` |
| 22 | `8864ef9a315f22f864f05a9d39b3197da03d542e0064c82435bb68858239fcad` | `6b1af44ab2f68b997ddd63274f70ca2636e5fb2d49f0e53841a4b7af3310968f` |
| 23 | `f306246c539be2dfc4bd34d751221cdcba7534b6fdc861e36f8ca670ddd17350` | `9d52120d0860f1d59b86fa972b2a7f21e1bbdeb24fdad684cb8b55431077ca54` |
| 24 | `dcf4bcda5fcb57a63ce1bd84f4f850b887c079ddebe27eaa4316d091a6860051` | `38b4253b869ee343ac7cb480b32c087f4c112020439a59f254e9f6d60bbd91f4` |
| 25 | `4de41eb6af1fa73a055964f47c53804212433af4114b4a2e4c457176d2d75493` | `a28f15432b73ee21f8976efcb01d311b6a375e10a0eec318814fcbd7ca451e8e` |
| 26 | `39af390176bfdcd544d66d893c6bc1a940f54ee8ed23d6dfa58ba54c876f9e9a` | `87160a02504b070cd340dcfc4b407a07324c65281e84cef97aa63a5bf497f502` |
| 27 | `d1c8d358a2d3b886078cf2422a8812dcfa1b2cf82847144ddc89c918ba7eedef` | `6a17032d6564582de8c848512da8ab2dca8015ad5df8f4beff488fc235ebfebd` |
| 28 | `d1d62ae91b78ad92264fbd840d8b1e8aba5c712c015d428b1a5c20cdd8cc0bc9` | `ad01da9e753f46e902b635c54a8fb0e7f9bf1bbf85e7cf7f0f676bb487cc7b78` |
| 29 | `483ab87d564630983f9575b396e8d3317850241a95a488e43085422d396ebecc` | `70fb36b2ee1d8d9f7b376950c4d6c9037dee6c0cc2f97b99a1f57aec09624334` |
| 30 | `7dd2dc7c1ab28da340d4f25ba2ce34212dd374f4129f98b5a863f37889912fc7` | `10283162108a12e624b3617d046857678a173929338164f1aef3a3be91468474` |
| 31 | `feb9a247b4e29f2ec0de3b602d78415b46d78198d7f6535bd57f311c3340ed67` | `35ec49c9b2ae33238d239339aaeb872bfa65aaa93a84947a3c3ab7593b308d1d` |
| 32 | `b105b81b78592ee32f60975a4eac4ea5f0892b02e25ed8b7837ad61fc3781d8b` | `6c0e14bc398fcbac1d297a16cd48b0be6e4962e1b13c33dd57767003537bb55e` |
| 33 | `780e413d486ea6038a7fbc327c2f9fffcb4efebd49f1a8bf89d89aa10d1aef76` | `83e2eacad8926031164a21342d941de19f3e340d3737e75df27c4eca926997c8` |
| 34 | `7328c6d4d48927f0afd4270897f5dff731ff36a803add5336269e7ae95de1f09` | `7fe6bb6d0f59a6ee38f04b763c91ab98f9ba589911e3fb7d85f5e1dd006ee96d` |
| 35 | `891745ff2288b0cb7052480bcb12590c2febd0b12b38b7c3db0e768ba28d98e1` | `a64219adfd43f6870bb6aba0255362e075e417fc0f0583cc4820ca6edcc0012e` |
| 36 | `f2b25d43a0f5507bcb0bf0e7ffb93d20e7075bbecee82ba4ce7edcbb47d5a021` | `27b4230c59cfa8fb2125572381916e5c379aba3cecfc6654e62e076a31eda47d` |
| 37 | `59e70a8f7539bd53d818750a831c3e948c5c047ab318b455db6898bfdc1f4201` | `f27907829324b176815393858bf60e90b3d290a7636631179ed0dfd585996b0a` |
| 38 | `985c9d886cb9bfd69a63df151096b32ed61a7837c89b99cfbee9f91fa816a039` | `360a7aaee99f9358d2bed4d833e35867b371ff358d31365b0064396b83b26f24` |
| 39 | `f4877638374cfe97bf1c5467cd3037640df1a63914725a8b192615e8dcb5f9cf` | `cb41e6daedc0f66e730061658f134e0f34d8b7582c5b2acab16adf168b394d15` |
| 40 | `e7a4389163ad36432b6c606ba153ebec93105a93735508c616af1595d6481576` | `757f842b5a0a90c77a90a45c1ae02852e27398b99a3974fb3b6e7b8971b4c376` |
| 41 | `8673ca9af9965adc1efee8654d46c3d3e356622b07eefbb15c7dd92f9862d449` | `fb2f7f67a812c88aa26c52bde3bf1a6cd87d7d2ec94d0e9631b7068dc70e852a` |
| 42 | `ec760b2bb220ef46381cc618a06c0d48f11307f098e838a0316cef1469bbd4a4` | `3b72d084479a1edcff19d77ec416af1d511d5f2a80d75b6dc8453aeefd379359` |
| 43 | `17443876904eeb2716f914f6d5eb72552dcb79f5fd7b7eeac4344add6602962e` | `2c08fe7fa6bc8631fff8ef47cbff01f7f0eeb24b1a0d789cb881dcb8c6efea8c` |
| 44 | `d59fe8ad36292b2ae99e91ac62b9491ea3116039325a0e085323d3295a70680a` | `8a8e5087b82297209f20c4b3dd67ab07f60e3834a2ee0f26588bebdec4f8d7b4` |
| 45 | `71f625ddc97d6840266bd3792412b90b5567c38a000ce1045118d489bde82018` | `3a4f27cbfacb5a0c7b6f0683cff2fde9292285bd3dc7e73be87cd29da2b872e4` |
| 46 | `3e4917b44837bf9347ab806f4f57db27451dc50cfc38bb720dfe69374fc3bab4` | `708fcba6bacc9710f55e3a34f2f0f4c1199ed1ab9b311c9b237ba56e78ccd7eb` |
| 47 | `df5aceec813dbb6a68acfab217c96c4dfeba26a9c51cc7fa8908538391f2a9d0` | `cd96689de1a7d6a3ed081f866ed02dd7042cab6c2e81cd733479f12cdbe18f39` |
| 48 | `c185a4efa237caf57fa6f9e463390c3b3db156a6cb02fd25cbbda496d30f42a3` | `e8168cc06947fd5989a913a6a79b7ec3dfbedc236224c5463ad7dfa46cb5a136` |
| 49 | `439a61d7be7d0d5483aa2f4edc20c709180b60a9e0115b3031c19d369961d3cc` | `7e428f34baeb7f87e7de1effa3b81f4fad6ae3bae7b1b5a3b4b40af59ddb3a03` |
| 50 | `d94c4c2aecd3cf01aff1ecf2cf1bfd2a71d75a96aa8bae0b685b347464be04c2` | `e9c21ec1e8e2fb78de1eb1b40a97532ad41fc3878b23ce0c7fb92f3753ae3a3c` |

## Offline and HTTP receipts

All four MP4s are nonempty, contain nonconstant decoded pixels, and decode to **9x64x96x3** at 24 FPS (0.375 s). The HTTP sequence exercised POST `/v1/videos`, GET status through `completed`, and GET `/content`; server logs confirm 50 steps, CFG 8 and seed 42 for both requests. Offline and server use different MP4 encoding settings; no encoded-byte equality between APIs is claimed.

| Entry point | Task | MP4 bytes | MP4 SHA256 |
| --- | --- | --- | --- |
| offline | t2v | 9448 | `3f11ecf8a8c6141ea0b65120fddd80602168220fe9fe550f0fb1b3902f6007c7` |
| offline | ti2v | 5139 | `319b913675666813c2574f1b2cedc56eaca7795fb601d3753fb3c58e83165a18` |
| HTTP | t2v | 22833 | `befb4fd0f4eac0b7060a5cdcb3aa79775b903546fbfde0a1cfa8e2057f61ca99` |
| HTTP | ti2v | 11948 | `0577d1b6ab1f566546cf3609001520d8a6efb5868f7328b8b26fbd677818f723` |

| Offline raw array | SHA256 |
| --- | --- |
| t2v, FP32 `[9,64,96,3]` | `eaf4edb75df13723f7e5f3385619cab9d3cf1a9301fe540cfae754d80646763f` |
| ti2v, FP32 `[9,64,96,3]` | `2023ef91b4f79cdb8881e8a959f05db25fc0a9ca2c3e967ff392080fa407f1ba` |

| HTTP task | Returned ID | Final status / frame count |
| --- | --- | --- |
| t2v | `video_gen_d7a6cf2dd8ed4615bd9680c703c50de3` | `completed` / 9 |
| ti2v | `video_gen_b416a465f55040c7b416c179ffb47def` | `completed` / 9 |

## Scope

These are local correctness/contract checks, not official Buildkite results. No full-size run or speedup benchmark was repeated. Historical release-size capacity observations in the recipe remain explicitly historical.

The runner writes `alignment-receipt.json`, `offline-receipt.json`, `http-receipt.json`, the two alignment NPZ files, offline raw NPY files, four MP4s and the input PNG. Re-run the checked-in commands to regenerate these artifacts; this receipt records their measured evidence without committing generated binary videos/trajectories.
