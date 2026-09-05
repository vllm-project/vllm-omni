<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
누구나 쉽고, 빠르고, 저렴하게 사용할 수 있는 옴니 모달리티 모델 제공
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>문서</b></a> | <a href="https://deepwiki.com/vllm-project/vllm-omni"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>사용자 포럼</b></a> | <a href="https://slack.vllm.ai"><b>개발자 Slack</b></a> | <a href="../../assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>논문</b></a> | <a href="https://docs.google.com/presentation/d/1aPj0OGl_-ZVoib-Qne5dGDAlrRFB-PdHl6E-EE99g8E/edit?usp=sharing"><b>슬라이드</b></a> |
</p>

---

*최신 소식* 🔥
- 최신 릴리스 및 프로젝트 소식은 [영문 README](https://github.com/vllm-project/vllm-omni/blob/main/README.md)에서 확인할 수 있습니다.

---

## 소개

[vLLM](https://github.com/vllm-project/vllm)은 원래 텍스트 기반 자기회귀 생성 작업을 위한 대규모 언어 모델을 지원하도록 설계되었습니다. vLLM-Omni는 이를 확장하여 옴니 모달리티 모델의 추론과 서빙을 지원하는 프레임워크입니다:

- **옴니 모달리티(Omni-modality)**: 텍스트, 이미지, 오디오, 비디오, 액션 데이터 처리
- **비자기회귀 아키텍처(Non-autoregressive Architectures)**: vLLM의 자기회귀(AR) 지원을 디퓨전 트랜스포머(DiT) 및 기타 병렬 생성 모델까지 확장
- **다양한 형태의 출력(Heterogeneous outputs)**: 기존의 텍스트 생성부터 멀티모달 및 액션 출력까지 지원

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni는 다음의 기능들로 빠른 처리 속도를 제공합니다:

- vLLM의 효율적인 KV 캐시 관리를 활용한 최고 수준의 자기회귀(AR) 지원
- 높은 처리량을 위한 파이프라인 단계 실행의 중첩 처리
- OmniConnector 기반의 완전 분리형 구조와 단계 간 동적 리소스 할당

vLLM-Omni는 다음 기능들을 통해 유연하고 쉽게 사용할 수 있습니다:

- 복잡한 모델 워크플로우를 관리하기 위한 다양한 형태의 파이프라인 추상화
- 주요 Hugging Face 모델과의 원활한 연동
- 분산 추론을 위한 텐서, 파이프라인, 데이터 및 전문가 병렬 처리 지원
- 스트리밍 출력
- OpenAI 호환 API 서버
- 스트리밍 오디오 동시 입출력을 지원하는 실시간 서빙 (실험적 기능)

vLLM-Omni는 Hugging Face에서 주요 오픈소스 모델 대부분을 원활하게 지원합니다:

- **옴니 모달리티 모델(Omni-modality models)** (예: Qwen3-Omni, MiniCPM-o 4.5, Cosmos3, HunyuanImage, BAGEL)
- **TTS 모델(TTS models)** (예: Qwen3-TTS, VoxCPM2, Ming-Omni-TTS, CosyVoice3)
- **디퓨전 모델(Diffusion models)** — 이미지, 비디오 및 오디오 생성 (예: MiniMax H3, Qwen-Image, Wan2.2, FLUX)
- **로봇 정책 및 액션 모델(Robot-policy and action models)** (예: GR00T-N1.7, DreamZero-DROID, InternVLA-A1, Cosmos3 action policy)

## 시작하기

자세한 내용은 [공식 문서](https://vllm-omni.readthedocs.io/en/latest/)에서 확인할 수 있습니다.

- [설치](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [빠른 시작](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [지원 모델 목록](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
- vLLM-Omni 모델 서빙을 위한 [배포 가이드](https://recipes.vllm.ai)

## 기여

저희는 모든 기여와 협업을 환영하며 소중하게 생각합니다.
참여 방법은 [vLLM-Omni에 기여하기](https://vllm-omni.readthedocs.io/en/latest/contributing/)에서 확인할 수 있습니다.

## 인용

연구에 vLLM-Omni를 사용하는 경우, 다음 [논문](https://arxiv.org/abs/2602.02204)을 인용해 주세요:

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## 커뮤니티 참여

질문이나 피드백이 있다면, [slack.vllm.ai](https://slack.vllm.ai)의 `#sig-omni` Slack 채널 또는 [discuss.vllm.ai](https://discuss.vllm.ai)의 vLLM 사용자 포럼에서 말씀해 주세요.

## Star History

<a href="https://www.star-history.com/?repos=vllm-project%2Fvllm-omni&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&theme=dark&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
  </picture>
</a>

## 라이선스

본 프로젝트는 [LICENSE](../../../LICENSE) 파일에 명시된 Apache License 2.0을 따릅니다.
