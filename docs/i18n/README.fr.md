<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
Serving de modèles omni-modaux simple, rapide et économique, pour tout le monde
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://discuss.vllm.ai"><b>Forum utilisateurs</b></a> | <a href="https://slack.vllm.ai"><b>Slack développeurs</b></a> | <a href="../assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>Article</b></a> | <a href="https://docs.google.com/presentation/d/1XJWgv79lORl8rbaVvp2d5Sqs6ZEBgAgj/edit?slide=id.p1#slide=id.p1"><b>Slides</b></a> |
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <b>Français</b> ·
  <a href="README.ja.md">日本語</a>
</p>

---

*Dernières nouvelles* 🔥
- [2026/03] Nous avons publié la version [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0) — elle renforce le runtime de base grâce à un important remaniement du point d'entrée et au nettoyage du planificateur et du runtime, étend la quantification unifiée ainsi que l'exécution des modèles de diffusion, élargit la couverture des modèles multimodaux et améliore la maturité production sur les axes audio, omni, image, vidéo, RL et déploiements multi-plateformes.
- [2026/03] Découvrez notre première [présentation publique approfondie](https://youtu.be/sgwNfsNnR9I) lors du Meetup vLLM de Hong Kong !
- [2026/03] **[vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills)** est une collection communautaire de compétences pour assistants IA, conçue pour aider les développeurs à travailler plus efficacement avec vLLM-Omni. Ces compétences sont compatibles avec les principaux assistants de codage agentiques tels que **Cursor IDE**, **Claude**, **Codex**, et d'autres.
- [2026/02] Nous avons publié la version [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0) — une version majeure d'alignement et de fonctionnalités qui rebase le projet sur **vLLM v0.16.0 en amont** et étend significativement les performances, l'exécution distribuée et la maturité production pour **Qwen3-Omni / Qwen3-TTS**, **Bagel**, **MiMo-Audio**, **GLM-Image** et la pile **Diffusion (DiT) image/vidéo** — tout en améliorant la couverture des plateformes (CUDA / ROCm / NPU / XPU), la qualité de la CI et la documentation.
- [2026/02] Nous avons publié la version [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0) — il s'agit de la première version **stable** de vLLM-Omni, qui étend la pile de diffusion / génération image-vidéo et la pile audio / TTS d'Omni, améliore l'exécution distribuée et l'efficacité mémoire, et élargit la couverture plateformes/back-ends (GPU/ROCm/NPU/XPU). Elle apporte également des améliorations notables aux API de serving, au profilage, au benchmarking et à la stabilité globale. Consultez notre [article](https://arxiv.org/abs/2602.02204) le plus récent pour les détails sur l'architecture et les performances.
- [2026/01] Nous avons publié la version [0.12.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.12.0rc1) — une étape RC majeure axée sur la maturité de la pile de diffusion, le renforcement du serving compatible OpenAI, l'élargissement de la couverture des modèles omni et l'amélioration de la stabilité sur l'ensemble des plateformes (GPU/NPU/ROCm).
- [2025/11] La communauté vLLM a officiellement publié [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) afin de prendre en charge le serving des modèles omni-modaux.

---

## À propos

[vLLM](https://github.com/vllm-project/vllm) a été conçu à l'origine pour prendre en charge les grands modèles de langage pour des tâches de génération auto-régressive basées sur le texte. vLLM-Omni est un framework qui étend cette prise en charge à l'inférence et au serving de modèles **omni-modaux** :

- **Omni-modalité** : traitement de texte, d'images, de vidéos et de données audio
- **Architectures non auto-régressives** : extension du support AR de vLLM aux Diffusion Transformers (DiT) et à d'autres modèles à génération parallèle
- **Sorties hétérogènes** : de la génération de texte traditionnelle aux sorties multimodales

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni est rapide grâce à :

- Un support AR à la pointe, qui s'appuie sur la gestion efficace du cache KV de vLLM
- Un recouvrement de l'exécution par étapes en pipeline pour un haut débit
- Une désagrégation complète basée sur OmniConnector, et une allocation dynamique des ressources entre les étapes

vLLM-Omni est flexible et facile à utiliser grâce à :

- Une abstraction de pipeline hétérogène pour gérer des workflows de modèles complexes
- Une intégration transparente avec les modèles populaires de Hugging Face
- Le support du parallélisme tensoriel, pipeline, de données et d'experts pour l'inférence distribuée
- Le streaming des sorties
- Un serveur d'API compatible OpenAI

vLLM-Omni prend en charge sans effort la plupart des modèles open source populaires sur HuggingFace, notamment :

- Modèles omni-modaux (ex. Qwen-Omni)
- Modèles de génération multimodale (ex. Qwen-Image)

## Démarrage

Consultez notre [documentation](https://vllm-omni.readthedocs.io/en/latest/) pour en savoir plus.

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [Démarrage rapide](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [Liste des modèles pris en charge](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)

## Contribuer

Nous accueillons et apprécions toute contribution et collaboration.
Veuillez consulter [Contribuer à vLLM-Omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) pour savoir comment participer.

## Citation

Si vous utilisez vLLM-Omni dans vos recherches, veuillez citer notre [article](https://arxiv.org/abs/2602.02204) :

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## Rejoindre la communauté
N'hésitez pas à poser vos questions, partager vos retours et échanger avec les autres utilisateurs de vLLM-Omni sur le canal Slack `#sig-omni` à [slack.vllm.ai](https://slack.vllm.ai), ou sur le forum utilisateurs vLLM à [discuss.vllm.ai](https://discuss.vllm.ai).

## Historique des étoiles

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/vllm-omni&type=date&legend=top-left)](https://www.star-history.com/#vllm-project/vllm-omni&type=date&legend=top-left)

## Licence

Apache License 2.0, comme indiqué dans le fichier [LICENSE](../../LICENSE).

---

> **Note** : ce fichier est une traduction du [`README.md`](../../README.md) anglais et est fourni à titre informatif. En cas de divergence avec la version anglaise, c'est cette dernière qui fait foi.
