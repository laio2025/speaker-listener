# Fine-tuning MATD3 no ambiente Multi-Agent Speaker–Listener (AgileRL + MPE2)

## Visão geral do notebook

Este notebook faz o **fine-tuning de um agente MATD3 multi-agente** no ambiente `simple_speaker_listener_v4` (MPE2), usando a biblioteca **AgileRL**.  
A estrutura geral é:

1. Instala as dependências (AgileRL, Gymnasium, MPE2, PettingZoo, etc.).
2. Organiza a pasta de trabalho em `/workspace/RL` (padrão RunPod/Jupyter).
3. Define o ambiente multi-agente Speaker–Listener.
4. Configura o MATD3 + hiperparâmetros + buffer de replay.
5. Treina **uma população de agentes MATD3**, usando:
   - **Tournament Selection** (seleção por torneio);
   - **Mutations** (mutações de hiperparâmetros);
   - Replay buffer multi-agente;
   - Log de métricas, plot das pontuações e salvamento do melhor agente (`elite`).

---

!pip install \
    "agilerl>=2.3.5" \
    "gymnasium>=1.2.1" \
    "imageio>=2.37.0" \
    "matplotlib>=3.9.4" \
    "mpe2>=0.0.1" \
    "pettingzoo[mpe]>=1.25.0" \
    "pillow>=12.0.0" \
    "datasets"

    %%writefile envs/speaker_listener_env.py
from mpe2 import simple_speaker_listener_v4

def make_speaker_listener_env(continuous=True, render=False):
    """
    Returns a single parallel MPE2 speaker-listener environment.
    """
    return simple_speaker_listener_v4.parallel_env(
        continuous_actions=continuous,
        render_mode="rgb_array" if render else None
    )