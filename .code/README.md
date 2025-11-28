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


  
