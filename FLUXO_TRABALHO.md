# Fluxo de Trabalho

1. **Coleta de imagens de satélite** com alta resolução pela empresa.
2. **Criação de máscaras** (rótulos) manuais para treinamento.
3. **Organização da estrutura** de pastas:
   - `data/train/`, `data/train_masks/`, `data/test/`, `data/test_masks/`
4. **Treinamento** do modelo com `train.py`
5. **Salvamento dos pesos** treinados em `models/`
6. **Predição e avaliação** com `predict.py`
7. **Resultados** salvos em `results/` + Dice Score médio impresso
