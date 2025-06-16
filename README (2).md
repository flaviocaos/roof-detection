
# roof-detection

Segmentação automática de telhados em imagens de satélite utilizando aprendizado profundo (Deep Learning). Este projeto é parte do sistema interno da empresa para geração de dados estruturados a partir de imagens aéreas, com foco em aplicações urbanas, cadastrais e ambientais.

---

## 📌 Visão Geral

Este repositório contém todos os arquivos necessários para treinar, validar, e realizar inferência com um modelo de segmentação semântica baseado em FPN (Feature Pyramid Network) com backbone ResNet34.

O sistema realiza a extração automatizada da área de telhados com base em imagens de satélite adquiridas pela empresa, contribuindo para processos como:
- Regularização fundiária
- Mapeamento urbano automatizado
- Planejamento territorial
- Geração de bases cartográficas atualizadas

---

## 📂 Estrutura do Projeto
```
├── data/
│   ├── train/             # Imagens de treinamento
│   ├── train_masks/       # Máscaras binárias de telhados para treino
│   ├── test/              # Imagens para teste
│   └── test_masks/        # Máscaras binárias de teste
├── models/                # Pesos dos modelos treinados
│   └── fpn_resnet34.pth
├── results/               # Resultados da inferência (máscaras preditas)
├── train.py               # Script de treinamento
├── predict.py             # Script de inferência e avaliação
├── utils.py               # Dataset, métricas e augmentações
└── README.md              # Documentação do projeto
```

---

## 🧠 Arquitetura Utilizada

- **Rede principal:** FPN (Feature Pyramid Network)
- **Backbone:** ResNet34 pré-treinado no ImageNet
- **Framework:** PyTorch + segmentation-models-pytorch
- **Perda:** BCEWithLogitsLoss
- **Métrica:** Dice Score

---

## ⚙️ Requisitos

Instale os pacotes necessários com:
```bash
pip install -r requirements.txt
```
**Dependências principais:**
- Python 3.8+
- torch
- torchvision
- segmentation-models-pytorch
- albumentations
- opencv-python
- matplotlib
- tqdm

---

## 🚀 Etapas de Execução

### 1. Preparação dos Dados
Organize suas imagens e máscaras nos diretórios:
- `data/train/`
- `data/train_masks/`
- `data/test/`
- `data/test_masks/`

As máscaras devem estar no formato binário (telhado = 255, fundo = 0).

### 2. Treinamento do Modelo
Rode o script de treinamento com:
```bash
python train.py
```
Isso irá:
- Carregar os dados
- Aplicar augmentações (flip, rotação, ruído, brilho)
- Treinar a FPN por 50 épocas (ajustável)
- Salvar o modelo em `models/fpn_resnet34.pth`

### 3. Inferência e Avaliação
Com o modelo treinado, execute:
```bash
python predict.py
```
O script:
- Carrega imagens de teste
- Aplica o modelo treinado
- Gera máscaras preditas em `results/`
- Calcula o Dice Score médio da segmentação

---

## 🧰 Conteúdo Técnico dos Arquivos

| Arquivo     | Finalidade |
|-------------|------------|
| `train.py`  | Executa o treinamento da FPN, salvando pesos ao final |
| `predict.py`| Realiza a segmentação e avalia os resultados com Dice Score |
| `utils.py`  | Define a classe `RoofDataset`, funções de augmentação e a métrica Dice |

---

## 📈 Resultados Esperados

- Dice Score médio: ≥ 0.76 em imagens similares ao treinamento
- Máscaras com bordas suavizadas e separação precisa de estruturas telhadas
- Geração rápida de predições (~0.3s por imagem com GPU)

---

## 🛰️ Aplicações
Este projeto é voltado para uso interno em ambientes urbanos e pode ser estendido para:
- Monitoramento de ocupações irregulares
- Atualização de cadastro territorial
- Análise de impacto de expansão urbana

---

## 📬 Contato
Para mais informações sobre uso corporativo, integrações com outros sistemas ou melhorias, entre em contato com o responsável técnico da equipe GAIA-Coleta360.

---

**Nota:** Este repositório é de uso interno. As imagens utilizadas são de propriedade da empresa, adquiridas via contratos com fornecedores de dados geoespaciais.
