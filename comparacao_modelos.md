
# 📊 Comparativo de Modelos: DeepLabV3+ vs FPN-ResNet34

Este relatório apresenta uma análise comparativa entre dois modelos utilizados para segmentação de telhados em imagens de satélite no projeto interno da empresa.

---

## 📌 Objetivo

Avaliar o desempenho de diferentes arquiteturas de segmentação semântica na detecção de telhados em imagens de satélite, com base em métricas, estabilidade de treinamento e capacidade de generalização.

---

## 📋 Tabela Comparativa

| Aspecto                     | 🧪 DeepLabV3+ (ResNet50)              | 🧪 FPN-ResNet34                         |
|-----------------------------|--------------------------------------|----------------------------------------|
| 📚 Arquitetura              | DeepLabV3+                           | FPN                                    |
| 🔧 Backbone                 | ResNet50                             | ResNet34                               |
| 📦 Framework                | PyTorch puro                         | PyTorch + segmentation_models_pytorch |
| 🧠 Dataset                  | 70k imagens                          | 70k imagens                            |
| 🎛️ Augmentação             | Flip, rotação, resize                | Flip, rotação, crop, brilho, ruído     |
| 🧮 Função de perda          | Dice Loss                            | BCEWithLogitsLoss                      |
| 📉 Métrica de avaliação     | Dice Score (não finalizado)          | Dice Score médio                       |
| ⚙️ Estabilidade             | Quebra com 70k imagens               | Treinamento completo (300 épocas)      |
| 📈 Dice Score (teste)       | ~0.70 (estimado)                     | 0.7640                                 |
| 📸 Visualização             | Parcial                              | Completa (15 imagens)                  |
| 💾 Salvamento do modelo     | Não concluído                        | ✅ Concluído                            |

---

## 🧠 Conclusões

- O modelo **DeepLabV3+** apresentou instabilidade durante o treinamento com conjuntos grandes e precisa de ajustes para escalabilidade.
- O modelo **FPN com ResNet34** teve desempenho superior, com estabilidade em 300 épocas e **Dice Score médio de 0.7640**, demonstrando maior eficiência e qualidade nas predições.
- A arquitetura FPN é mais indicada para continuidade do projeto neste momento.

---

## 📂 Recomendação

Utilizar o modelo FPN como base para produção, mantendo DeepLabV3+ para estudos comparativos futuros ou aplicação em imagens mais balanceadas.

