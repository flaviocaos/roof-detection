# Arquitetura do Sistema

```
            +-----------------+
            |  Imagens Satélite |
            +--------+--------+
                     |
                     v
        +------------+------------+
        |  Pré-processamento (resize, aug) |
        +------------+------------+
                     |
                     v
            +--------+--------+
            |   FPN + ResNet34 |
            +--------+--------+
                     |
             +-------+-------+
             | Máscara Segmentada |
             +------------------+
                     |
             +------------------+
             | Avaliação (Dice) |
             +------------------+
```

O modelo FPN realiza a segmentação com múltiplas escalas, auxiliado por um encoder ResNet34 que extrai as feições da imagem.
