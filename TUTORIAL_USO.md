# Tutorial de Uso

## 1. Pré-requisitos
Instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

## 2. Organize os dados
Crie as pastas:
```
data/train/
data/train_masks/
data/test/
data/test_masks/
```

## 3. Treine o modelo
```bash
python train.py
```

## 4. Rode a inferência
```bash
python predict.py
```

As máscaras geradas estarão em `results/` e o Dice Score será exibido no terminal.
