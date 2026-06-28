"""
Análise de Acurácia por Macro-Grupo FCI
========================================
Este script calcula a acurácia do classificador de raças no nível de
macro-grupo FCI, usando os dados já disponíveis localmente.

Abordagem:
- Lê as métricas por raça (recall e support) do CSV existente
- Lê o mapeamento de macro-grupos FCI do JSON
- Calcula a acurácia mínima garantida por macro-grupo
  (= soma dos acertos por raça agrupados)
- Para o cálculo EXATO, gera um snippet de código que deve ser
  executado no Google Colab (onde all_labels e all_preds estão disponíveis)
"""

import json
import csv
import os
from collections import defaultdict

# ==============================================================
# Paths
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

METRICS_CSV = os.path.join(RESULTS_DIR, "metricas_por_raca_stanford.csv")
MACRO_GROUPS_JSON = os.path.join(RESULTS_DIR, "breed_macro_groups.json")

# ==============================================================
# 1. Carregar dados
# ==============================================================
with open(MACRO_GROUPS_JSON, "r", encoding="utf-8") as f:
    macro_data = json.load(f)

breed_to_group = macro_data["breed_to_group"]

# Ler métricas por raça
breed_metrics = []
with open(METRICS_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Raca"] in ("macro avg", "weighted avg"):
            continue
        breed_metrics.append({
            "breed": row["Raca"],
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1-Score"]),
            "support": int(row["Support"]),
        })

# ==============================================================
# 2. Calcular acurácia MÍNIMA por macro-grupo
#    (contando apenas os acertos por raça, sem considerar
#     se os erros vão para o mesmo grupo ou não)
# ==============================================================
group_correct = defaultdict(int)   # acertos por grupo
group_total = defaultdict(int)     # total de amostras por grupo

total_correct = 0
total_samples = 0

for bm in breed_metrics:
    breed = bm["breed"]
    group = breed_to_group.get(breed, "Unknown")
    correct = round(bm["recall"] * bm["support"])
    support = bm["support"]
    
    group_correct[group] += correct
    group_total[group] += support
    total_correct += correct
    total_samples += support

print("=" * 70)
print("ACURÁCIA MÍNIMA POR MACRO-GRUPO (lower bound)")
print("= Contagem de predições corretas por raça, agrupadas por grupo")
print("= Este é o mínimo pois erros intra-grupo elevam a acurácia real")
print("=" * 70)
print()

# Ordenar por nome do grupo
sorted_groups = sorted(group_total.keys())

print(f"{'Macro-Grupo':<60} {'Acertos':>8} {'Total':>6} {'Acc':>8}")
print("-" * 86)

for g in sorted_groups:
    acc = group_correct[g] / group_total[g] if group_total[g] > 0 else 0
    print(f"{g:<60} {group_correct[g]:>8} {group_total[g]:>6} {acc:>7.2%}")

breed_acc = total_correct / total_samples
print("-" * 86)
print(f"{'TOTAL (= Acurácia por raça)':<60} {total_correct:>8} {total_samples:>6} {breed_acc:>7.2%}")

# ==============================================================
# 3. Gerar snippet para cálculo EXATO no Colab
# ==============================================================
print()
print()
print("=" * 70)
print("PARA CALCULAR A ACURÁCIA EXATA POR MACRO-GRUPO:")
print("Cole o código abaixo no notebook do Colab, APÓS a célula ")
print("que computa all_labels e all_preds")
print("=" * 70)

colab_snippet = '''
# =====================================================
# ACURÁCIA POR MACRO-GRUPO FCI
# =====================================================
# Certifique-se de que all_labels e all_preds já foram computados

import json
import numpy as np
from collections import defaultdict

# Carregar mapeamento de macro-grupos
with open("/content/drive/MyDrive/classification_exp/results/breed_macro_groups.json", "r") as f:
    macro_data = json.load(f)

breed_to_group = macro_data["breed_to_group"]

# Carregar label encoder
with open("/content/drive/MyDrive/classification_exp/results/label_encoder_stanford.json", "r") as f:
    encoder = json.load(f)

idx_to_label = encoder["idx_to_label"]

# Mapear cada índice para seu macro-grupo
idx_to_group = {}
for idx_str, breed in idx_to_label.items():
    idx_to_group[int(idx_str)] = breed_to_group.get(breed, "Unknown")

# Converter labels e preds para macro-grupos
true_groups = np.array([idx_to_group[int(l)] for l in all_labels])
pred_groups = np.array([idx_to_group[int(p)] for p in all_preds])

# Acurácia global por macro-grupo
macro_correct = (true_groups == pred_groups).sum()
macro_total = len(true_groups)
macro_acc = macro_correct / macro_total

print(f"\\n{'='*70}")
print(f"ACURÁCIA POR MACRO-GRUPO FCI")
print(f"{'='*70}")
print(f"\\nAcurácia por raça (120 classes):    {(all_labels == all_preds).mean():.4f} ({(all_labels == all_preds).mean()*100:.2f}%)")
print(f"Acurácia por macro-grupo FCI:       {macro_acc:.4f} ({macro_acc*100:.2f}%)")
print(f"\\nAmostras corretas (raça):           {(all_labels == all_preds).sum()} / {macro_total}")
print(f"Amostras corretas (macro-grupo):    {macro_correct} / {macro_total}")
print(f"Erros que viram acertos no grupo:   {macro_correct - (all_labels == all_preds).sum()}")
print()

# Acurácia por grupo individual
group_names_sorted = sorted(set(true_groups))
print(f"{'Macro-Grupo':<60} {'Acc':>8} {'Correct':>8} {'Total':>6}")
print("-" * 86)

for g in group_names_sorted:
    mask = (true_groups == g)
    g_correct = (pred_groups[mask] == g).sum()
    g_total = mask.sum()
    g_acc = g_correct / g_total if g_total > 0 else 0
    print(f"{g:<60} {g_acc:>7.2%} {g_correct:>8} {g_total:>6}")

# Matriz de confusão entre macro-grupos
unique_groups = sorted(set(true_groups))
n_groups = len(unique_groups)
group_to_idx_map = {g: i for i, g in enumerate(unique_groups)}

cm_groups = np.zeros((n_groups, n_groups), dtype=int)
for tg, pg in zip(true_groups, pred_groups):
    cm_groups[group_to_idx_map[tg], group_to_idx_map[pg]] += 1

print(f"\\n\\nMATRIZ DE CONFUSÃO ENTRE MACRO-GRUPOS:")
print("-" * 70)
# Header
header = f"{'True \\\\ Pred':<12}"
for g in unique_groups:
    short = g.split(" - ")[0].replace("Group ", "G") if "Group" in g else g[:8]
    header += f" {short:>6}"
print(header)
print("-" * 70)

for i, g in enumerate(unique_groups):
    short = g.split(" - ")[0].replace("Group ", "G") if "Group" in g else g[:8]
    row = f"{short:<12}"
    for j in range(n_groups):
        row += f" {cm_groups[i,j]:>6}"
    print(row)

# Classification report por macro-grupo
from sklearn.metrics import classification_report
print(f"\\n\\nCLASSIFICATION REPORT POR MACRO-GRUPO:")
print("-" * 70)
print(classification_report(true_groups, pred_groups, zero_division=0))

# Quantos erros por raça se resolvem ao olhar por macro-grupo
errors_breed = (all_labels != all_preds)
errors_at_breed_level = errors_breed.sum()
errors_at_group_level = (true_groups != pred_groups).sum()
resolved = errors_at_breed_level - errors_at_group_level

print(f"\\n{'='*70}")
print(f"RESUMO")
print(f"{'='*70}")
print(f"Total de erros (nível raça):        {errors_at_breed_level}")
print(f"Total de erros (nível macro-grupo): {errors_at_group_level}")
print(f"Erros resolvidos pelo agrupamento:  {resolved} ({resolved/errors_at_breed_level*100:.1f}% dos erros)")
print(f"\\nConclusão: {resolved/errors_at_breed_level*100:.1f}% das confusões do classificador")
print(f"ocorrem entre raças do MESMO macro-grupo FCI.")
'''

print(colab_snippet)

# ==============================================================
# 4. Salvar os resultados
# ==============================================================
output_path = os.path.join(RESULTS_DIR, "acuracia_macro_grupo.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("ACURÁCIA MÍNIMA POR MACRO-GRUPO FCI (lower bound)\n")
    f.write("=" * 70 + "\n")
    f.write("Nota: Esta é a acurácia MÍNIMA. A acurácia real será mais alta\n")
    f.write("pois parte dos erros de raça são entre raças do mesmo grupo.\n")
    f.write("Para o cálculo exato, execute o snippet no Colab.\n\n")
    
    f.write(f"{'Macro-Grupo':<60} {'Acertos':>8} {'Total':>6} {'Acc':>8}\n")
    f.write("-" * 86 + "\n")
    
    for g in sorted_groups:
        acc = group_correct[g] / group_total[g] if group_total[g] > 0 else 0
        f.write(f"{g:<60} {group_correct[g]:>8} {group_total[g]:>6} {acc:>7.2%}\n")
    
    f.write("-" * 86 + "\n")
    f.write(f"{'TOTAL (= Acurácia por raça = lower bound)':<60} {total_correct:>8} {total_samples:>6} {breed_acc:>7.2%}\n")

print(f"\nResultados salvos em: {output_path}")
