import numpy as np
import matplotlib.pyplot as plt

#  Dimensioni dei dataset che vogliamo analizzare
datasets = [20, 50, 100, 200]

for n in datasets:
    file_name = f"smo_log_{n}.npy"
    print(f" Carico file: {file_name}")

    #  Carica il dzionario salvato in prove.py
    data = np.load(file_name, allow_pickle=True).item()
    log_smo = data["log_smo"]
    dual_obj_slsqp = data["dual_obj_slsqp"]

    #  Estrai iterazioni e dual_obj di SMO
    iterations = [entry["iter"] for entry in log_smo]
    dual_obj_values = [entry["dual_obj"] for entry in log_smo]

    #  Calcola dual gap relativo step-by-step
    relative_gaps = [
        abs(obj - dual_obj_slsqp) / (abs(dual_obj_slsqp) + 1e-12)
        for obj in dual_obj_values
    ]

    #  Grafico dual gap relativo vs iterazioni (log scale)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, relative_gaps, marker="o", linestyle="-", color="purple")
    plt.yscale("log")
    plt.title(f"Dual Gap Relativo vs Iterazioni – {n} campioni")
    plt.xlabel("Iterazioni SMO")
    plt.ylabel("Dual Gap Relativo (scala log)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    file_output = f"dual_gap_{n}.png"
    plt.savefig(file_output)
    plt.close()

    print(f" Grafico dual gap relativo salvato: {file_output}")

print("\n Tutti i grafici del dual gap relativo sono stati generati correttamente.")
