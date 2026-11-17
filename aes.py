# ==============================================================================
# TAREA DES: GENERACIÓN DE 16 SUBCLAVES SIN PERMUTACIONES Y 100 GRAFOS DE BRAUER
# Ejecutable directamente en Google Colab (copiar y pegar en una celda o subir .py)
# ==============================================================================

import os
import csv
import shutil
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Tabla estándar de desplazamientos circulares a la izquierda (16 rondas de DES)
DES_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# ==============================================================================
# 1. CONVERSIÓN DE FRASES A 64 BITS Y KEY SCHEDULE DE DES (SIN PERMUTACIONES)
# ==============================================================================

def frase_a_64_bits(frase: str) -> str:
    """
    Convierte una frase UTF-8 a una cadena exacta de 64 bits.
    Si tiene menos de 8 bytes se rellena con ceros a la derecha; si excede, se trunca a 64 bits.
    """
    bits = "".join(f"{byte:08b}" for byte in frase.encode("utf-8"))
    return bits.ljust(64, "0")[:64]

def rotar_izquierda(cadena: str, n: int) -> str:
    """Aplica desplazamiento circular a la izquierda de n posiciones."""
    n = n % len(cadena)
    return cadena[n:] + cadena[:n]

def generar_subclaves_des_sin_permutaciones(clave_64_bits: str) -> list:
    """
    Genera las 16 subclaves K1..K16 sin aplicar tablas de permutación (ni PC-1 ni PC-2).
    Divide la clave de 64 bits en C0 (32 bits) y D0 (32 bits) y aplica las 16 rotaciones.
    """
    C = clave_64_bits[:32]
    D = clave_64_bits[32:]
    subclaves = []
    for shift in DES_SHIFTS:
        C = rotar_izquierda(C, shift)
        D = rotar_izquierda(D, shift)
        subclaves.append(C + D)
    return subclaves

# ==============================================================================
# 2. CONSTRUCCIÓN DEL GRAFO DE BRAUER Q_GAMMA
# ==============================================================================

def construir_grafo_brauer(subclaves: list):
    """
    Construye el multigrafo dirigido de Brauer Q_Gamma:
      - Nodos (Gamma_1): 16 subclaves K1..K16.
      - Vértices de configuración (Gamma_0): {0, 1} con multiplicidad mu = 1.
      - Ciclos especiales C_0 y C_1: conectan cíclicamente las subclaves que contienen cada bit.
      - Lazos implícitos: occ(alpha, K_i) - 1 por cada nodo.
    """
    G = nx.MultiDiGraph()
    nodos = [f"K{i+1}" for i in range(16)]
    G.add_nodes_from(nodos)

    val_0 = sum(k.count("0") for k in subclaves)
    val_1 = sum(k.count("1") for k in subclaves)
    total_lazos = 0

    for simbolo, color in [("0", "#2563eb"), ("1", "#dc2626")]:
        presentes = [
            (f"K{i+1}", k.count(simbolo))
            for i, k in enumerate(subclaves)
            if k.count(simbolo) > 0
        ]
        if not presentes:
            continue

        for nodo, occ in presentes:
            lazos = occ - 1
            total_lazos += lazos
            G.nodes[nodo][f"lazos_{simbolo}"] = lazos

        m = len(presentes)
        for idx in range(m):
            u = presentes[idx][0]
            v = presentes[(idx + 1) % m][0]
            G.add_edge(u, v, simbolo=simbolo, color=color)

    dim_algebra = 2 * 16 + val_0 * (val_0 - 1) + val_1 * (val_1 - 1)
    metricas = {
        "peso_hamming": subclaves[0].count("1"),
        "ceros_por_subclave": subclaves[0].count("0"),
        "val_0": val_0,
        "val_1": val_1,
        "aristas_ciclo": G.number_of_edges(),
        "lazos_implicitos": total_lazos,
        "dim_algebra_brauer": dim_algebra
    }
    return G, metricas

def dibujar_grafo_brauer(G: nx.MultiDiGraph, frase: str, indice: int, metricas: dict, ruta_salida: str, mostrar: bool = False):
    """Renderiza y guarda la imagen PNG del grafo de Brauer."""
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=820, node_color="#1e293b", edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=8.5, font_weight="bold")

    aristas_0 = [(u, v) for u, v, d in G.edges(data=True) if d.get("simbolo") == "0"]
    aristas_1 = [(u, v) for u, v, d in G.edges(data=True) if d.get("simbolo") == "1"]

    if aristas_0:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=aristas_0, edge_color="#2563eb",
            connectionstyle="arc3,rad=0.14", arrows=True, arrowsize=11, width=1.3,
            label="Ciclo especial C_0 (bit 0)"
        )
    if aristas_1:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=aristas_1, edge_color="#dc2626",
            connectionstyle="arc3,rad=-0.14", arrows=True, arrowsize=11, width=1.3,
            label="Ciclo especial C_1 (bit 1)"
        )

    for nodo, (x, y) in pos.items():
        l0 = G.nodes[nodo].get("lazos_0", 0)
        l1 = G.nodes[nodo].get("lazos_1", 0)
        ax.text(
            x * 1.19, y * 1.19, f"L0:{l0}\nL1:{l1}",
            fontsize=6.8, ha="center", va="center", color="#334155",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.6)
        )

    ax.set_title(
        f"Grafo de Brauer DES #{indice:03d} | Frase: \"{frase}\"\n"
        f"Peso Hamming={metricas['peso_hamming']} | val(0)={metricas['val_0']}, val(1)={metricas['val_1']} | dim(Λ)={metricas['dim_algebra_brauer']}",
        fontsize=10.5, fontweight="bold", pad=14
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=160)
    if mostrar:
        plt.show()
    plt.close(fig)

# ==============================================================================
# 3. BANCO DE 100 FRASES DISTINTAS (LAS PRIMERAS 5 SON LOS CASOS DEL INFORME)
# ==============================================================================

FRASES_100 = [
    # --- 5 Casos representativos seleccionados para el informe LaTeX ---
    "AAAAAAAA",  # 001: Baja entropía (caracteres repetidos, 16 unos)
    "AAAAAAA@",  # 002: Diferencia de 1 bit respecto a 001 (15 unos)
    "DES_2026",  # 003: Clave alfanumérica estándar de 8 bytes (27 unos)
    "~~~~~~~~",  # 004: Alto peso de Hamming y dualidad exacta con 001 (48 unos)
    "BogotaDC",  # 005: Frase balanceada en bits (31 unos, 33 ceros)
    # --- 95 Frases adicionales para completar las 100 claves ---
    "Cripto01", "Segura02", "Brauer03", "Algebra4", "Clave005",
    "Matriz06", "Vector07", "Ciclos08", "Nodos009", "Arista10",
    "Simetric", "Bloque64", "Ronda_16", "BitShift", "SinPerm1",
    "Colombia", "Medellin", "Cali2026", "Cartagen", "Barranqu",
    "Bucaramg", "Pereira1", "Manizale", "SantaMar", "Cucuta26",
    "SolLuna1", "NubeAzul", "MarVerde", "RioClaro", "Montaña1",
    "Bosque02", "Viento03", "Fuego004", "Tierra05", "AguaPura",
    "Estrella", "Galaxia1", "Planeta2", "Cometa03", "Orbita04",
    "Quijote1", "Sancho02", "Macondo3", "Aureliano", "Soledad5",
    "Teorema1", "Axioma02", "Lema_003", "Corolari", "Prueba05",
    "Grafo001", "Quiver02", "Valencia", "Camino04", "Lazo_005",
    "Modulo01", "Anillo02", "Cuerpo03", "Grupo004", "Ideal005",
    "Python01", "Colab_02", "GitHub03", "Commit04", "Branch05",
    "Linux_01", "Kernel02", "Shell_03", "Socket04", "Packet05",
    "Alpha_01", "Beta_002", "Gamma_03", "Delta_04", "Epsilon5",
    "Zeta_006", "Eta_0007", "Theta_08", "Iota_009", "Kappa_10",
    "Lambda11", "Sigma_12", "Omega_13", "Phi_0014", "Psi_0015",
    "12345678", "87654321", "00000001", "10000000", "01010101",
    "10101010", "ABCDEFGH", "HGFEDCBA", "QWERTY12", "ASDFGH34"
]

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL, GRÁFICA COMPARATIVA Y EMPAQUETADO PARA GIT
# ==============================================================================

def ejecutar_experimento():
    carpeta_repo = "repo_des_brauer"
    carpeta_grafos = os.path.join(carpeta_repo, "grafos_100")
    os.makedirs(carpeta_grafos, exist_ok=True)

    registros = []

    print(f"Procesando {len(FRASES_100)} frases bajo DES sin permutaciones...\n")

    for idx, frase in enumerate(FRASES_100, start=1):
        clave_bits = frase_a_64_bits(frase)
        subclaves = generar_subclaves_des_sin_permutaciones(clave_bits)
        G, metricas = construir_grafo_brauer(subclaves)

        ruta_png = os.path.join(carpeta_grafos, f"grafo_des_{idx:03d}.png")
        mostrar_en_colab = (idx <= 5)  # Muestra los 5 grafos principales en pantalla
        dibujar_grafo_brauer(G, frase, idx, metricas, ruta_png, mostrar=mostrar_en_colab)

        registros.append({
            "ID": idx,
            "Frase": frase,
            "Clave_64_bits": clave_bits,
            "Peso_Hamming_w": metricas["peso_hamming"],
            "Ceros_64_menos_w": metricas["ceros_por_subclave"],
            "val_0": metricas["val_0"],
            "val_1": metricas["val_1"],
            "Aristas_Ciclo": metricas["aristas_ciclo"],
            "Lazos_Implicitos": metricas["lazos_implicitos"],
            "Dim_Algebra_Brauer": metricas["dim_algebra_brauer"]
        })

    # Guardar tabla CSV con las 100 claves
    df = pd.DataFrame(registros)
    ruta_csv = os.path.join(carpeta_repo, "comparacion_100_claves.csv")
    df.to_csv(ruta_csv, index=False, encoding="utf-8")

    # Generar gráfico comparativo global (Peso de Hamming vs Dimensión del Álgebra de Brauer)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(df["Peso_Hamming_w"], df["Dim_Algebra_Brauer"], color="#1d4ed8", alpha=0.8, s=55, label="100 Claves analizadas")
    w_grid = np.linspace(0, 64, 200)
    dim_teorica = 32 + (16 * (64 - w_grid)) * (16 * (64 - w_grid) - 1) + (16 * w_grid) * (16 * w_grid - 1)
    ax.plot(w_grid, dim_teorica, color="#dc2626", linestyle="--", linewidth=1.4, label=r"Curva teórica $\dim \Lambda_\Gamma(w)$")
    ax.set_title("Comparación de las 100 Claves: Peso de Hamming vs. Dimensión del Álgebra de Brauer", fontsize=11, fontweight="bold")
    ax.set_xlabel("Peso de Hamming de la semilla de 64 bits (w)", fontsize=10)
    ax.set_ylabel(r"Dimensión del Álgebra $\dim \Lambda_\Gamma$", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    ruta_comp = os.path.join(carpeta_repo, "comparacion_global_100_claves.png")
    plt.savefig(ruta_comp, dpi=160)
    plt.show()
    plt.close(fig)

    # Crear README.md listo para el repositorio Git
    ruta_readme = os.path.join(carpeta_repo, "README.md")
    with open(ruta_readme, "w", encoding="utf-8") as f:
        f.write("# Grafos de Brauer sobre el Key Schedule de DES (Sin Permutaciones)\n\n")
        f.write("Repositorio con los **100 grafos de Brauer** generados a partir de 100 frases distintas convertidas a semillas de 64 bits, ")
        f.write("aplicando las 16 iteraciones de rotación circular de DES sin tablas de permutación (`PC-1` / `PC-2`).\n\n")
        f.write("## Estructura del repositorio\n")
        f.write("- `grafos_100/`: Imágenes PNG (`grafo_des_001.png` a `grafo_des_100.png`) de cada configuración.\n")
        f.write("- `comparacion_100_claves.csv`: Métricas completas (valencias, lazos, aristas y dimensión del álgebra $\\dim \\Lambda_\\Gamma$).\n")
        f.write("- `comparacion_global_100_claves.png`: Gráfica comparativa de invariantes estructurales.\n")

    # Mostrar resumen de los 5 casos principales en consola
    print("\n=== RESUMEN DE LOS 5 CASOS REPRESENTATIVOS PARA EL DOCUMENTO ===")
    print(df.head(5)[["ID", "Frase", "Peso_Hamming_w", "val_0", "val_1", "Dim_Algebra_Brauer"]].to_string(index=False))

    # Comprimir carpeta en ZIP para descargar desde Colab y subir a Git
    shutil.make_archive("repo_des_brauer", "zip", carpeta_repo)
    print("\nArchivo 'repo_des_brauer.zip' generado con éxito.")

    try:
        from google.colab import files
        files.download("repo_des_brauer.zip")
    except ImportError:
        print("Ejecución local detectada: revisa la carpeta 'repo_des_brauer/' y 'repo_des_brauer.zip'.")

if __name__ == "__main__":
    ejecutar_experimento()