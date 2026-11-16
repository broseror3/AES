# Grafos de Brauer sobre el Key Schedule de DES (Sin Permutaciones)

Repositorio con los **100 grafos de Brauer** generados a partir de 100 frases distintas convertidas a semillas de 64 bits, aplicando las 16 iteraciones de rotación circular de DES sin tablas de permutación (`PC-1` / `PC-2`).

## Estructura del repositorio
- `grafos_100/`: Imágenes PNG (`grafo_des_001.png` a `grafo_des_100.png`) de cada configuración.
- `comparacion_100_claves.csv`: Métricas completas (valencias, lazos, aristas y dimensión del álgebra $\dim \Lambda_\Gamma$).
- `comparacion_global_100_claves.png`: Gráfica comparativa de invariantes estructurales.
