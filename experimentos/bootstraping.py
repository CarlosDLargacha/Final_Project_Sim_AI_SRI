from textwrap import fill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

def load_evaluations(file_path):
    """Carga las evaluaciones desde un archivo CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    df = pd.read_csv(file_path)
    if 'eval' not in df.columns:
        raise ValueError("El CSV debe contener una columna llamada 'eval'")
    
    return df['eval'].values

def bootstrap_evaluations(original_scores, max_iterations=1000, confidence_width=0.1):
    """
    Realiza bootstrapping hasta que el ancho del intervalo de confianza se estabilice
    
    :param original_scores: Array con las puntuaciones iniciales
    :param max_iterations: Límite máximo de iteraciones
    :param confidence_width: Ancho deseado del intervalo de confianza
    :return: Tuple (bootstrap_samples, convergence_iteration, final_ci)
    """
    bootstrap_samples = []
    n = len(original_scores)
    convergence_data = []
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iterations:
        # Genera muestra bootstrap
        sample = np.random.choice(original_scores, size=n, replace=True)
        bootstrap_samples.append(sample)
        
        # Calcula intervalo de confianza cada 10 iteraciones
        if iteration % 10 == 0 and iteration > 0:
            all_means = [np.mean(s) for s in bootstrap_samples]
            mean_mean = np.mean(all_means)
            ci = stats.t.interval(0.95, len(all_means)-1, 
                                 loc=mean_mean, 
                                 scale=stats.sem(all_means))
            ci_width = ci[1] - ci[0]
            
            convergence_data.append({
                'iteration': iteration,
                'mean': mean_mean,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'ci_width': ci_width
            })
            
            if ci_width < confidence_width:
                converged = True
                print(f"Convergencia alcanzada en iteración {iteration}")
                print(f"Intervalo de confianza final: {ci}")
        
        iteration += 1
    
    return bootstrap_samples, pd.DataFrame(convergence_data)

def plot_convergence(convergence_df):
    """Genera gráficos de convergencia"""
    plt.figure(figsize=(12, 6))
    
    # Gráfico de la media y intervalo de confianza
    plt.subplot(1, 2, 1)
    plt.plot(convergence_df['iteration'], convergence_df['mean'], label='Media')
    plt.fill_between(convergence_df['iteration'], 
                    convergence_df['ci_lower'], 
                    convergence_df['ci_upper'], 
                    alpha=0.2, label='IC 95%')
    plt.xlabel('Iteración')
    plt.ylabel('Puntuación media')
    plt.title('Convergencia de la Media')
    plt.legend()
    
    # Gráfico del ancho del intervalo de confianza
    plt.subplot(1, 2, 2)
    plt.plot(convergence_df['iteration'], convergence_df['ci_width'], 'r-')
    plt.xlabel('Iteración')
    plt.ylabel('Ancho del IC')
    plt.title('Reducción del Intervalo de Confianza')
    
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    plt.close()

def save_results(bootstrap_samples, convergence_df, original_scores):
    """Guarda los resultados en archivos CSV"""
    # Guardar estadísticas de convergencia
    convergence_df.to_csv('convergence_stats.csv', index=False)
    
    # Guardar todas las muestras bootstrap
    bootstrap_df = pd.DataFrame({
        'original_mean': np.mean(original_scores),
        'bootstrap_mean': [np.mean(sample) for sample in bootstrap_samples],
        'bootstrap_std': [np.std(sample) for sample in bootstrap_samples]
    })
    bootstrap_df.to_csv('bootstrap_samples.csv', index=False)

def test_mean_significance(bootstrap_means, target=3.5):
    t_stat, p_value = stats.ttest_1samp(bootstrap_means, target)
    print(f"Test t contra objetivo {target}:")
    print(f"Media bootstrap: {np.mean(bootstrap_means):.2f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.5f}")
    if p_value < 0.05:
        print("Resultado SIGNIFICATIVO (sistema funciona)")
    else:
        print("Resultado NO significativo")
        
def analyze_consistency(original_scores, bootstrap_samples):
    original_mean = np.mean(original_scores)
    bootstrap_means = [np.mean(sample) for sample in bootstrap_samples]
    
    # Intervalos de confianza
    ci = np.percentile(bootstrap_means, [2.5, 97.5])
    print(f"\nIntervalo de confianza 95%: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # Coeficiente de variación
    cv = (np.std(bootstrap_means) / np.mean(bootstrap_means)) * 100
    print(f"Coeficiente de variación: {cv:.2f}%")
    print("(Valores < 20% indican buena consistencia)")

def main():
    # Configuración
    INPUT_CSV = 'evaluations.csv'  # Asegúrate de tener este archivo con columna 'eval'
    OUTPUT_DIR = 'results'
    
    # Crear directorio de resultados si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Cargar evaluaciones
        evaluations = load_evaluations(INPUT_CSV)
        print(f"Cargadas {len(evaluations)} evaluaciones iniciales")
        print(f"Estadísticas iniciales - Media: {np.mean(evaluations):.2f}, Desviación: {np.std(evaluations):.2f}")
        
        # 1. Gráfico de distribución de evaluaciones originales
        plt.figure(figsize=(10, 5))
        plt.hist(evaluations, bins=30, edgecolor='k', alpha=0.7)
        plt.axvline(x=np.mean(evaluations), color='r', linestyle='--', label=f'Media: {np.mean(evaluations):.2f}')
        plt.xlabel('Puntuación')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Evaluaciones Originales')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'original_distribution.png'))
        plt.close()
        
        # Aplicar bootstrapping
        bootstrap_samples, convergence_df = bootstrap_evaluations(
            evaluations,
            max_iterations=500,
            confidence_width=0.15
        )
        
        # Generar y guardar resultados
        plot_convergence(convergence_df)
        save_results(bootstrap_samples, convergence_df, evaluations)
        
        print("Proceso completado. Resultados guardados en:")
        print(f"- convergence_stats.csv: Estadísticas de convergencia")
        print(f"- bootstrap_samples.csv: Muestras bootstrap generadas")
        print(f"- convergence_plot.png: Gráficos de convergencia")
        print(f"- original_distribution.png: Distribución de datos originales")
        
        # Cargar datos bootstrap generados
        bootstrap_df = pd.read_csv('bootstrap_samples.csv')

        # 2. Gráfico de distribución de medias bootstrap (ya existente)
        plt.figure(figsize=(10, 5))
        plt.hist(bootstrap_df['bootstrap_mean'], bins=30, edgecolor='k', alpha=0.7)
        plt.axvline(x=3.5, color='r', linestyle='--', label='Mínimo aceptable')
        plt.xlabel('Puntuación media bootstrap')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Medias Bootstrap')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bootstrap_distribution.png'))
        plt.close()
        
        # 3. Boxplot comparativo
        plt.figure(figsize=(10, 5))
        plt.boxplot([evaluations, bootstrap_df['bootstrap_mean']], 
                   tick_labels=['Original', 'Bootstrap'])
        plt.title('Comparación Distribución Original vs Medias Bootstrap')
        plt.ylabel('Puntuación')
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_boxplot.png'))
        plt.close()
        
        # 4. Gráfico de densidad comparativo
        plt.figure(figsize=(10, 5))
        sns.kdeplot(evaluations, label='Original', fill=True)
        sns.kdeplot(bootstrap_df['bootstrap_mean'], label='Bootstrap Means', fill=True)
        plt.axvline(x=3.5, color='r', linestyle='--', label='Mínimo aceptable')
        plt.title('Densidad Comparativa: Original vs Bootstrap')
        plt.xlabel('Puntuación')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'density_comparison.png'))
        plt.close()
        
        # Ejecutar tests
        print("\n=== RESULTADOS ESTADÍSTICOS ===")
        test_mean_significance(bootstrap_df['bootstrap_mean'].values)
        analyze_consistency(evaluations, bootstrap_samples)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()