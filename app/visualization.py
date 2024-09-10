import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import base64
from io import BytesIO
import pandas as pd

def generate_visualizations(evaluation_result):
    df = pd.DataFrame(evaluation_result)
    heatmap_data = df[['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']]
    cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)
    heatmap_img = save_plt_image()

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(df['question'])), df['answer_relevancy'])
    plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)
    plt.xlabel('Answer Relevancy')
    plt.ylabel('Questions')
    bar_chart_img = save_plt_image()

    return heatmap_img, bar_chart_img

def save_plt_image():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
