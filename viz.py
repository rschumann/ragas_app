import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to read JSON data from a file and parse it into a pandas DataFrame
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to create a table from the JSON data
def create_results_table(results):
    df = pd.DataFrame(results)
    return df

# Function to create a combined page with results table, heatmap, and bar chart
def create_combined_page(df):
    # Define the metrics to display
    metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']

    # Create the combined figure with subplots, specifying 'table' type for the first row
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Results Table", "Heatmap of Performance Metrics", "Bar Chart of Average Performance Metrics"),
        row_heights=[0.25, 0.4, 0.35],  # Adjust heights of each section for better separation
        vertical_spacing=0.2,  # Adjust the spacing between subplots
        specs=[[{"type": "table"}], [{"type": "heatmap"}], [{"type": "xy"}]]  # Specify types for each row
    )

    # 1. Add Table to the first row
    header_values = ['Question', 'Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision']
    table_values = [df['question'], df['faithfulness'], df['answer_relevancy'], df['context_recall'], df['context_precision']]
    fig.add_trace(
        go.Table(
            header=dict(values=header_values, fill_color='paleturquoise', align='left'),
            cells=dict(values=table_values, fill_color='lavender', align='left')
        ),
        row=1, col=1
    )

    # 2. Create and add Heatmap in the second row with color bar at the bottom
    heatmap_fig = go.Heatmap(
        z=df[metrics].values.T,  # Transpose to show metrics as y-axis
        x=df['question'], 
        y=metrics,
        colorscale='Blues',
        colorbar=dict(title="Score", orientation="h", x=0.5, xanchor="center")  # Position colorbar at the bottom
    )
    fig.add_trace(heatmap_fig, row=2, col=1)

    # 3. Create and add Bar Chart in the third row
    avg_scores = df[metrics].mean()
    bar_chart = go.Bar(
        x=metrics,
        y=avg_scores,
        marker_color='skyblue'
    )
    fig.add_trace(bar_chart, row=3, col=1)

    # Update layout for the whole page
    fig.update_layout(
        height=1400,  # Adjust height to fit all components with better separation
        showlegend=False,
        title_text="RAG Evaluation Metrics Summary",
        title_x=0.5  # Center the main title
    )

    # Show the combined page
    fig.show()

# Main function to load data, display the table, and generate visualizations
def main(file_path):
    # Load the data
    data = load_json_data(file_path)

    # Create and display the results table
    df = create_results_table(data['results'])

    # Generate and display the combined page with table, heatmap, and bar chart
    create_combined_page(df)

# Example of using the function with the file path
if __name__ == "__main__":
    main('files/source.json')
