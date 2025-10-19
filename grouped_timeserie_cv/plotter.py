import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go

class Plotter:
    @staticmethod
    def plot_confusion_matrix(confusion_matrices, class_labels):
        cm_display = ConfusionMatrixDisplay(confusion_matrices, display_labels=class_labels)
        plt.figure(figsize=(8, 6))
        cm_display.plot(values_format='.0f')
        plt.title("Confusion Matrix")
        plt.show()
    
    @staticmethod
    def plot_prediction_vs_actual(actual_values, predicted_values):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_values, y=predicted_values, mode='markers', name='Predicted vs Actual', 
                                 marker=dict(color='blue', opacity=0.5)))
        fig.add_trace(go.Scatter(x=[min(actual_values), max(actual_values)], y=[min(actual_values), max(actual_values)], 
                                 mode='lines', name='Perfect Prediction', 
                                 line=dict(color='red')))

        fig.update_layout(
            title="Prediction vs Actual Values",
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            showlegend=True,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        fig.show()

    @staticmethod
    def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines',
            name='Training score',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines',
            name='Cross-validation score',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))

        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Training examples",
            yaxis_title="Score",
            legend=dict(x=0.5, y=0.1, xanchor='center', yanchor='top'),
            template='plotly_white'
        )

        fig.show()