"""
Chart Builder Module
Creates branded visualizations using Plotly
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any


class ChartBuilder:
    """
    Build branded charts and visualizations
    
    Attributes:
        brand_colors (Dict): Brand color scheme
    """
    
    BRAND_COLORS = {
        'blue': '#003399',
        'gold': '#FFD700',
        'white': '#FFFFFF',
        'light_blue': '#E6F0FF',
        'dark_gold': '#B8960A'
    }
    
    COLOR_SCALE = [
        [0, BRAND_COLORS['blue']],
        [0.5, BRAND_COLORS['light_blue']],
        [1, BRAND_COLORS['gold']]
    ]
    
    def __init__(self):
        """Initialize chart builder with brand styling"""
        self.default_layout = {
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {'family': 'Segoe UI, sans-serif'},
            'title_font': {'color': self.BRAND_COLORS['blue'], 'size': 20}
        }
    
    def _apply_branding(self, fig):
        """Apply brand styling to figure"""
        fig.update_layout(**self.default_layout)
        return fig
    
    def histogram(self, df: pd.DataFrame, column: str, title: Optional[str] = None, bins: int = 30):
        """
        Create branded histogram
        
        Args:
            df (pd.DataFrame): Dataset
            column (str): Column to plot
            title (str, optional): Chart title
            bins (int): Number of bins
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'Distribution of {column}'
        
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=title,
            color_discrete_sequence=[self.BRAND_COLORS['blue']]
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(color=self.BRAND_COLORS['gold'], width=1)
            )
        )
        
        return self._apply_branding(fig)
    
    def bar_chart(self, data: pd.Series, title: Optional[str] = None, orientation: str = 'v'):
        """
        Create branded bar chart
        
        Args:
            data (pd.Series): Data to plot (value counts or aggregated data)
            title (str, optional): Chart title
            orientation (str): 'v' for vertical, 'h' for horizontal
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'Top {len(data)} by {data.name}'
        
        if orientation == 'v':
            fig = px.bar(
                x=data.index,
                y=data.values,
                title=title,
                color=data.values,
                color_continuous_scale=self.COLOR_SCALE
            )
            fig.update_layout(
                xaxis_title=data.index.name or 'Category',
                yaxis_title=data.name or 'Value',
                showlegend=False
            )
        else:
            fig = px.bar(
                x=data.values,
                y=data.index,
                title=title,
                orientation='h',
                color=data.values,
                color_continuous_scale=self.COLOR_SCALE
            )
            fig.update_layout(
                xaxis_title=data.name or 'Value',
                yaxis_title=data.index.name or 'Category',
                showlegend=False
            )
        
        return self._apply_branding(fig)
    
    def scatter_plot(self, df: pd.DataFrame, x: str, y: str, 
                     color: Optional[str] = None, title: Optional[str] = None):
        """
        Create branded scatter plot
        
        Args:
            df (pd.DataFrame): Dataset
            x (str): X-axis column
            y (str): Y-axis column
            color (str, optional): Column for color coding
            title (str, optional): Chart title
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'{y} vs {x}'
        
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_sequence=[
                self.BRAND_COLORS['blue'],
                self.BRAND_COLORS['gold'],
                self.BRAND_COLORS['dark_gold']
            ]
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
        return self._apply_branding(fig)
    
    def line_chart(self, df: pd.DataFrame, x: str, y: str, 
                   color: Optional[str] = None, title: Optional[str] = None):
        """
        Create branded line chart
        
        Args:
            df (pd.DataFrame): Dataset
            x (str): X-axis column
            y (str): Y-axis column
            color (str, optional): Column for color coding
            title (str, optional): Chart title
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'{y} over {x}'
        
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_sequence=[
                self.BRAND_COLORS['blue'],
                self.BRAND_COLORS['gold'],
                self.BRAND_COLORS['dark_gold']
            ]
        )
        
        fig.update_traces(line=dict(width=3))
        
        return self._apply_branding(fig)
    
    def correlation_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                           title: str = "Correlation Matrix"):
        """
        Create correlation heatmap
        
        Args:
            df (pd.DataFrame): Dataset
            columns (List[str], optional): Columns to include
            title (str): Chart title
        
        Returns:
            plotly.graph_objects.Figure
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale=self.COLOR_SCALE,
            aspect='auto',
            text_auto='.2f'
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title=''
        )
        
        return self._apply_branding(fig)
    
    def pie_chart(self, data: pd.Series, title: Optional[str] = None):
        """
        Create branded pie chart
        
        Args:
            data (pd.Series): Data to plot
            title (str, optional): Chart title
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'Distribution by {data.index.name or "Category"}'
        
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=title,
            color_discrete_sequence=[
                self.BRAND_COLORS['blue'],
                self.BRAND_COLORS['gold'],
                self.BRAND_COLORS['light_blue'],
                self.BRAND_COLORS['dark_gold']
            ]
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2))
        )
        
        return self._apply_branding(fig)
    
    def box_plot(self, df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                 title: Optional[str] = None):
        """
        Create branded box plot
        
        Args:
            df (pd.DataFrame): Dataset
            column (str): Column to plot
            group_by (str, optional): Column to group by
            title (str, optional): Chart title
        
        Returns:
            plotly.graph_objects.Figure
        """
        if title is None:
            title = f'Distribution of {column}'
            if group_by:
                title += f' by {group_by}'
        
        if group_by:
            fig = px.box(
                df,
                x=group_by,
                y=column,
                title=title,
                color=group_by,
                color_discrete_sequence=[
                    self.BRAND_COLORS['blue'],
                    self.BRAND_COLORS['gold'],
                    self.BRAND_COLORS['dark_gold']
                ]
            )
        else:
            fig = px.box(
                df,
                y=column,
                title=title,
                color_discrete_sequence=[self.BRAND_COLORS['blue']]
            )
        
        return self._apply_branding(fig)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    builder = ChartBuilder()
    
    print("Creating sample visualizations...")
    
    # Histogram
    fig1 = builder.histogram(df, 'petal length (cm)')
    fig1.write_html('/home/claude/powerdata-ai-enhanced/examples/reports/histogram.html')
    
    # Bar chart
    species_counts = df['species'].value_counts()
    fig2 = builder.bar_chart(species_counts, title='Species Distribution')
    fig2.write_html('/home/claude/powerdata-ai-enhanced/examples/reports/bar_chart.html')
    
    # Scatter
    fig3 = builder.scatter_plot(df, 'petal length (cm)', 'petal width (cm)', color='species')
    fig3.write_html('/home/claude/powerdata-ai-enhanced/examples/reports/scatter.html')
    
    # Correlation
    fig4 = builder.correlation_heatmap(df)
    fig4.write_html('/home/claude/powerdata-ai-enhanced/examples/reports/correlation.html')
    
    print("✅ Visualizations created successfully!")
