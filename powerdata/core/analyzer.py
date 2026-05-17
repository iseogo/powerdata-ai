"""
Core Data Analyzer Module
Provides statistical analysis and data understanding capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """
    Main class for analyzing datasets and extracting insights
    
    Attributes:
        df (pd.DataFrame): The dataset being analyzed
        numeric_cols (List[str]): List of numeric column names
        categorical_cols (List[str]): List of categorical column names
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataset
        
        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_overview(self) -> Dict[str, Any]:
        """
        Get a comprehensive overview of the dataset
        
        Returns:
            Dict containing shape, columns, types, missing values, etc.
        """
        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numeric column
        
        Args:
            column (str): Column name
            method (str): Method to use ('iqr' or 'zscore')
        
        Returns:
            Boolean series indicating outliers
        """
        if column not in self.numeric_cols:
            raise ValueError(f"{column} is not a numeric column")
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            return z_scores > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
    
    def get_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comprehensive statistics for numeric columns
        
        Args:
            columns (List[str], optional): Specific columns to analyze
        
        Returns:
            DataFrame with statistical measures
        """
        if columns is None:
            columns = self.numeric_cols
        
        stats_df = self.df[columns].describe().T
        stats_df['variance'] = self.df[columns].var()
        stats_df['skewness'] = self.df[columns].skew()
        stats_df['kurtosis'] = self.df[columns].kurtosis()
        
        return stats_df
    
    def find_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find strong correlations between numeric columns
        
        Args:
            threshold (float): Minimum correlation coefficient to report
        
        Returns:
            List of dictionaries with correlation pairs
        """
        if len(self.numeric_cols) < 2:
            return []
        
        corr_matrix = self.df[self.numeric_cols].corr()
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    correlations.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def get_top_categories(self, column: str, n: int = 10) -> pd.Series:
        """
        Get top N categories from a categorical column
        
        Args:
            column (str): Column name
            n (int): Number of top categories to return
        
        Returns:
            Series with top categories and their counts
        """
        if column not in self.categorical_cols:
            raise ValueError(f"{column} is not a categorical column")
        
        return self.df[column].value_counts().head(n)
    
    def segment_by_column(self, group_by: str, agg_column: str, agg_func: str = 'mean') -> pd.DataFrame:
        """
        Segment data by a categorical column and aggregate
        
        Args:
            group_by (str): Column to group by
            agg_column (str): Column to aggregate
            agg_func (str): Aggregation function ('mean', 'sum', 'count', etc.)
        
        Returns:
            DataFrame with segmented results
        """
        return self.df.groupby(group_by)[agg_column].agg(agg_func).sort_values(ascending=False)
    
    def get_key_insights(self, max_insights: int = 5) -> List[str]:
        """
        Generate key insights about the dataset
        
        Args:
            max_insights (int): Maximum number of insights to return
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # Missing data insight
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        high_missing = missing_pct[missing_pct > 10]
        if not high_missing.empty:
            insights.append(f"⚠️ {len(high_missing)} columns have >10% missing data: {', '.join(high_missing.index.tolist()[:3])}")
        
        # Duplicate insight
        if self.df.duplicated().sum() > 0:
            dup_pct = (self.df.duplicated().sum() / len(self.df) * 100)
            insights.append(f"🔄 {self.df.duplicated().sum()} duplicate rows found ({dup_pct:.1f}%)")
        
        # Correlation insight
        strong_corrs = self.find_correlations(threshold=0.8)
        if strong_corrs:
            top_corr = strong_corrs[0]
            insights.append(f"🔗 Strong correlation between {top_corr['column1']} and {top_corr['column2']} ({top_corr['correlation']:.2f})")
        
        # Categorical dominance
        for col in self.categorical_cols[:2]:
            top_category = self.df[col].value_counts().iloc[0]
            pct = (top_category / len(self.df) * 100)
            if pct > 50:
                insights.append(f"📊 '{self.df[col].value_counts().index[0]}' dominates {col} ({pct:.1f}%)")
        
        # Outliers
        for col in self.numeric_cols[:2]:
            outliers = self.detect_outliers(col)
            if outliers.sum() > 0:
                outlier_pct = (outliers.sum() / len(self.df) * 100)
                insights.append(f"🎯 {outliers.sum()} outliers detected in {col} ({outlier_pct:.1f}%)")
        
        return insights[:max_insights]


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    analyzer = DataAnalyzer(df)
    
    print("=" * 60)
    print("POWER DATA AI - Data Analyzer Demo")
    print("=" * 60)
    
    print("\n1. Overview:")
    overview = analyzer.get_overview()
    print(f"   Shape: {overview['shape']}")
    print(f"   Columns: {len(overview['columns'])}")
    
    print("\n2. Key Insights:")
    for insight in analyzer.get_key_insights():
        print(f"   {insight}")
    
    print("\n3. Correlations:")
    for corr in analyzer.find_correlations()[:3]:
        print(f"   {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f}")
    
    print("\n" + "=" * 60)
