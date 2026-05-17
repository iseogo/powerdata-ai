# 📊 Power Data AI

<div align="center">

![Power Data AI](https://img.shields.io/badge/Power%20Data%20AI-v1.0.0-blue?style=for-the-badge&logo=python)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

**Turning your data into direction.**

*Empower yourself with AI-powered data analysis — no coding required.*

[🚀 Live Demo](#) | [📖 Documentation](docs/) | [🎯 Examples](examples/) | [🌐 Website](https://seogoglobalimpacts.com)

---

</div>

## 🎯 What is Power Data AI?

Power Data AI is a bilingual (English/French) data analysis platform that helps **leaders, analysts, entrepreneurs, and students** analyze, interpret, and visualize data effortlessly.

Built by **Issaka Seogo** at **Seogo Global Impact**, Power Data AI bridges the gap between raw data and confident decision-making.

### ✨ Key Features

- 📤 **Easy Upload** - Drop CSV or Excel files, get instant insights
- 🤖 **AI-Powered Analysis** - Ask questions in plain English or French
- 📊 **Auto-Visualizations** - Charts and graphs generated automatically
- 📄 **Branded Reports** - Export professional PDF reports with your insights
- 🌍 **Bilingual** - Full support for English and French
- 🎨 **Branded Design** - Professional blue & gold theme throughout

---

## 🚀 Quick Start

### Option 1: Try Online (Easiest)

👉 **[Launch Power Data AI](https://your-app-url.streamlit.app)** *(Deploy to get this link)*

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/iseogo/powerdata-ai.git
cd powerdata-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your OpenAI API key
# Create .streamlit/secrets.toml:
echo 'OPENAI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# 5. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📚 How It Works

### 1. Upload Your Data
Drag and drop any CSV or Excel file, or try one of our sample datasets.

### 2. Ask Questions
Type questions in plain language:
- *"What were our top 5 products by sales?"*
- *"Which region has the highest profit margin?"*
- *"Show me trends over time"*

### 3. Get Insights
Power Data AI analyzes your data using GPT-4 and provides:
- Clear, actionable answers
- Automatic visualizations
- Statistical summaries
- Business recommendations

### 4. Export Reports
Download professional, branded reports in HTML or Markdown format.

---

## 🎨 Brand Identity

**Colors:**
- 🔵 **Blue** (#003399) - Trust, Intelligence
- 🟡 **Gold** (#FFD700) - Excellence, Value
- ⚪ **White** (#FFFFFF) - Clarity, Simplicity

**Mission:**  
Empower people and businesses through data-driven clarity and leadership growth.

**Values:**
- **Simplicity** - Make complex data understandable
- **Accuracy** - Provide reliable insights
- **Transformation** - Turn data into decisions
- **Empowerment** - Build confident decision-makers

---

## 📦 Project Structure

```
powerdata-ai/
├── app.py                      # Main Streamlit application
├── powerdata/                  # Core Python package
│   ├── core/
│   │   ├── analyzer.py        # Data analysis engine
│   │   └── reporter.py        # Report generation
│   ├── visualizations/
│   │   └── charts.py          # Branded chart creation
│   └── utils/                 # Utility functions
├── notebooks/                  # Jupyter examples
│   ├── iris_analysis.ipynb
│   └── sales_analysis.ipynb
├── examples/
│   ├── datasets/              # Sample data
│   └── reports/               # Generated reports
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Core Capabilities

### 1. Data Understanding
- Automatic detection of data types
- Missing value analysis
- Duplicate detection
- Outlier identification

### 2. Statistical Analysis
- Descriptive statistics (mean, median, std, etc.)
- Correlation analysis
- Distribution analysis
- Trend identification

### 3. Visualization
- Histograms and distributions
- Bar charts and rankings
- Scatter plots and relationships
- Correlation heatmaps
- Time series plots

### 4. AI-Powered Insights
- Natural language question answering
- Automated insight generation
- Business recommendations
- Pattern detection

### 5. Report Generation
- Branded HTML reports
- Markdown documentation
- Statistical summaries
- Actionable recommendations

---

## 💻 Usage Examples

### Python API

```python
from powerdata import DataAnalyzer, ChartBuilder, ReportGenerator
import pandas as pd

# Load your data
df = pd.read_csv('sales_data.csv')

# Analyze
analyzer = DataAnalyzer(df)
insights = analyzer.get_key_insights()
correlations = analyzer.find_correlations()

# Visualize
builder = ChartBuilder()
chart = builder.bar_chart(df['product'].value_counts())

# Generate report
report = ReportGenerator(df, "Q4 Sales Analysis")
report.add_insights(insights)
report.save_html('sales_report.html')
```

### Streamlit Interface

1. Upload `sales_data.csv`
2. Ask: *"What are my top 5 products by revenue?"*
3. Get instant AI-powered analysis with charts
4. Download professional report

---

## 🌍 Bilingual Support

Power Data AI fully supports both **English** and **French**:

```python
# English
"What are the top performing categories?"

# French
"Quelles sont les catégories les plus performantes ?"
```

Toggle languages in the sidebar for instant translation of the entire interface.

---

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `OPENAI_API_KEY` to secrets
5. Deploy!

### Deploy with Docker

```bash
# Build image
docker build -t powerdata-ai .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key powerdata-ai
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💼 About the Creator

**Issaka Seogo**  
Founder & CEO, Seogo Global Impact Ltd. Co.

- 📧 Email: issaka.seogo@seogoglobalimpacts.com
- 🌐 Website: [seogoglobalimpacts.com](https://seogoglobalimpacts.com)
- 💼 LinkedIn: [Issaka Seogo](https://linkedin.com/in/issaka-seogo)
- 🗣️ Toastmasters: Realtalkers Club

**Background:**
- Master's in Data Science & AI
- Master's in Public Economics & Applied Statistics
- CNA/CMA Healthcare Experience
- Maxwell DISC Certified
- Based in Omaha, Nebraska

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT-4](https://openai.com/)
- Visualizations by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)

---

## 📞 Support

Need help? Have questions?

- 📧 **Email:** issaka.seogo@seogoglobalimpacts.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/iseogo/powerdata-ai/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/iseogo/powerdata-ai/discussions)

---

<div align="center">

**Power Data AI** – *Turning your data into direction.*

Made with ❤️ by [Issaka Seogo](https://seogoglobalimpacts.com)

⭐ **Star this repo** if you find it useful!

</div>
