# 🧠 NeuroSense - AI-Powered Autism Support Tool

An intelligent, AI-driven application built with Python and Streamlit for autism-related assessment support. Features real-time machine learning predictions, interactive visualizations, and interpretable insights.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

## ✨ Features

- 🤖 **AI-Powered Predictions** - Random Forest classifier for real-time autism screening
- 📊 **Interactive Visualizations** - Plotly-based charts, gauges, and radar plots
- 💡 **Interpretable Insights** - Clear explanations and personalized recommendations
- 🔄 **Data Preprocessing Pipeline** - Robust validation and feature engineering
- 📥 **Export Capabilities** - Download results in JSON or CSV format
- 🎨 **Modern UI/UX** - Professional design with gradient headers and animations

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project directory
cd d:/Projects/Autism

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
d:/Projects/Autism/
├── app.py                    # Main Streamlit application (700+ lines)
├── config.py                 # Configuration and constants
├── data_preprocessing.py     # Data validation and preprocessing
├── visualizations.py         # Interactive Plotly charts
├── utils.py                  # Helper functions and interpretation
├── requirements.txt          # Python dependencies
├── best_autism_model.pkl    # Trained Random Forest model
└── README.md                # This file
```

## 🎯 How It Works

### 1. Assessment Input
- **10 Behavioral Questions** (A1-A10) based on validated screening tools
- **6 Demographic Fields** including age, gender, and medical history

### 2. ML Prediction
- Uses pre-trained Random Forest classifier
- Generates probability scores (0-100%)
- Classifies risk level (Low/Medium/High)

### 3. Visualization & Insights
- **Risk Analysis**: Probability gauges and distribution charts
- **Behavioral Breakdown**: Bar charts and radar plots
- **Feature Importance**: ML model interpretation
- **Demographics**: Clean data presentation

### 4. Recommendations
- Risk-specific guidance
- Professional consultation suggestions
- Educational resources

## 📊 Visualizations

The application includes multiple interactive visualizations:

- **Probability Gauge** - Animated risk meter with color zones
- **Risk Distribution** - Pie chart showing ASD vs. no ASD probability
- **Behavioral Scores** - Bar chart of individual A1-A10 responses
- **Radar Chart** - Comparative behavioral pattern analysis
- **Feature Importance** - ML model decision factors
- **Demographic Table** - Clean presentation of user information

## 🛠️ Technology Stack

### Core
- **Python 3.8+** - Programming language
- **Streamlit 1.31.0** - Web application framework

### Machine Learning
- **Scikit-learn 1.4.0** - ML models and training
- **Joblib 1.3.2** - Model serialization
- **Imbalanced-learn 0.12.0** - Class balancing

### Data & Visualization
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.3** - Numerical operations
- **Plotly 5.18.0** - Interactive charts
- **Seaborn 0.13.1** - Statistical plots
- **Matplotlib 3.8.2** - Additional plotting

## 📖 Usage Guide

### Navigation

The application has 4 main sections accessible via the sidebar:

1. **Assessment** - Complete the screening questionnaire
2. **About** - Learn about the project and technology
3. **How It Works** - Understand the ML process
4. **Disclaimer** - Important medical and legal information

### Completing an Assessment

1. Answer all 10 behavioral questions (Yes/No)
2. Fill in demographic information
3. Click "Analyze Assessment"
4. View results, visualizations, and recommendations
5. Export results if needed (JSON or CSV)

### Understanding Results

**Risk Levels:**
- 🟢 **Low Risk** (< 30%) - Few ASD characteristics observed
- 🟡 **Medium Risk** (30-60%) - Some characteristics warrant attention
- 🔴 **High Risk** (> 60%) - Professional evaluation recommended

## ⚠️ Important Disclaimer

**NeuroSense is a screening tool for educational and research purposes only.**

- ❌ NOT a medical diagnostic system
- ❌ NOT a replacement for professional evaluation
- ✅ For educational and awareness purposes
- ✅ Should be used alongside professional consultation

**Always consult qualified healthcare professionals for proper diagnosis and treatment.**

## 🔒 Privacy & Security

- ✅ All processing happens locally
- ✅ No data transmitted to external servers
- ✅ Results only saved if you export them
- ✅ No personal information stored

## 📝 License

This project is for educational and research purposes only. Not intended for commercial use or medical diagnosis.

## 🤝 Contributing

This is an educational project. For improvements or suggestions, please ensure any modifications maintain the educational and non-diagnostic nature of the tool.

## 📧 Support

For questions or issues:
- Review the "How It Works" section in the app
- Check the disclaimer for limitations
- Consult healthcare professionals for medical advice

## 🎓 Educational Value

This project demonstrates:
- ✅ Machine learning model integration
- ✅ Real-time prediction systems
- ✅ Data preprocessing pipelines
- ✅ Interactive data visualization
- ✅ Streamlit application development
- ✅ Interpretable AI implementation

## 🌟 Acknowledgments

Built using:
- Streamlit for the web framework
- Scikit-learn for machine learning
- Plotly for interactive visualizations
- Open-source Python ecosystem

---

**Built with ❤️ using Machine Learning | Python | Streamlit | Scikit-learn**

*For educational and research purposes only • Not a medical diagnostic tool*
