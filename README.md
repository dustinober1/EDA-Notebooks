# 📊 EDA Notebooks Repository

A comprehensive collection of exploratory data analysis (EDA) notebooks featuring advanced statistical methods, machine learning techniques, and data visualization.

---

## 📚 Notebooks Overview

### 1. 🐆 **Cheeta Detection.ipynb** (Original - Full Analysis)
**Domain**: Natural Language Processing / Academic Integrity  
**Dataset**: Academic submission text data (269 samples)  
**Objective**: Detect AI-generated or cheating submissions using advanced NLP and machine learning

#### Key Features:
- ✅ **Comprehensive Statistical Analysis**
  - T-tests, chi-square tests, effect size calculations
  - P-value analysis with significance indicators
  
- ✅ **Advanced Linguistic Feature Engineering**
  - Readability scores and complexity metrics
  - POS tagging approximations
  - Lexical diversity and function word analysis
  - Character-level pattern analysis
  
- ✅ **Sophisticated Machine Learning Pipeline**
  - TF-IDF vectorization (150+ features)
  - Multiple algorithms: Random Forest, SVM, Neural Networks, Gradient Boosting
  - Ensemble methods: Hard/Soft Voting, Stacking, Weighted Ensembles
  - Hyperparameter optimization with GridSearchCV
  
- ✅ **Model Performance**
  - **Best Accuracy**: 94.12% (Soft Voting Ensemble)
  - Precision: 94.44%
  - Recall: 91.89%
  - F1-Score: 93.15%

#### Statistical Findings:
| Feature | P-Value | Significance | Effect |
|---------|---------|--------------|--------|
| Average Word Length | < 0.001 | *** | Cheating uses longer words |
| Lexical Diversity | < 0.001 | *** | Higher in cheating |
| Capitalization Ratio | < 0.001 | *** | Lower in cheating |
| Exclamation Count | < 0.001 | *** | Lower in cheating |

---

### 2. 🐆 **Cheeta Detection - Clean.ipynb** (Organized + Transformers)
**Domain**: Natural Language Processing / Academic Integrity  
**Dataset**: Same as above  
**Objective**: Clean, production-ready implementation with transformer model fine-tuning

#### Structure:
1. **Section 1**: Data Loading & Initial Exploration
2. **Section 2**: Statistical Analysis & Key Findings
3. **Section 3**: Advanced Feature Engineering
4. **Section 4**: Transformer Model Fine-Tuning
   - DistilBERT fine-tuning
   - DeBERTa-v3 fine-tuning
   - Optimized for Kaggle environment
5. **Section 5**: Results Visualization
6. **Section 6**: Conclusions & Recommendations

#### Key Features:
- ✅ **Clean, Organized Structure**: Markdown sections with clear headings
- ✅ **Transformer Models**: 
  - Pre-trained DistilBERT (`/kaggle/input/distillbert-base-uncased/transformers/default/1`)
  - Pre-trained DeBERTa-v3 (`/kaggle/input/deberta_v3/keras/deberta_v3_small_en/3`)
- ✅ **Optimized Training**: 
  - 3 epochs with early stopping
  - Batch size: 8
  - Max sequence length: 512
- ✅ **Expected Performance**: 95-99% accuracy (transformer-based)

#### Advantages Over Original:
- 📝 Better organization and readability
- 🚀 State-of-the-art transformer models
- 🎯 Production-ready code structure
- 📊 Comprehensive visualizations
- 🔧 Easy to maintain and extend

---

### 3. 🚬 **Smokers Delight.ipynb**
**Domain**: Public Health / Epidemiology  
**Dataset**: Global smoking prevalence data  
**Objective**: Analyze worldwide smoking trends and patterns

#### Key Features:
- ✅ **Interactive Visualizations** using Plotly
- ✅ **Time-Series Analysis** of smoking trends
- ✅ **Geographic Analysis** of smoking prevalence
- ✅ **Demographic Insights** across regions
- ✅ **Statistical Testing** for trend significance
- ✅ **Advanced Plotting**:
  - Interactive choropleth maps
  - Time-series animations
  - Multi-panel dashboards
  - 3D visualizations

#### Analysis Techniques:
- Temporal trend analysis
- Geographic heatmaps
- Correlation studies
- Comparative regional analysis
- Statistical hypothesis testing

---

## 🛠️ Technical Stack

### Core Libraries:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistics**: scipy, statsmodels
- **Machine Learning**: scikit-learn
- **Deep Learning**: transformers, torch, datasets
- **NLP**: TF-IDF, tokenization, text preprocessing

### Advanced Techniques:
- Ensemble learning (voting, stacking, weighted)
- Hyperparameter optimization (GridSearchCV)
- Feature selection (SelectKBest, RFE)
- Cross-validation (StratifiedKFold)
- Probability calibration
- Transformer fine-tuning

---

## 🎯 Use Cases

### Cheeta Detection Notebooks:
- **Academic Institutions**: Detect AI-generated submissions
- **Online Learning Platforms**: Monitor assignment authenticity
- **Research**: Study linguistic patterns in AI-generated text
- **EdTech Companies**: Build plagiarism detection systems

### Smokers Delight Notebook:
- **Public Health Research**: Track smoking trends globally
- **Policy Making**: Inform tobacco control policies
- **Healthcare**: Identify high-risk regions
- **Academic Research**: Study epidemiological patterns

---

## 📈 Results Summary

| Notebook | Domain | Best Model | Accuracy | Key Insight |
|----------|--------|------------|----------|-------------|
| Cheeta Detection | NLP | Soft Voting Ensemble | 94.12% | Word complexity is strongest predictor |
| Cheeta Detection - Clean | NLP | Transformer (expected) | 95-99%* | BERT-based models excel at text classification |
| Smokers Delight | Public Health | N/A (EDA) | N/A | Interactive visualizations reveal temporal/geographic patterns |

*Expected performance based on transformer architecture

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly
pip install transformers datasets torch  # For transformer models
```

### For Kaggle Environment:
The notebooks are optimized for Kaggle with pre-downloaded models:
- DistilBERT: `/kaggle/input/distillbert-base-uncased/transformers/default/1`
- DeBERTa-v3: `/kaggle/input/deberta_v3/keras/deberta_v3_small_en/3`

### Running the Notebooks:
1. Clone the repository
2. Open notebooks in Jupyter/Kaggle
3. Run cells sequentially
4. For transformer models, ensure sufficient GPU memory (recommended: 16GB)

---

## 📊 Model Performance Comparison

### Cheeta Detection Models:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Random Forest | 88.24% | 0.89 | 0.88 | 0.88 | ~1s |
| SVM | 94.12% | 0.95 | 0.92 | 0.93 | ~3s |
| Neural Network | 86.76% | 0.87 | 0.87 | 0.87 | ~5s |
| Gradient Boosting | 79.41% | 0.80 | 0.79 | 0.80 | ~2s |
| **Soft Voting Ensemble** | **94.12%** | **0.94** | **0.92** | **0.93** | ~10s |
| Transformer (expected) | 95-99%* | 0.95+ | 0.95+ | 0.95+ | ~5min |

---

## 🎓 Learning Outcomes

From these notebooks, you'll learn:

1. **Statistical Analysis**:
   - Hypothesis testing (t-tests, chi-square)
   - Effect size calculations
   - Multiple comparison corrections

2. **Feature Engineering**:
   - Text feature extraction
   - TF-IDF vectorization
   - Linguistic pattern analysis

3. **Machine Learning**:
   - Ensemble methods
   - Hyperparameter tuning
   - Model evaluation metrics
   - Cross-validation strategies

4. **Deep Learning**:
   - Transformer model fine-tuning
   - Transfer learning
   - Sequence classification

5. **Data Visualization**:
   - Static plots (matplotlib, seaborn)
   - Interactive visualizations (plotly)
   - Confusion matrices and performance metrics

---

## 🔬 Advanced Topics Covered

- **Natural Language Processing**:
  - Tokenization and text preprocessing
  - TF-IDF and n-gram analysis
  - Transformer architectures (BERT, DeBERTa)
  - Sequence classification

- **Statistical Learning**:
  - Feature selection methods
  - Dimensionality reduction
  - Model calibration
  - Ensemble learning theory

- **Best Practices**:
  - Code organization and documentation
  - Reproducible research
  - Production-ready implementations
  - Performance optimization

---

## 📝 Notes

- All notebooks include detailed comments and markdown explanations
- Statistical significance is marked with asterisks (*, **, ***)
- Visualizations are publication-ready
- Code is modular and reusable
- Compatible with Kaggle, Colab, and local Jupyter environments

---

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new analysis techniques
- Improve documentation

---

## 📧 Contact & Support

For questions or collaboration opportunities, please open an issue or contact the repository owner.

---

## 📜 License

This project is available for educational and research purposes.

---

## 🌟 Highlights

### What Makes These Notebooks Special:

1. **🎯 Production-Ready**: Clean code, proper documentation, best practices
2. **📊 Comprehensive**: From basic EDA to advanced transformer models
3. **🔬 Rigorous**: Statistical tests, significance levels, effect sizes
4. **🚀 State-of-the-Art**: Latest ML techniques and transformer models
5. **📚 Educational**: Detailed explanations and learning resources
6. **🎨 Beautiful**: Publication-quality visualizations
7. **⚡ Optimized**: Efficient code, optimized for Kaggle environment

---

**Last Updated**: October 11, 2025  
**Repository**: EDA-Notebooks  
**Author**: dustinober1
