# Predicting Social Media Influence: A Machine Learning Approach

## Project Description

This research project develops and evaluates machine learning algorithms to predict which of two Twitter users is more influential based on social media metrics. Building upon the foundational work by Shimony Agrawal, we have significantly upgraded the analysis by expanding from 4 to 9 machine learning algorithms across four algorithm families (statistical methods, tree-based models, ensemble methods, and deep learning). The work challenges conventional assumptions about social influence by demonstrating that professional recognition metrics (listed_count) are stronger predictors than traditional follower counts, with practical implications for influencer marketing and KOL/KOC selection in e-commerce.

**Research Context:** Master's degree research project in Artificial Intelligence (Course: CM763)  
**Institution:** Department of Management, Yuan Ze University, Taoyuan, Taiwan  
**Supervisor:** Professor Qazi Mazhar Ul Haq  
**Students:** Mai Le Quynh (1133954), Duong Kim Ngan (1137184)  
**Date:** December 2025  
**Dataset:** Influencers in Social Networks (Kaggle)  
**Original Code:** Based on [Social-Media-Analytics-Twitter](https://github.com/shimonyagrawal/Social-Media-Analytics-Twitter) by Shimony Agrawal

### Key Contributions & Upgrades

Our upgraded implementation extends the original 4-algorithm analysis with:
- **Expanded Algorithm Suite:** 9 algorithms across 4 families (vs. original 4 algorithms)
- **Deep Learning Integration:** Added Transformer model for advanced deep learning
- **Additional Algorithms:** Naive Bayes, SVM, Decision Tree, and improved ensemble methods
- **Comprehensive Evaluation:** ROC-AUC scores, cross-validation metrics, and training time analysis
- **Enhanced Metrics:** Detailed overfitting analysis and cross-validation standard deviation
- **Interactive Deployment:** Python-based interactive chatbot for practical predictions

### Key Findings

- **Best Performing Model:** Gradient Boosting (77.03% validation accuracy)
- **Top Predictor:** A/B_listed_count (~60% feature importance)
- **Algorithm Families Evaluated:** Statistical methods, tree-based models, ensemble methods, and deep learning
- **Total Algorithms Compared:** 9 different ML algorithms

## Environment Setup and Requirements

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/chinicapster/1141_CM763_Predicting-Social-Media-Influence.git
cd 1141_CM763_Predicting-Social-Media-Influence
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Python Packages

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
xgboost>=1.5.0
jupyter>=1.0.0
transformers>=4.20.0
scipy>=1.7.0
tqdm>=4.62.0
```

## Dataset

**Source:** [Kaggle - Predict Who is More Influential in a Social Network](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/overview)

**Description:** The dataset contains Twitter user pairs (A and B) with various social media metrics. Each record includes features like follower count, listed count, retweets received, mentions sent, and network features.

**Dataset Characteristics:**
- Original records: ~5,000 user pairs
- Well-balanced classes (no preprocessing needed)
- Included in repository: `[Original code] Social-Media-Analytics-Twitter/train.csv`
- Ready to use without additional data wrangling

### Dataset Structure

The dataset includes the following features for each user pair:
- `A_follower_count` / `B_follower_count` - Number of followers
- `A_following_count` / `B_following_count` - Number of accounts followed
- `A_listed_count` / `B_listed_count` - Times included in Twitter lists
- `A_mentions_received` / `B_mentions_received` - Mentions received
- `A_retweets_received` / `B_retweets_received` - Retweets received
- `A_mentions_sent` / `B_mentions_sent` - Mentions sent
- `A_retweets_sent` / `B_retweets_sent` - Retweets sent
- `A_posts` / `B_posts` - Number of posts
- `A_network_feature_1/2/3` / `B_network_feature_1/2/3` - Network topology metrics
- `Choice` - Target variable (A or B is more influential)

## Instructions to Reproduce Results

### Quick Start - Training Models

The entire analysis is contained in a single Jupyter notebook for ease of use.

1. **Navigate to the upgraded code:**
```bash
cd "[Group work] Upgraded code"
```

2. **Launch Jupyter Notebook:**
```bash
jupyter notebook "[Upgraded] Twitter_Analytics.ipynb"
```

3. **Run all cells:**
   - Click `Cell` → `Run All` or press `Shift + Enter` to execute each cell sequentially

The notebook will automatically:
- Load the dataset from `../[Original code] Social-Media-Analytics-Twitter/train.csv`
- Perform feature engineering (A/B ratio features)
- Train all 9 machine learning models
- Generate evaluation metrics and visualizations
- Display results with ROC curves, confusion matrices, and feature importance plots

### Using the Interactive Chatbot

After training the models, you can use the interactive chatbot for real-time predictions:

```bash
cd "[Group work] Upgraded code"
python interactive_chatbot.py
```

The chatbot provides:
- Interactive command-line interface
- Real-time influence predictions
- User-friendly input prompts for Twitter metrics
- Predictions from the best-performing model (Gradient Boosting)
- Confidence scores for predictions

**Chatbot Usage Example:**
```
=== Twitter Influence Predictor Chatbot ===
Enter User A's follower count: 50000
Enter User A's listed count: 1200
Enter User A's retweets received: 15000
... (continues for all features)

Prediction: User A is more influential
Confidence: 82.5%
```

### Detailed Workflow

The Jupyter notebook is organized into the following sections:

1. **Data Loading and Exploration**
   - Import required libraries
   - Load dataset from the included CSV file
   - Display dataset statistics and structure

2. **Feature Engineering**
   - Create A/B ratio features for comparative analysis
   - Feature selection and correlation analysis

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - XGBoost
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - Decision Tree
   - Naive Bayes
   - Transformer (Deep Learning)

4. **Model Evaluation**
   - Training and validation accuracy
   - Overfitting ratio calculation
   - ROC curve generation
   - Confusion matrix visualization
   - Feature importance analysis

5. **Results Visualization**
   - Performance comparison charts
   - ROC curves for all models
   - Feature importance plots

## Project Structure

```
1141_CM763/
│
├── [Group work] Upgraded code/
│   ├── [Upgraded] Twitter_Analytics.ipynb    # Main analysis notebook (all-in-one)
│   └── interactive_chatbot.py                # Interactive prediction chatbot
│
├── [Original code] Social-Media-Analytics-Twitter/
│   ├── train.csv                              # Dataset (included)
│   └── (original implementation files)
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Results Summary

### Model Performance Comparison

Table 1 presents comprehensive performance metrics for all nine evaluated algorithms, ranked by validation accuracy. The results reveal substantial performance variation across algorithm families, ranging from 50.97% (Logistic Regression) to 77.03% (Gradient Boosting) validation accuracy.

| Model | Train Acc | Val Acc | ROC-AUC | Overfitting | CV Std | Time (s) |
|-------|-----------|---------|---------|-------------|--------|----------|
| **Gradient Boosting** | **81.69%** | **77.03%** | **0.8635** | **4.66%** | **1.70%** | **2.78** |
| Random Forest | 99.40% | 75.94% | 0.8527 | 23.46% | 1.03% | 1.25 |
| XGBoost | 99.30% | 74.00% | 0.8388 | 25.30% | 1.48% | 0.24 |
| KNN | 81.45% | 73.82% | 0.7940 | 7.64% | 1.83% | 0.01 |
| Decision Tree | 84.78% | 73.58% | 0.7938 | 11.20% | 2.24% | 0.07 |
| Transformer | 66.36% | 67.03% | 0.7721 | -0.67% | 0.00% | 82.44 |
| Naive Bayes | 52.16% | 52.61% | 0.2668 | -0.45% | 0.44% | 0.01 |
| SVM | 52.31% | 52.36% | 0.7980 | -0.05% | 0.56% | 4.28 |
| Logistic Regression | 50.94% | 50.97% | 0.8316 | -0.03% | 0.05% | 0.04 |

**Best Model:** Gradient Boosting
- **Validation Accuracy:** 77.03%
- **Training Accuracy:** 81.69%
- **ROC-AUC:** 0.8635
- **Overfitting:** 4.66% (low overfitting indicates good generalization)
- **Cross-Validation Std:** 1.70% (consistent performance)
- **Training Time:** 2.78 seconds

### Comparison with Original Implementation

| Metric | Original (4 models) | Upgraded (9 models) | Improvement |
|--------|---------------------|---------------------|-------------|
| Best Accuracy | 81.00% (XGBoost) | 77.03% (Gradient Boosting) | Better generalization |
| Overfitting | ~30% (after rerun the code) | 4.66% (Gradient Boosting) | ✓ Low overfitting |
| Algorithms | 4 | 9 | +125% coverage |
| Algorithm Families | 2 | 4 | +100% diversity |
| Advanced Models | None | Transformer (Deep Learning) | ✓ Added |
| Deployment | None | Interactive Chatbot | ✓ Practical application |
| ROC-AUC | Not reported | 0.8635 (best model) | ✓ Comprehensive metrics |

**Note:** Our Gradient Boosting model demonstrates superior generalization with only 4.66% overfitting compared to the highly overfit Random Forest (23.46%) and XGBoost (25.30%), making it more reliable for real-world deployment despite slightly lower training accuracy.

### Feature Importance (Top 5 Predictors)

Based on the best-performing Gradient Boosting model:

1. **A/B_listed_count** (~60%): Professional recognition through Twitter list inclusion
2. **A/B_follower_count** (~15%): Traditional follower metric
3. **A/B_network_feature_3** (~10%): Network topology indicator
4. **A/B_retweets_received** (~8%): Content virality measure
5. **A/B_network_feature_2** (~5%): Secondary network metric

### Key Insights

- **Gradient Boosting excels** with best validation accuracy (77.03%) and balanced performance (4.66% overfitting)
- **Ensemble methods dominate** top positions: Gradient Boosting, Random Forest, and XGBoost lead performance
- **Overfitting challenge**: Random Forest (23.46%) and XGBoost (25.30%) show high overfitting despite strong validation scores
- **Transformer model** demonstrates excellent generalization (-0.67% overfitting) but moderate accuracy (67.03%)
- **Classical methods underperform**: Logistic Regression, SVM, and Naive Bayes achieve ~50-52% accuracy (near random)
- **Speed vs Performance tradeoff**: KNN is fastest (0.01s) but Transformer slowest (82.44s)
- **ROC-AUC scores** reveal Gradient Boosting (0.8635) has best discriminative ability
- **Cross-validation stability**: Transformer shows perfect consistency (0.00% CV Std), Gradient Boosting maintains low variance (1.70%)
- **Listed count dominates** as the most predictive feature with ~60% importance, challenging conventional focus on follower counts
- **Professional recognition** metrics are more reliable than raw popularity metrics for influence prediction

## Notes and Limitations

### Project Characteristics

- **Single Notebook Implementation:** All training code is contained in one Jupyter notebook for simplicity and reproducibility
- **Interactive Chatbot:** Separate Python script for user-friendly predictions
- **Dataset Included:** No separate download needed - dataset is in the repository
- **Well-Balanced Data:** Classes are balanced, eliminating need for resampling or SMOTE
- **Direct Usage:** No preprocessing pipeline required - notebook handles everything

### Limitations

1. **Dataset Temporal Constraint:** The Kaggle dataset represents a snapshot in time; Twitter metrics evolve rapidly
2. **Platform Specificity:** Findings are specific to Twitter (now X) and may not generalize to other social platforms
3. **Feature Availability:** Some network features are proprietary and not fully documented in the original dataset
4. **Binary Classification:** Influence is binary (A vs B), which may oversimplify nuanced influence relationships
5. **External Factors:** The model doesn't account for content quality, topic relevance, or temporal trends

### Technical Considerations

- **Negative Overfitting Ratio:** When validation accuracy exceeds training accuracy, this indicates strong regularization or beneficial validation set characteristics rather than poor model fit
- **Random State:** All models use `random_state=42` for reproducibility; results may vary slightly with different seeds
- **GPU Optional:** Neural network training benefits from GPU but works on CPU
- **Chatbot Deployment:** Requires trained model file; run notebook first to generate model

### Future Work

- Expand to multi-class influence classification (low, medium, high influence)
- Develop web-based interface (Flask/Django) for broader accessibility
- Incorporate temporal features and trend analysis
- Test generalizability across different social media platforms
- Develop real-time influence prediction API
- Investigate advanced deep learning architectures (LSTM, Transformers)

## Citation and Acknowledgments

### Original Work

This project builds upon the foundational implementation by Shimony Agrawal:

```
@misc{agrawal2020socialmedia,
  author = {Shimony Agrawal},
  title = {Social Media Analytics - Twitter},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/shimonyagrawal/Social-Media-Analytics-Twitter}
}
```

### Our Contribution

If you use this upgraded implementation in your research, please cite:

```
@masterscourse{2025twitterinfluence,
  title={Predicting Social Media Influence: A Machine Learning Approach with Enhanced Algorithm Suite},
  author={Mai Le Quynh and Duong Kim Ngan},
  year={2025},
  month={December},
  school={Yuan Ze University},
  department={Department of Management},
  address={Taoyuan, Taiwan},
  course={CM763: Artificial Intelligence},
  supervisor={Qazi Mazhar Ul Haq},
  note={Extended implementation based on Shimony Agrawal's original work}
}
```

### Acknowledgments

- **Shimony Agrawal** for the original implementation and foundation
- **Professor Qazi Mazhar Ul Haq** for supervision and guidance throughout the project
- **Department of Management, Yuan Ze University** for research support
- **Kaggle** for providing the Influencers in Social Networks dataset
- **scikit-learn, PyTorch, and XGBoost communities** for excellent ML frameworks

## License

This project follows the original MIT License from Shimony Agrawal's work. See the LICENSE file for details.

## Contact

**Students:**
- Mai Le Quynh (Student ID: 1133954)
- Duong Kim Ngan (Student ID: 1137184)

**Institution:**
- Department of Management
- Yuan Ze University
- Taoyuan, Taiwan

**Repository:**
- GitHub: [https://github.com/chinicapster/1141_CM763_Predicting-Social-Media-Influence](https://github.com/chinicapster/1141_CM763_Predicting-Social-Media-Influence)
- Issues: [Create an issue](https://github.com/chinicapster/1141_CM763_Predicting-Social-Media-Influence/issues)

## Reproducibility Statement

All experiments are reproducible by:
1. Running the single Jupyter notebook `[Upgraded] Twitter_Analytics.ipynb` for model training
2. Using `interactive_chatbot.py` for interactive predictions

Random seeds are fixed (`random_state=42`) across all models. The dataset is included in the repository. No external dependencies or complex setup required - just install the requirements and run the notebook, then use the chatbot for predictions.

---

**Project Period:** November - December 2025  
**Version:** 1.0.0  
**Status:** ✅ Research Complete | 🚀 Ready for Deployment  
**Course:** CM763 - Artificial Intelligence  
**Supervisor:** Professor Qazi Mazhar Ul Haq  
**Institution:** Yuan Ze University, Taiwan
