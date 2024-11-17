# Anime Dataset Analysis Project - Finding out what drives a successful anime
## SC1015_FEL1_Group2

### Overview
This project involves an in-depth analysis of an anime dataset to uncover insights, explore trends and predict anime scores using machine learning models. This dataset contains over 25,000 anime titles with features such as Genres, Episodes, Duration, Score and more. The objective is to understand the factors that influence anime rating and predict scores based on various attributes.

### Dataset Information
- Source: [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data)
- Size: 25,000+ anime titles
- **Key Features:**
  - Name, Type (TV, OVA, ONA, etc.)
  - Episodes, Durations Genres
  - Producers, Studio
  - Score, Rank, Popularity
  - Start Date, End Date

### Objective
- Analyze the relationship between anime features & their scores
- Predict anime scores using machine learning models
- Explore trends such as seasonal releases, episode count impact, and genre-based insights

### Data Cleaning
#### Challenges Addressed
1. **Irrelevant Data**
  - Dropped Other Name, Licensors due to irrelevance or excessive unknown values
  - Removed anime types (Movie, Music, Special) not relevant to the analysis
2. Missing and Unknown Values
  - Replaced Unknown Values in Score, Scored By, Episodes and Duration using techniques:
  - Mode & Median Imputation
  - Multivariate Imputation by Chained Equations (MICE)
3. Feature Engineering:
  - Encoded Genres, Producers, and Studios using Multi-Label Binarization
  - Converted Rating, Status and Source to numerical codes via Label Encoding
  - Transformed Duration to total minutes and split Aired into Start and End Date
4. Outliers
   -- Removed outliers in Duration and Episodes for better model performance.

#### Final Dataset
The cleaned dataset includes:
- Standardized and consistent features
- Encoded Categorical Fields
- Imputed Missing Values for Score & Episodes
- Approximately 15,206 titles

### Exploratory Data Analysis (EDA)
#### Key Insights
- **Seasonal Trends**
  - Identified scoring trends across seasons (Winter, Spring, Summer, Fall).
  - Mapped monthly and yearly average
- **Category-Based Analysis**
  - Categorized anime by episode count and analyzed score distributions
  - Explored genre-specific scoring patterns.
- **Correlations**
  - Identified Score to be strongly correlated with Popularity and Members.

### Machine Learning Models
#### Models Implemented
1. Random Forest Regressor
   - Predicted Score Using Duration & Episodes
   - Evaluated with Mean Squared Error (MSE) and R-squared
2. Decision Tree Classifier
   - Classified anime scores and visualized feature importance.
3. XGBoost Regressor
   - Leveraged categorical features for robust predictions
4. CatBoost Classifier
   - Utilized GPU accelerations for genre-based score predictions.

### Visualizations
- Boxplots & scatterplots for score distributions
- Trend lines for predictions across Episodes & Duration
- Seasonal & Monthly Average Score Visualizations
- Decision Tree & Feature Importance Plots

### Contributors
- Jordan Choi - Data Cleaning & EDA
- Yu Kai - Machine Learning & Analysis
- Kye Yong - Visualizations & Reporting

   
 
