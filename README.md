## SC1015_FEL1_Group2
# Anime Dataset Analysis Project - Finding out what drives a successful anime

## Overview
This project involves an in-depth analysis of an anime dataset to uncover insights, explore trends and predict anime scores using machine learning models. This dataset contains over 25,000 anime titles with features such as Genres, Episodes, Duration, Score and more. The objective is to understand the factors that influence anime rating and predict scores based on various attributes.

## Dataset Information
- Source: [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data)
- Size: 25,000+ anime titles
- **Key Features:**
  - Name, Type (TV, OVA, ONA, etc.)
  - Episodes, Durations Genres
  - Producers, Studio
  - Score, Rank, Popularity
  - Start Date, End Date

## Objective
- Analyze the relationship between anime features & their scores
- Predict anime scores using machine learning models
- Explore trends such as seasonal releases, episode count impact, and genre-based insights

## Data Cleaning
### Challenges Addressed
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

### Final Dataset
The cleaned dataset includes:
- Standardized and consistent features
- Encoded Categorical Fields
- Imputed Missing Values for Score & Episodes
- Approximately 15,206 titles

## Exploratory Data Analysis (EDA)
### Key Insights
- **Seasonal Trends**
  - Identified scoring trends across seasons (Winter, Spring, Summer, Fall).
    - Anime which debuted in Winter and Summer often have higher scores  
  - Identified trends in monthly and yearly average. Monthly averages peak in April, July, and October, having a sharp drop in May, August, and November respectively

<p align="left">
  <img src=https://github.com/user-attachments/assets/0b149d78-1ee9-4534-856b-213af5061687 alt="Image 1" width="300"><br>
  <img src=https://github.com/user-attachments/assets/51d770ad-f9ba-4ba2-86b5-61a37908fe9f alt="Image 2" width="300"><br>
  <img src= https://github.com/user-attachments/assets/57890c0b-17a2-4148-9ba9-a465b84117a6 width="300"><br>
</p>

- **Category-Based Analysis**
  - Categorized anime by episode count and analyzed score distributions. Medium (13-24 eps) length anime has the highest median score, followed by short (1-12 episodes), extremely long (100+ eps), and very long (51-100 eps) anime. 
<img src=https://github.com/user-attachments/assets/762739a9-a4b0-44c0-b17c-1d6ff438fa72 width="500">

  - Explored genre-specific scoring patterns. The average Score per genre of Award Winning Anime is highest, followed closely by Girls Love. However, when a t test in conducted on the 
<img src=https://github.com/user-attachments/assets/a9ef2899-6664-4a84-b74e-0e1ee6d84498 width="500">

- **Correlations**
  - Identified Score to be strongly correlated with Popularity and Members.
<img src= https://github.com/user-attachments/assets/9d157e36-14d8-49c9-9d09-ed5c69444ae7 width="500">

- **Hypothesis Testing**
  - Default 5% significance. Performed the following tests, depending on datatype: 
  <table border="1">
  <thead>
    <tr>
      <th>Type</th>
      <th>Analysis Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Date / Numerical</td>
      <td>Spearman Rank Correlation (ordinality matters but not magnitude of difference)</td>
    </tr>
    <tr>
      <td>Categorical</td>
      <td>ANOVA Analysis of Variance</td>
    </tr>
    <tr>
      <td>One-hot</td>
      <td>Independent Sample t-test</td>
    </tr>
    <tr>
      <td>Multivalues</td>
      <td>Thousands of one-hot columns/ Unique Values. Originally from the Studios, Producers, and Genres columns. These columns have been dropped under "Data Cleaning"</td>
    </tr>
  </tbody>
</table>

##### non-Multivalues

The following is the significance of all the non-multivalues. Of note is that having an english name raises the average score by 0.89, being a TV anime raises your average score, and newer anime tending to perform better than old anime. 



 <table border="1" class="dataframe table table-striped">
  <thead>
    <tr style="text-align: center;">
      <th>Field</th>
      <th>Significance</th>
      <th>Significance Level</th>
      <th>Change in Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>anime_id</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Name</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>English name</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>0.890</td>
    </tr>
    <tr>
      <td>Synopsis</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Type</td>
      <td>Significant</td>
      <td>1.21e-122</td>
      <td>Highest: TV, mean: 6.007; Lowest: ONA, mean: 5.495</td>
    </tr>
    <tr>
      <td>Episodes</td>
      <td>Significant</td>
      <td>1.08e-24</td>
      <td>-0.093/unit</td>
    </tr>
    <tr>
      <td>Premiered</td>
      <td>Not Significant</td>
      <td>6.83e-01</td>
      <td>0.005/unit</td>
    </tr>
    <tr>
      <td>Rank</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>-1.012/unit</td>
    </tr>
    <tr>
      <td>Popularity</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>-0.997/unit</td>
    </tr>
    <tr>
      <td>Favorites</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>0.963/unit</td>
    </tr>
    <tr>
      <td>Scored By</td>
      <td>Significant</td>
      <td>7.24e-07</td>
      <td>-0.045/unit</td>
    </tr>
    <tr>
      <td>Members</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>0.997/unit</td>
    </tr>
    <tr>
      <td>Image URL</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Start Date</td>
      <td>Significant</td>
      <td>4.63e-24</td>
      <td>0.092/unit</td>
    </tr>
    <tr>
      <td>End Date</td>
      <td>Significant</td>
      <td>1.26e-05</td>
      <td>-0.045/unit</td>
    </tr>
    <tr>
      <td>Status Code</td>
      <td>Significant</td>
      <td>6.03e-05</td>
      <td>Highest: 2, mean: 6.141; Lowest: 0, mean: 5.816</td>
    </tr>
    <tr>
      <td>Source Code</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>Highest: 4, mean: 6.913; Lowest: 11, mean: 5.141</td>
    </tr>
    <tr>
      <td>Duration_Minutes</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>0.475/unit</td>
    </tr>
    <tr>
      <td>Rating Code</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>Highest: 3, mean: 6.836; Lowest: 1, mean: 4.841</td>
    </tr>
  </tbody>
</table>

##### Genre Significance:
Of particular significance is that Girl's Love is the most popular genre, while Gourmet is the least popular genre

 
 <table border="1" class="dataframe table table-striped">
  <thead>
    <tr style="text-align: center;">
      <th>Field</th>
      <th>Significance</th>
      <th>Significance Level</th>
      <th>Change in Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Action</td>
      <td>Significant</td>
      <td>1.32e-11</td>
      <td>0.182</td>
    </tr>
    <tr>
      <td>Adventure</td>
      <td>Significant</td>
      <td>7.96e-15</td>
      <td>0.239</td>
    </tr>
    <tr>
      <td>Avant Garde</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Award Winning</td>
      <td>Significant</td>
      <td>1.07e-04</td>
      <td>0.544</td>
    </tr>
    <tr>
      <td>Boys Love</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Comedy</td>
      <td>Significant</td>
      <td>3.37e-02</td>
      <td>0.051</td>
    </tr>
    <tr>
      <td>Drama</td>
      <td>Significant</td>
      <td>2.46e-07</td>
      <td>0.164</td>
    </tr>
    <tr>
      <td>Ecchi</td>
      <td>Significant</td>
      <td>2.82e-04</td>
      <td>0.180</td>
    </tr>
    <tr>
      <td>Erotica</td>
      <td>Significant</td>
      <td>1.57e-02</td>
      <td>0.465</td>
    </tr>
    <tr>
      <td>Fantasy</td>
      <td>Significant</td>
      <td>4.06e-03</td>
      <td>0.084</td>
    </tr>
    <tr>
      <td>Girls Love</td>
      <td>Significant</td>
      <td>1.64e-04</td>
      <td>0.483</td>
    </tr>
    <tr>
      <td>Gourmet</td>
      <td>Significant</td>
      <td>3.14e-02</td>
      <td>-0.287</td>
    </tr>
    <tr>
      <td>Hentai</td>
      <td>Significant</td>
      <td>1.50e-04</td>
      <td>0.130</td>
    </tr>
    <tr>
      <td>Horror</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Mystery</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Romance</td>
      <td>Significant</td>
      <td>7.18e-17</td>
      <td>0.286</td>
    </tr>
    <tr>
      <td>Sci-Fi</td>
      <td>Significant</td>
      <td>3.46e-24</td>
      <td>0.314</td>
    </tr>
    <tr>
      <td>Score</td>
      <td>Significant</td>
      <td>0.00e+00</td>
      <td>1.118/unit</td>
    </tr>
    <tr>
      <td>Slice of Life</td>
      <td>Significant</td>
      <td>2.62e-07</td>
      <td>-0.209</td>
    </tr>
    <tr>
      <td>Sports</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Supernatural</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <td>Suspense</td>
      <td>Not Significant</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
  </tbody>
</table>

## Machine Learning Models
### Models Implemented
1. Random Forest Regressor
   - Predicted Score Using Duration & Episodes: 12-24 episodes is the best range for anime. Increase in duration increases anime score very little, though through hypothesis testing it has been shown that increase in duration definitely leads to increase in anime score, though the magnitude may be small. 
   - Suitable for these non-linear relationships
   - Evaluated with Mean Squared Error (MSE) and R-squared
![image](https://github.com/user-attachments/assets/447c6520-5bb0-404b-8241-e5a0c34099a9)
![image](https://github.com/user-attachments/assets/2ba8b812-9500-4fd4-a86c-fecc7f883f99)


2. Decision Tree Classifier
   - Classified anime scores and allow user to input anime parameters to get back predicted anime score. You can try it out yourself! Under Models->Decision Trees-> train_significant_fields_model(), please input in your preferred anime details and it will output the predicted anime score. 
![image](https://github.com/user-attachments/assets/176fc2ff-51c8-4b07-94cd-7f72fc3c8bb4)

3. XGBoost Regressor
   - Leveraged categorical features for robust predictions. Estimates feature importance with regression. Notably, it estimated that the duration of the anime impacts rating more than twice as much as the genre! 
   - Used numeric and categorical values to determine if feature has significant impact on the score

![image](https://github.com/user-attachments/assets/2846e78c-5e64-4ffa-8947-0997cb5fd6fc)

4. CatBoost Classifier
   - Utilized GPU accelerations for genre-based score predictions. Tried to apply it, but it performed worse than linear regression when tested. 
   - Evaluated with Classification Accuracy, Classification Report,
   - Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared
<p align="center">
  <nl>
  <img src=https://github.com/user-attachments/assets/216793ba-ba82-4b67-9709-2ac5adb0c9fb height="300">
  </nl>
</p>


## Visualizations
- Boxplots & scatterplots for score distributions. For example, here TV has higher score than OVA, which has higher score on average than ONAs
<p align="center">
  <nl>
  <img src=https://github.com/user-attachments/assets/66f22efe-1965-47b7-88b3-d01af11b01e7 height="300">
  </nl>
</p>

- Trend lines for predictions across Episodes & Duration. It appears Medium Length anime (from 13-24 episodes) perform best
![image](https://github.com/user-attachments/assets/da9ebc4b-fdd0-4260-9d24-69503f57e820)
![image](https://github.com/user-attachments/assets/78c30799-0c83-47d5-8d9b-d5715169edc3)
![image](https://github.com/user-attachments/assets/9d149400-14b9-4ac7-b608-4d98d06f3721)
![image](https://github.com/user-attachments/assets/75db698c-e656-4a0d-ad5d-9276fe8aad84)
![image](https://github.com/user-attachments/assets/9993727a-01df-40b7-a2e3-31d7594631a6)
![image](https://github.com/user-attachments/assets/66d465ed-5c6f-4f58-a7d0-f29a6188f146)

- Yearly & Monthly Average Score Visualizations. Anime scores fluctuate by as much as 2.5 points from year to year!
![image](https://github.com/user-attachments/assets/d50d5682-cc4c-4d9a-b102-83a19fe6a023)

- Decision Tree & Feature Importance Plots. Duration of anime may be twice as important as anime genre, according to this
![image](https://github.com/user-attachments/assets/176fc2ff-51c8-4b07-94cd-7f72fc3c8bb4)
![image](https://github.com/user-attachments/assets/2846e78c-5e64-4ffa-8947-0997cb5fd6fc)

## Contributors
- Jordan Choi - Data Cleaning & EDA
- Yu Kai - Advanced Machine Learning & Analysis
- Kye Yong - Decision Trees & Hypothesis Testing

   
 
