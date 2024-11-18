# SC1015_FEL1_Group2
## Anime Dataset Analysis Project - 
### Finding out what drives a successful anime

## Overview
This project involves an in-depth analysis of an anime dataset to uncover insights, explore trends and predict anime scores using machine learning models. This dataset contains roughly 25,000 anime titles with features such as Genres, Episodes, Duration, Score and more. We aim to understand the factors influencing anime scores and popularity using machine learning and exploratory analysis, and to find out patterns or trends in high-scoring animes that can help creators, studios develop anime that are more likely to be successful, and help fans make better selection when choosing an anime to watch.
## Dataset Information
- Source: [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data)
- Size: 24,905 anime titles
- 24 columns x 24905 rows in Original Dataset
- **Key Features:**
  - Name, Type (TV, OVA, ONA, etc.)
  - Episodes, Durations Genres
  - Producers, Studio
  - Score, Rank, Popularity
  - Start Date, End Date
<p align=center>
  <table style="width: 100%; text-align: center">
      <tr>
          <th>Data Types Before Cleaning</th>
          <th>Data Types After Cleaning</th>
      </tr>
      <tr>
          <th> <img src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/dtypes%20Original.png width="300"> </th>
          <th><img src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/anime_cleaned_df.dtypes%20after%20cleaning.png width="450"></th>
      </tr>
  </table>
</p>

### Statistical Summary of anime_cleaned_df
![Statistical Summary of anime_cleaned_df](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/anime_cleaned_df.describe().png)
![Statistical Summary of anime_cleaned_df 2](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/anime_cleaned_df.describe()%202%20.png)

Numerical variables are converted to int64 or float.

#### Separated Dataframes
![Data Dims of Separated Dataframes](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Data%20Dims%20of%20Dataframes.png)
![Info of Combined Dataframes](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Info%20of%20merged_df%20(all%20cols).png)

## Objective
- Analyze the relationship between anime features & their scores
- Predict anime scores using machine learning models
- Explore trends such as seasonal releases, episode count impact, and genre-based insights

## Data Cleaning
<table style="width: 100%; text-align: left">
    <tr>
        <th>Field Name</th>
        <th>Field Description</th>
        <th>Clean Up Approach</th>
        <th>Status</th>
        <th>Remarks</th>
    </tr>
    <tr>
        <td>anime_id</td>
        <td>Unique ID for each anime.</td>
        <td>Same as Raw</td>
        <td>✔</td>
        <td></td>
    </tr>
    <tr>
        <td>Name</td>
        <td>The name of the anime in its original language.</td>
        <td>Same as Raw.</td>
        <td>✔</td>
        <td>Use as Primary Name out of 3 Different Name Data Columns. This data column contains unique values for all rows</td>
    </tr>
    <tr>
        <td>English Name</td>
        <td>The English name of the anime.</td>
        <td>Rows with duplicated values with "Name" -> 1<br>"UNKNOWN" -> None<br>Unique Values -> Kept as it is</td>
        <td>✔</td>
        <td>Can be used to find out whether the existence of an English Name affects the Score and other values</td>
    </tr>
    <tr>
        <td>Other Name</td>
        <td>Other names or titles of the anime in different languages.</td>
        <td>Excluded from Cleaned Data Frame.</td>
        <td>✔</td>
        <td>Majority of values are in Japanese Hiragana. We lack the proficiency to process these data, and is unmeaningful.</td>
    </tr>
    <tr>
        <td>Score</td>
        <td>The score or rating given to the anime.</td>
        <td>9213 UNKNOWN values, out of 15206 remain after cleaning up from other columns. Applied Multivariate Imputation by Chained Equations (MICE) to fill these missing data.</td>
        <td>✔</td>
        <td>One of the main objective of our project. Find out what can achieve a high score.</td>
    </tr>
    <tr>
        <td>Genres</td>
        <td>The genres of the anime, separated by commas.</td>
        <td>Separated from the main, cleansed data frame, can be joined back. Applied Multi-Label Binarization to indicate the Genres of the title - since a title can belong to than 1 Genres.<br><br> "1" - True<br>"0" - False</td> 
        <td>✔</td>
        <td>genres_df | 4929 "UNKNOWN" Genres - will be 0s for all genres. Can be used to find out which genres are the most successful titles.</td>
    </tr>
    <tr>
        <td>Synopsis</td>
        <td>A brief description or summary of the anime's plot.</td>
        <td>4535 rows, out of 15206 titles after cleaning up has no description, indicated by "No description available for this anime." -> Transformed to "NA" value</td>
        <td>✔</td>
        <td>Useful to check if existence of a synopsis - using language models, affects the probability of success for an anime title.</td>
    </tr>
    <tr>
        <td>Type</td>
        <td>The type of the anime (e.g., TV series, movie, OVA, etc.).</td>
        <td>9699 titles(rows) of type "Music (2686)", "Movie", "UNKNOWN (74)" and "Special" are dropped from the data frame, as it is not our project focus. 15206 titles(rows) remains - TV, OVA and ONA.</td>
        <td>✔</td>
        <td>The method of release (TV Air, Original Video Animation (Home Video Format) or Original Net Animation (Direct Online)</td>
    </tr>
    <tr>
        <td>Episodes</td>
        <td>The number of episodes in the anime.</td>
        <td>Replace UNKNOWN Episodes (611) value with (Global/Overall Average No. Episodes per Week * No. of Running Weeks). Have to obtain the average number of released episodes/week, from other titles first. Duration may not be available, for such cases: will try to apply KNN.</td>
        <td>✔</td>
        <td>Can consider to categorize them into ranges, since the exact number of episodes may not be meaningful</td>
    </tr>
    <tr>
        <td>Aired</td>
        <td>The dates when the anime was aired.</td>
        <td>Raw string "MMM DD YYYY to MMM DD YYYY" are split into Start Date and End Date, in DateTime format.<br> 915 Titles with "Not Available" Aired Value are replaced with "NaT".<br>Some End Dates are "?", will also be replaced with "NaT" - likely indicates that anime are still airing or end date are not recorded.</td>
        <td>✔</td>
        <td>Can consider to categorize them into ranges, since the exact number of episodes may not be meaningful</td>
    </tr>
    <tr>
        <td>Premiered</td>
        <td>The season and year when the anime premiered.</td>
        <td>9700/15206 Titles are "UNKNOWN". Will input these UNKNOWN values, using the START DATE when available to compute the season and year.</td>
        <td>✔</td>
        <td></td>
    </tr>
    <tr>
        <td>Status</td>
        <td>The status of the anime (e.g., Finished Airing, Currently Airing, etc.)</td>
        <td>Text Categorical Value are transformed to Numerical Categorical Value to represent each status.<br><br>0 - Currently Airing<br>1 - Finished Airing<br>2 - Not yet aired</td>
        <td>✔</td>
        <td></td>
    </tr>
    <tr>
        <td>Producers</td>
        <td>The production companies or producers of the anime.</td>
        <td>Separated from the main, cleansed data frame, can be joined back. Applied Multi-Label Binarization to indicate the Producers of the title - since a title can be produced by a collaboration of producers<br><br> "1" - Produced by the Company<br>"0" - Not Produced By the Company</td>
        <td>✔</td>
        <td>producers_df | 7306/15206 titles with UNKNOWN value are retained.</td>
    </tr>
    <tr>
        <td>Licensors</td>
        <td>The licensors of the anime (e.g., streaming platforms).</td>
        <td>Excluded from Cleaned Data Frame.<br>11666/15206 titles are "UNKNOWN", it is unlikely this column can provide any valuable insights on our problem. This column will be expelled from the data since majority of the values are UNKNOWN.</td>
        <td>✔</td>
        <td></td>
    </tr>
    </tr>
    <tr>
        <td>Studios</td>
        <td>The animation studios that worked on the anime.</td>
        <td>Separated from the main, cleansed data frame, can be joined back. Applied Multi-Label Binarization to indicate the Studios of the title - since a title can be produced by a collaboration of Studios<br><br> "1" - Drawn by the Company<br>"0" - Not Drawn By the Company</td>
        <td>✔</td>
        <td>studios_df | 5470/15206 titles with UNKNOWN value are retained. Since the studios who produced the anime titles could be an important factor to determine the success of an anime title, this column will be retained. Titles can be a collaboration of more than 1 studios, hence, we will follow a similar approach to Genres. Multi-label binarization will be employed. A unique column label will be created for each studio. Binary values (0 or 1) will be used to indicate if the title belongs to the respective studio.</td>
    </tr>
    <tr>
        <td>Source</td>
        <td>The source material of the anime (e.g., manga, light novel, original).</td>
        <td>Converted the Categorical Data Field - "Source" from Text Representation to Numerical Representation, Represented under "Source Code"<br><br>
            0	4-koma manga<br>
            1	Book<br>
            2	Card game<br>
            3	Game<br>
            4	Light novel<br>
            5	Manga<br>
            6	Mixed media<br>
            7	Music<br>
            8	Novel<br>
            9	Original<br>
            10	Other<br>
            11	Picture book<br>
            12	Radio<br>
            13	Unknown<br>
            14	Visual Novel<br>
            15	Web manga<br>
            16	Web novel<br>
        </td>
        <td>✔</td>
        <td>2117/15206 titles with UNKNOWN source values. </td>
    </tr>
    <tr>
        <td>Duration</td>
        <td>The duration of each episode.</td>
        <td>Converted all duration runtime of the titles to minutes</td>
        <td>✔</td>
        <td>416 titles with "UNKNOWN" value for Duration are converted to NaN value.</td>
    </tr>
    <tr>
        <td>Rating</td>
        <td>The age rating of the anime.</td>
        <td>Converted the Categorical Data Field - "Rating" from Text Representation to Numerical Representation, Represented under "Rating Code"<br><br>
            0	G - All Ages<br>
            1	PG - Children<br>
            2	PG-13 - Teens 13 or older<br>
            3	R - 17+ (violence & profanity)<br>
            4	R+ - Mild Nudity<br>
            5	Rx - Hentai<br>
            6	Unknown (Removed)</td>
        <td>✔</td>
        <td>669 UNKNOWN values retained</td>
    </tr>
    <tr>
        <td>Rank</td>
        <td>The rank of the anime based on popularity or other criteria.</td>
        <td>UNKNOWN values are filled with Median Rank Values. Titles with Rank 0 are appended to the end of the ranking in running order.</td>
        <td>✔</td>
        <td>1797/15206 Unknowns, 80 Rank 0 (Invalid). </td>
    </tr>
    <tr>
        <td>Popularity</td>
        <td>The popularity rank of the anime.</td>
        <td>Titles with Rank 0 are appended to the end of the ranking in running order</td>
        <td>✔</td>
        <td>80 values with "UNKNOWN" </td>
    </tr>
    <tr>
        <td>Favorites</td>
        <td>The number of times the anime was marked as a favorite by users.</td>
        <td>No cleaning required. Left as it is.</td>
        <td>✔</td>
        <td>0 Favorites means there are nobody who added the title as favorites.</td>
    </tr>
    <tr>
        <td>Scored By</td>
        <td>The number of users who scored the anime.</td>
        <td>Converted to Int. Replaced 5512 rows with Unknowns using values computed from MICE, for both Score and Scored By, using other columns with numerical data that possess high correlation with Score and Scored By, like Members, Favourites and Popularity.</td>
        <td>✔</td>
        <td></td>
    </tr>
    <tr>
        <td>Members</td>
        <td>The number of members who have added the anime to their list on the platform.</td>
        <td>No cleaning required. Left as it is.</td>
        <td>✔</td>
        <td>0 members means there are no members who added the anime to their list.</td>
    </tr>
    <tr>
        <td>Image URL</td>
        <td>The URL of the anime's image or poster.</td>
        <td>No cleaning required. Left as it is.</td>
        <td>✔</td>
        <td>Kept in case we want to use Machine Learning to see how the poster style may affect the popularity, and other things we want to find out.</td>
    </tr>
</table>
<hr>

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
    - Removed outliers in Duration and Episodes for better model performance.

### Final Dataset
The cleaned dataset includes:
- Standardized and consistent features
- Encoded Categorical Fields
- Imputed Missing Values for Score & Episodes
- Approximately 15,206 titles

## Exploratory Data Analysis (EDA)

## Key Insights
- **Seasonal Trends**
  - Identified scoring trends across seasons (Winter, Spring, Summer, Fall).
    - Anime which debuted in Winter and Summer often have higher scores  
  - Identified trends in monthly and yearly average. Monthly averages peak in April, July, and October, having a sharp drop in May, August, and November respectively
 
### Genres
![Genre Breakdown by Count](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Most%20Common%20Genres.png)
![Average Score by Genre](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Average%20Scores%20of%20Each%20Genres.png)
![Genres Score Heatmap](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Genres%20x%20Score%20heatmap.png)

#### Insights:
- Majority of the anime (~5000) are comedies, followed by Fantasy, Action and Adventure.
- Genres - "Award Winning" , "Girls Love", "Erotica", "Sci-Fi" and "Romance" tends to fetch higher scores in order, but the first three genres has the least anime produced. This means that most of the animes of these genres are consistent, and was not skewed by worse anime of the genres. "Sci-Fi" and "Romance" are within the Top 10 Genres with the most anime titles.
- Hence, "Sci-Fi" and "Romance" are good genres for producers, and studios to produce if they are seeking for a genre with higher likelihood to be successful.

### Producers
![Top 50 Producers by Number of Anime Titles Produced](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Top%2050%20Producers%20with%20Most%20Produced%20Titles.png)
![Heatmap for Average Scores of Producers in Descending Order - 1](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Producers%20Heatmap%201.png)
![Heatmap for Average Scores of Producers in Descending Order - 2](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Producers%20Heatmap%202.png)
![Heatmap for Average Scores of Producers in Descending Order - 3](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Producers%20Heatmap%2024.png)
![Popularity, Score & Producers Relationship Visualized](https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Producers%20x%20Score%20x%20Popularity%20Relationship.png)

### Studios

<p align="center">
  <br>
  <img style="display: inline" src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Top%2050%20Studios%20with%20Most%20Works.png  width="1000"><br>
  <img style="display: inline" src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Studios%20Heatmap%201.png  width="300">
  <img style="display: inline" src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Studios%20Heatmap%202.png  width="300">
  <img style="display: inline" src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Best%20Performing%20Studios%20Heatmap%2012.png  width="300">
  <img style="display: inline" src=https://github.com/agentyk/SC1015_FEL1_Group2/blob/main/Readme%20Images/Studios%20x%20Score%20x%20Popularity%20Relationship.png  width="1000"><br>
  <img style="display: inline" src=https://github.com/user-attachments/assets/0b149d78-1ee9-4534-856b-213af5061687 width="1000"><br>
  <img style="display: inline" src=https://github.com/user-attachments/assets/51d770ad-f9ba-4ba2-86b5-61a37908fe9f width="800"><br>
  <img style="display: inline" src= https://github.com/user-attachments/assets/57890c0b-17a2-4148-9ba9-a465b84117a6 width="1000"><br>
</p>

- **Category-Based Analysis**
  - Categorized anime by episode count and analyzed score distributions. Medium (13-24 eps) length anime has the highest median score, followed by short (1-12 episodes), extremely long (100+ eps), and very long (51-100 eps) anime. 
<img src=https://github.com/user-attachments/assets/762739a9-a4b0-44c0-b17c-1d6ff438fa72 width="500">

  - Explored genre-specific scoring patterns. The average Score per genre of Award Winning Anime is highest, followed closely by Girls Love.
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

**non-Multivalues**

The following is the significance of all the non-multivalues. Having an English name raises the average score by 0.89, being a TV anime raises your average score, and newer anime tending to perform better than old anime. 



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
  <br>
  <img src=https://github.com/user-attachments/assets/66f22efe-1965-47b7-88b3-d01af11b01e7 height="300">
</p>

- Trend lines for predictions across Episodes & Duration. It appears Medium Length anime (from 13-24 episodes) perform best
<p align="center">
  <br>
  <img src=https://github.com/user-attachments/assets/da9ebc4b-fdd0-4260-9d24-69503f57e820 width="1000">
  <br>
  <img src=https://github.com/user-attachments/assets/78c30799-0c83-47d5-8d9b-d5715169edc3 width="1000">
  <br>
  <img src=https://github.com/user-attachments/assets/9d149400-14b9-4ac7-b608-4d98d06f3721 height="340">
  <img src=https://github.com/user-attachments/assets/75db698c-e656-4a0d-ad5d-9276fe8aad84 height="340">
  <br>
  <img src=https://github.com/user-attachments/assets/9993727a-01df-40b7-a2e3-31d7594631a6 height="340">
  <img src=https://github.com/user-attachments/assets/66d465ed-5c6f-4f58-a7d0-f29a6188f146 height="340">
</p>

- Yearly & Monthly Average Score Visualizations. Anime scores fluctuate by as much as 2.5 points from year to year!
![image](https://github.com/user-attachments/assets/d50d5682-cc4c-4d9a-b102-83a19fe6a023)

- Decision Tree & Feature Importance Plots. Duration of anime may be twice as important as anime genre
![image](https://github.com/user-attachments/assets/176fc2ff-51c8-4b07-94cd-7f72fc3c8bb4)
![image](https://github.com/user-attachments/assets/2846e78c-5e64-4ffa-8947-0997cb5fd6fc)

## Contributors
- Jordan Choi - Data Cleaning & EDA
- Yu Kai - Advanced Machine Learning & Analysis
- Kye Yong - Decision Trees & Hypothesis Testing

   
 
