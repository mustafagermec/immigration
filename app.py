# region 1. IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import statsmodels.stats.api as sms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
# endregion
# region 2. IMPORTING NEW DATASET
df = pd.read_csv('immigration_to_switzerland.csv')
# endregion
# region 3. DIVIDING THE DATA SET AS TRAIN AND TEST SETS
def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:22682], all_data.loc[22683:]

# dividing the dataset
train, test = divide_df(df)

# endregion
# region 4. TESTSET FOR PREDICTION
df_people = pd.DataFrame(test['n_people'])
# endregion
#region 5. PREPROCESSING
class Preprocessing():
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # categorical, numerical and cardinal variables
    def grab_col_names(self, cat_th=10, car_th=30):
        # cat_cols, cat_but_car
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if self.dataframe[col].nunique() < cat_th and
                           self.dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in self.dataframe.columns if self.dataframe[col].nunique() > car_th and
                           self.dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

            # num_cols
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {self.dataframe.shape[0]}")
        print(f"Variables: {self.dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car


    def cat_summary(self, col_name, plot=False, ascending=False):
        counts = self.dataframe[col_name].value_counts(ascending=ascending)
        percentages = counts / len(self.dataframe) * 100
        summary_df = pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage'])

        if plot:
            st.bar_chart(self.dataframe[col_name].value_counts(ascending=ascending))

    # Analysis of target variable with categorical variables
    def target_summary_with_cat(self, target, categorical_col, plot=False):
        target_mean = round(self.dataframe.groupby(categorical_col)[target].mean(), 2)
        if plot:
            fig = px.bar(self.dataframe, x=self.dataframe[categorical_col], y=self.dataframe[target])

            fig.update_layout(
                title=f"Target Summary with {categorical_col.upper()}",
                xaxis=dict(title=categorical_col.upper()),
                yaxis=dict(title=target.upper()),
                xaxis_tickangle=-90
            )

            st.plotly_chart(fig)

    # label encoding
    def label_encoder(self, binary_col):
        labelencoder = LabelEncoder()
        self.dataframe[binary_col] = labelencoder.fit_transform(self.dataframe[binary_col])
        return self.dataframe

    # one-hot encoding
    def one_hot_encoder(self, categorical_cols, drop_first=False):
        self.dataframe = pd.get_dummies(self.dataframe, columns=categorical_cols, drop_first=drop_first)
        return self.dataframe

    # rare analyzer
    def rare_analyzer(self, target, categorical_columns, threshold=0.05):
        for col in categorical_columns:
            counts = self.dataframe[col].value_counts(normalize=True)
            rare_labels = counts[counts < threshold].index
            print(f'{col} : {len(rare_labels)}')
            print(pd.DataFrame({'COUNT': self.dataframe[col].value_counts(),
                                'RATIO': self.dataframe[col].value_counts(normalize=True),
                                'TARGET_MEAN': self.dataframe.groupby(col)[target].mean()}), end='\n\n\n')

    # encoding rare colums
    def rare_encoder(self, rare_perc=0.05):
        temp_df = self.dataframe.copy()
        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' and (
                temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
        for col in rare_columns:
            counts = temp_df[col].value_counts() / len(temp_df)
            rare_labels = counts[counts < rare_perc].index
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
        return temp_df

# Create preprocessing object
prep = Preprocessing(df)
# endregion
# region 6. STREAMLIT APPLICATION

st.image('image/schweiz.jpg', width=400, use_column_width=False)

st.title('Prediction of Migrant Population in Switzerland')
st.sidebar.header('Immigration to Switzerland')
# Add some options to the sidebar
option = st.sidebar.selectbox(
    "Choose an option",
    ["Introduction", 'Materials & Methods', 'Results', "Application"]
)

# Show the selected option
if option == 'Introduction':
    st.subheader("Introduction")
    st.write(f'The worldwide economic crisis of 2008 brought far-reaching changes to the situation for migrants, generated by an indissoluble link with rapid and vertiginous changes to international migrant flows owing to the conflicts that broke out across the Middle East and North Africa. In Italy, the worldwide economic crisis of 2008 ushered in a transitional phase for immigration, which brought far-reaching change to the situation for migrants, generated by an indissoluble link with rapid and vertiginous changes to international migrant flows owing to the conflicts that broke out across the Middle East and North Africa. The COVID-19 pandemic has affected everyone unequally, and undocumented migrants are one of the most affected groups with regard to hospitalization rates and mortality worldwide. In some countries, undocumented migrants may be excluded from national health programs and financial protection for health and social services. The pandemic caused a decrease in migration flows, but variations between regions are significant. The method used for immigrant selection in Australia may affect immigrant quality and labor-market performance.')

    st.write(f'Immigration can have various reasons, including economic, social, political, and personal factors. The following are some of the reasons for immigration that can be found in the search results:')
    st.markdown("""
    - Economic reasons: People may immigrate to another country to find better job opportunities, higher wages, or better living conditions.
    - Social reasons: People may immigrate to join family members or loved ones who have already migrated to another country.
    - Political reasons: People may immigrate to escape persecution, war, or political instability in their home country.
    - Environmental reasons: People may immigrate due to climate change or natural disasters that threaten their existence in their home country.
    Personal reasons: People may immigrate for personal growth, education, or adventure. 
    """)
    st.write(f"The reasons for immigration can vary depending on the individual's circumstances and the country they are migrating from and to. Understanding the motivations behind immigration is essential for policymakers and researchers to develop effective policies and programs that address the needs of immigrants and host communities.")

    st.markdown("Switzerland has observed huge changes in the structure of migratory flows over the last three decades, with an increased proportion of highly skilled migrants being registered in the last 10 years. The employment and socioeconomic integration of immigrants in Switzerland has attracted considerable public and policy concern. Cantonal authorities have strong discretionary powers in admitting non-EU and non-EFTA workers to Switzerland. Migrants living in Switzerland who hold a temporary F residency permit are subjected to a series of limitations regarding their rights, which render their social integration more difficult. Popular initiatives have been used in Switzerland concerning migration, and they have been analysed from historical and current perspectives. The unemployment observed in Switzerland since 1991 is not simply a consequence of a deterioration in the functioning of the Swiss labour market compared with earlier periods, but rather a result of changes in immigration policies and also the reflection in the statistics of a truer picture of the labour market imbalance created by the restructuring of the economy. Overall, the immigration situation in Switzerland is complex and multifaceted, with various factors affecting the integration and socioeconomic status of immigrants.")

    st.write(f'The objectives of the current study are to analyze the data using the advanced visualization methods and predict the number of migrants to be able to come to Switzerland in the future by using advanced machine learning models.')

elif option == 'Materials & Methods':
    st.subheader('Materials & Methods')

    st.markdown("<h4 style='font-weight: bold;'>Materials</h4>", unsafe_allow_html=True)

    st.markdown('To analyze the data and make prediction, PyCharm and Kaggle Jupyter Notebook were used. Besides, to deploy the model over GitHub, Streamlit was employed. The libraries used in the present study were numpy, pandas, seaborn, matplotlib, plotly, sklearn (preprocessing, model_selection, metrics), shap, KNeighborsRegressor, SVR, LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, CatBoostRegressor, LGBMRegressor, XGBRegressor, DecisionTreeRegressor, ExtraTreeRegressor, joblib, optuna, warnings, and lazypredict.')

    st.markdown("<h4 style='font-weight: bold;'>Dataset</h4>", unsafe_allow_html=True)

    st.markdown("The dataset used in the present study was obtained from [opendata.swiss](https://opendata.swiss/en/dataset/einwanderung-der-standigen-und-nicht-standigen-auslandischen-wohnbevolkerung-mit-erwerb) and [SEM](https://www.sem.admin.ch/sem/de/home/publiservice/statistik/auslaenderstatistik.html). It displays information about the year, permanent/temporal population, foreign group, sector, sex, working canton, nation, continent, Europa/Third country, and the number of people. The dataset contains 33,455 rows and 10 columns. In this dataset, there are one integer column (n_people) and nine object columns (year, permanent_temporal_population, foreign_group, sector, sex, working_canton, nation, continent, europa_third_country). Besides, there are no missing values in any of the columns.")

    st.dataframe(df.head())

    st.markdown(f'Total number of migrants coming to Switzerland between 2019 and 2021 is {df.n_people.sum()}. Besides, the minimum, mean, median, and maximum values of the target variable are {df.n_people.min()}, {int(df.n_people.mean())}, {df.n_people.median()}, and {df.n_people.max()}. Since the mean and median values are not close to each other, the target value has outliers as follows:')

    fig = px.box(df['n_people'])
    st.plotly_chart(fig)

    st.markdown('In summary, this dataset shows demographic information, including details about the year, population type, foreign group, sector, gender, working canton, nationality, continent, and the number of people coming to Switzerland between 2019 and 2021.')

    st.markdown("<h4 style='font-weight: bold;'>Handling Dataset</h4>", unsafe_allow_html=True)

    st.markdown("In the present study, the data in 2019, 2020, and 2021 were used for training and testing. To prevent the data leakage, the three different datasets were merged into one and the language was converted to English. Besides, the names of columns were also changed to English to make it more readable and understandable. The dataset was then divided into two as train set and test set and to measure the prediction success, the target variable of test set was saved as a separate dataframe. ")

    st.markdown("<h4 style='font-weight: bold;'>Exploratory Data Analysis and Preprocessing</h4>", unsafe_allow_html=True)

    st.markdown("Since the dataset includes nine categorical features and one numerical column (target), the dateset was analyzed in terms of categorical features. For this, the summary of categorical features was evaluated and then the analysis of numerical target feature with categorical features was performed.")

    st.markdown("<h4 style='font-weight: bold;'>Models Evaluation</h4>", unsafe_allow_html=True)

    st.markdown("Because the study is a regression problem, the root mean squared error (RMSE), mean absolute error (MAE), mean squared error (MSE) and determination of coefficient (R-Squared) were used to evaluate the success of the models")


elif option == 'Results':

    option = st.sidebar.selectbox(
        "Choose a sub option",
        ["Exploratory Data Analysis", "Prediction with Basic Models", "Prediction with Optimized Models"]
    )

    if option == "Exploratory Data Analysis":

        # Sidebar
        option = st.sidebar.selectbox('Select a variable', df.select_dtypes(include='object').columns)

        # Display summary
        st.markdown("<h4 style='font-weight: bold;'>Summary of Categorical Columns</h4>", unsafe_allow_html=True)

        prep.cat_summary(option, plot=True)

        st.markdown("<h4 style='font-weight: bold;'>Analysis of Target Variable with Categorical Columns</h4>", unsafe_allow_html=True)

        prep.target_summary_with_cat('n_people', option, plot=True)

    elif option == 'Prediction with Basic Models':

        st.markdown("<h4 style='font-weight: bold;'>Comparison of the Predicted Values with the Observed Values using Base Models</h4>",
                    unsafe_allow_html=True)

        st.markdown('Based on the results, the R-squared values of base models were determined as 0.8148 for Random Forest Model, 0.8585 for XGBoost Model, 0.8414 for CatBoost Model, and 0.8757 for Voting Regressor. As you see that the highest R-square value was yielded with Voting Regressor, while the Random Forest Model gave its lowest value. Therefore, we can say that Voting Regressor was the most successful model among base models.')

        # calling the dataset
        df = pd.read_csv('one_hot_encodered_df.csv')

        # dividing the dataset
        train, test = divide_df(df)

        # for the train dataset
        y = train['n_people']
        X = train.drop('n_people', axis=1)

        test.drop('n_people', axis=1, inplace=True)

        basic_rf = joblib.load('models/01rf_basic.pkl')
        df1 = test.copy()
        df1['number_people'] = basic_rf.predict(df1)
        df_new1 = pd.concat([df1, df_people], axis=1)

        basic_xgboost = joblib.load('models/02xgboost_basic.pkl')
        df2 = test.copy()
        df2['number_people'] = basic_xgboost.predict(df2)
        df_new2 = pd.concat([df2, df_people], axis=1)

        basic_catboost = joblib.load('models/03catboost_basic.pkl')
        df3 = test.copy()
        df3['number_people'] = basic_catboost.predict(df3)
        df_new3 = pd.concat([df3, df_people], axis=1)

        basic_voting = VotingRegressor([('catboost', basic_catboost),
                                            ('xgboost', basic_xgboost),
                                            ('random_forest', basic_rf)])

        basic_voting.fit(X, y)
        df4 = test.copy()
        df4['number_people'] = basic_voting.predict(df4)
        df_new4 = pd.concat([df4, df_people], axis=1)

        def plot_dataframes(df_new1, df_new2, df_new3, df_new4):
            fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

            sns.regplot(x='n_people', y='number_people', data=df_new1, ax=axs[0])
            r2_1 = round(r2_score(df_new1.n_people, df_new1.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new2, ax=axs[1])
            r2_2 = round(r2_score(df_new2.n_people, df_new2.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new3, ax=axs[2])
            r2_3 = round(r2_score(df_new3.n_people, df_new3.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new4, ax=axs[3])
            r2_4 = round(r2_score(df_new4.n_people, df_new4.number_people), 4)

            fig.suptitle(
                f"R2 values:\nBasic_{type(basic_rf).__name__} = {r2_1}\nBasic_{type(basic_xgboost).__name__} = {r2_2}\nBasic_{type(basic_catboost).__name__} = {r2_3}\nBasic_{type(basic_voting).__name__} = {r2_4}")
            axs[0].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[1].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[2].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[3].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            st.pyplot(fig)

        plot_dataframes(df_new1, df_new2, df_new3, df_new4)

        st.markdown("<h4 style='font-weight: bold;'>Some Statistics with Base Models</h4>",unsafe_allow_html=True)

        st.markdown("The table provides information on the observed number of people and the predictions from different models (RF, Xgboost, Catboost, and Voting Regressor) for a specific variable. By comparing the results, the predictions from the RF model, Xgboost, Catboost, and Voting Regressor are all relatively close, with values ranging from 133,393 to 142,985. The average number predicted by each model is the same, indicating that, on average, the models are predicting a similar number of people. The confidence intervals for each model provide a range within which the true value is expected to lie. The intervals overlap to some extent, indicating similarity in the predictions made by the models. However, there are slight variations in the range of the confidence intervals for each model, suggesting some differences in the models' level of uncertainty in their predictions. Overall, the models' predictions are relatively consistent, with small variations in the predicted numbers and confidence intervals. Further analysis and evaluation would be necessary to determine the performance and accuracy of each model in predicting the number of people accurately.")

        comparison = pd.DataFrame(
            {'Observed': [int(df_new1.n_people.sum()), int(df_new1.n_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new1["n_people"]).tconfint_mean()]],

             'Random Forest Model': [int(df_new1.number_people.sum()), int(df_new1.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new1["number_people"]).tconfint_mean()]],

             'XGBoost Model': [int(df_new2.number_people.sum()), int(df_new2.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new2["number_people"]).tconfint_mean()]],

             'CatBoost Model': [int(df_new3.number_people.sum()), int(df_new3.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new3["number_people"]).tconfint_mean()]],

             'Voting Regressor': [int(df_new4.number_people.sum()), int(df_new4.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new4["number_people"]).tconfint_mean()]]},

            index=['Number of People', 'Average Number', 'Confidence Interval']
        )
        st.write(comparison)

    else:

        st.markdown("<h4 style='font-weight: bold;'>Comparison of the Predicted Values with the Observed Values using Optimized Models</h4>", unsafe_allow_html=True)

        st.markdown('According to the figure, the R-squared values of optimized models were found as 0.8217 for Random Forest Model, 0.8673 for XGBoost Model, 0.8572 for CatBoost Model, and 0.8689 for Voting Regressor. As you see that the highest R-square value was yielded as 0.8689 with Voting Regressor, while the Random Forest Model gave its lowest value (R-square = 0.8217). Therefore, we can say that Voting Regressor was the most successful model among optimized models. On the other hand, when the results were compared to the results of the base models, it is seen that the R-square values, except for Voting Regressor, increased. Therefore, since there is no difference between the R-square values of XGBoost Regressor and Voting Regressor, the XGBoost Regressor was preferred for the Application to reduce the process time.')

        # calling the dataset
        df = pd.read_csv('one_hot_encodered_df.csv')

        # dividing the dataset
        train, test = divide_df(df)

        # for the train dataset
        y = train['n_people']
        X = train.drop('n_people', axis=1)

        test.drop('n_people', axis=1, inplace=True)

        optimized_rf = joblib.load('models/06rf_optimized.pkl')
        df5 = test.copy()
        df5['number_people'] = optimized_rf.predict(df5)
        df_new5 = pd.concat([df5, df_people], axis=1)

        optimized_xgboost = joblib.load('models/05xgboost_optimized.pkl')
        df6 = test.copy()
        df6['number_people'] = optimized_xgboost.predict(df6)
        df_new6 = pd.concat([df6, df_people], axis=1)

        optimized_catboost = joblib.load('models/04catboost_optimized.pkl')
        df7 = test.copy()
        df7['number_people'] = optimized_catboost.predict(df7)
        df_new7 = pd.concat([df7, df_people], axis=1)

        optimized_voting = joblib.load('models/08voting_optimized.pkl')
        optimized_voting.fit(X, y)
        df8 = test.copy()
        df8['number_people'] = optimized_voting.predict(df8)
        df_new8 = pd.concat([df8, df_people], axis=1)

        def plot_dataframes(df_new1, df_new2, df_new3, df_new4):
            fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

            sns.regplot(x='n_people', y='number_people', data=df_new1, ax=axs[0])
            r2_1 = round(r2_score(df_new1.n_people, df_new1.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new2, ax=axs[1])
            r2_2 = round(r2_score(df_new2.n_people, df_new2.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new3, ax=axs[2])
            r2_3 = round(r2_score(df_new3.n_people, df_new3.number_people), 4)

            sns.regplot(x='n_people', y='number_people', data=df_new4, ax=axs[3])
            r2_4 = round(r2_score(df_new4.n_people, df_new4.number_people), 4)

            fig.suptitle(
                f"R2 values:\nOptimized_{type(optimized_rf).__name__} = {r2_1}\nOptimized_{type(optimized_xgboost).__name__} = {r2_2}\nOptimized_{type(optimized_catboost).__name__} = {r2_3}\nOptimized_{type(optimized_voting).__name__} = {r2_4}")
            axs[0].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[1].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[2].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            axs[3].set(xlabel='Observed number of people', ylabel='Predicted number of people')
            st.pyplot(fig)

        plot_dataframes(df_new5, df_new6, df_new7, df_new8)

        st.markdown(
            "<h4 style='font-weight: bold;'>Some Statistics with Optimized Models</h4>",
            unsafe_allow_html=True)

        st.markdown('The table compares the results of different models for predicting the number of people in a certain situation. Based on the result, the observed number of people is highest for XGBoost Model and lowest for Random Forest Model. The average number of people predicted by each model is similar, ranging from 11 to 13. On the other hand, the confidence intervals for each model overlap, indicating that there is no significant difference between the predictions of each model. Overall, the results suggest that the different models perform similarly in predicting the number of people in this situation. However, it is important to note that this is just one example and the performance of each model may vary depending on the specific context and data used.')

        comparison = pd.DataFrame(
            {'Observed': [int(df_new5.n_people.sum()), int(df_new5.n_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new5["n_people"]).tconfint_mean()]],

             'Random Forest Model': [int(df_new5.number_people.sum()), int(df_new5.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new5["number_people"]).tconfint_mean()]],

             'XGBoost Model': [int(df_new6.number_people.sum()), int(df_new6.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new6["number_people"]).tconfint_mean()]],

             'CatBoost Model': [int(df_new7.number_people.sum()), int(df_new7.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new7["number_people"]).tconfint_mean()]],

             'Voting Regressor': [int(df_new8.number_people.sum()), int(df_new8.number_people.mean()), [round(value, 2) for value in sms.DescrStatsW(df_new8["number_people"]).tconfint_mean()]]},

            index=['Number of People', 'Average Number', 'Confidence Interval']
        )
        st.write(comparison)

else:

    st.subheader('Streamlit Application')
    # Upload the model
    model = joblib.load('models/05xgboost_optimized.pkl')
    df = pd.read_csv('one_hot_encodered_df.csv')
    df.drop('n_people', axis=1, inplace=True)

    # Define input fields
    permanent_temporal_population = st.number_input('Permit', min_value=0, max_value=1, step=1)
    sex = st.number_input('Gender', min_value=0, max_value=1, step=1)
    year_2020 = st.number_input('Year 2020', min_value=0, max_value=1, step=1)
    year_2021 = st.number_input('Year 2021', min_value=0, max_value=1, step=1)
    foreign_group_musician_artist_8_month = st.number_input('Foreign Group Musician Artist 8 Month', min_value=0,
                                                            max_value=1, step=1)
    foreign_group_resident_B = st.number_input('Foreign Group Resident B', min_value=0, max_value=1, step=1)
    foreign_group_service_providers_4_month = st.number_input('Foreign Group Service Providers 4 Month', min_value=0,
                                                              max_value=1, step=1)
    foreign_group_short_resident_4_12_month = st.number_input('Foreign Group Short Resident 4-12 Month', min_value=0,
                                                              max_value=1, step=1)
    foreign_group_short_resident_4_month = st.number_input('Foreign Group Short Resident 4 Month', min_value=0,
                                                           max_value=1, step=1)
    foreign_group_short_resident_L_12_month = st.number_input('Foreign Group Short Resident L-12 Month', min_value=0,
                                                              max_value=1, step=1)
    sector_agriculture_sector = st.number_input('Agriculture Sector', min_value=0, max_value=1, step=1)
    sector_industry_craft_sector = st.number_input('Industry Craft Sector', min_value=0, max_value=1, step=1)
    sector_services_sector = st.number_input('Services Sector', min_value=0, max_value=1, step=1)
    working_canton_Basel_City = st.number_input('Working Canton Basel City', min_value=0, max_value=1, step=1)
    working_canton_Basel_Country = st.number_input('Working Canton Basel Country', min_value=0, max_value=1, step=1)
    working_canton_Bern = st.number_input('Working Canton Bern', min_value=0, max_value=1, step=1)
    working_canton_Freiburg = st.number_input('Working Canton Freiburg', min_value=0, max_value=1, step=1)
    working_canton_Genf = st.number_input('Working Canton Genf', min_value=0, max_value=1, step=1)
    working_canton_Grisons = st.number_input('Working Canton Grisons', min_value=0, max_value=1, step=1)
    working_canton_Luzern = st.number_input('Working Canton Luzern', min_value=0, max_value=1, step=1)
    working_canton_Neuenburg = st.number_input('Working Canton Neuenburg', min_value=0, max_value=1, step=1)
    working_canton_Rare = st.number_input('Other Cantons', min_value=0, max_value=1, step=1)
    working_canton_Schaffhausen = st.number_input('Working Canton Schaffhausen', min_value=0, max_value=1, step=1)
    working_canton_Schwyz = st.number_input('Working Canton Schwyz', min_value=0, max_value=1, step=1)
    working_canton_Solothurn = st.number_input('Working Canton Solothurn', min_value=0, max_value=1, step=1)
    working_canton_St_Gallen = st.number_input('Working Canton Gallen', min_value=0, max_value=1, step=1)
    working_canton_Tessin = st.number_input('Working Canton Tessin', min_value=0, max_value=1, step=1)
    working_canton_Thurgau = st.number_input('Working Canton Thurgau', min_value=0, max_value=1, step=1)
    working_canton_Waadt = st.number_input('Working Canton Waadt', min_value=0, max_value=1, step=1)
    working_canton_Wallis = st.number_input('Working Canton Wallis', min_value=0, max_value=1, step=1)
    working_canton_Zug = st.number_input('Working Canton Zug', min_value=0, max_value=1, step=1)
    working_canton_Zurich = st.number_input('Working Canton Zurich', min_value=0, max_value=1, step=1)
    nation_Bulgaria = st.number_input('Bulgaria', min_value=0, max_value=1, step=1)
    nation_Czech_Republic = st.number_input('Czech Republic', min_value=0, max_value=1, step=1)
    nation_France = st.number_input('France', min_value=0, max_value=1, step=1)
    nation_Germany = st.number_input('Germany', min_value=0, max_value=1, step=1)
    nation_Hungary = st.number_input('Hungary', min_value=0, max_value=1, step=1)
    nation_Italy = st.number_input('Italy', min_value=0, max_value=1, step=1)
    nation_Netherlands = st.number_input('Netherlands', min_value=0, max_value=1, step=1)
    nation_Poland = st.number_input('Poland', min_value=0, max_value=1, step=1)
    nation_Portugal = st.number_input('Portugal', min_value=0, max_value=1, step=1)
    nation_Rare = st.number_input('Other Countries', min_value=0, max_value=1, step=1)
    nation_Romania = st.number_input('Romania', min_value=0, max_value=1, step=1)
    nation_Slovak_Republic = st.number_input('Slovak Republic', min_value=0, max_value=1, step=1)
    nation_Spain = st.number_input('Spain', min_value=0, max_value=1, step=1)
    nation_USA = st.number_input('USA', min_value=0, max_value=1, step=1)
    continent_America = st.number_input('America', min_value=0, max_value=1, step=1)
    continent_Asia = st.number_input('Asia', min_value=0, max_value=1, step=1)
    continent_Europe = st.number_input('Europe', min_value=0, max_value=1, step=1)
    continent_Rare = st.number_input('Other Continents', min_value=0, max_value=1, step=1)
    europa_third_country_EU_17 = st.number_input('Third Countries EU_17', min_value=0, max_value=1, step=1)
    europa_third_country_EU_2 = st.number_input('Third Countries EU_2', min_value=0, max_value=1, step=1)
    europa_third_country_EU_8 = st.number_input('Third Countries EU_8', min_value=0, max_value=1, step=1)
    europa_third_country_Rare = st.number_input('Other Third Countries', min_value=0, max_value=1, step=1)
    europa_third_country_third_countries = st.number_input('Third Countries', min_value=0, max_value=1, step=1)

    # Make a prediction
    if st.button('Predict'):
        input_data = pd.DataFrame({
            'permanent_temporal_population': [permanent_temporal_population],
            'sex': [sex],
            'year_2020': [year_2020],
            'year_2021': [year_2021],
            'foreign_group_musician_artist_8_month': [foreign_group_musician_artist_8_month],
            'foreign_group_resident_B': [foreign_group_resident_B],
            'foreign_group_service_providers_4_month': [foreign_group_service_providers_4_month],
            'foreign_group_short_resident_4_12_month': [foreign_group_short_resident_4_12_month],
            'foreign_group_short_resident_4_month': [foreign_group_short_resident_4_month],
            'foreign_group_short_resident_L_12_month': [foreign_group_short_resident_L_12_month],
            'sector_agriculture_sector': [sector_agriculture_sector],
            'sector_industry_craft_sector': [sector_industry_craft_sector],
            'sector_services_sector': [sector_services_sector],
            'working_canton_Basel_City': [working_canton_Basel_City],
            'working_canton_Basel_Country': [working_canton_Basel_Country],
            'working_canton_Bern': [working_canton_Bern],
            'working_canton_Freiburg': [working_canton_Freiburg],
            'working_canton_Genf': [working_canton_Genf],
            'working_canton_Grisons': [working_canton_Grisons],
            'working_canton_Luzern': [working_canton_Luzern],
            'working_canton_Neuenburg': [working_canton_Neuenburg],
            'working_canton_Rare': [working_canton_Rare],
            'working_canton_Schaffhausen': [working_canton_Schaffhausen],
            'working_canton_Schwyz': [working_canton_Schwyz],
            'working_canton_Solothurn': [working_canton_Solothurn],
            'working_canton_St_Gallen': [working_canton_St_Gallen],
            'working_canton_Tessin': [working_canton_Tessin],
            'working_canton_Thurgau': [working_canton_Thurgau],
            'working_canton_Waadt': [working_canton_Waadt],
            'working_canton_Wallis': [working_canton_Wallis],
            'working_canton_Zug': [working_canton_Zug],
            'working_canton_Zurich': [working_canton_Zurich],
            'nation_Bulgaria': [nation_Bulgaria],
            'nation_Czech_Republic': [nation_Czech_Republic],
            'nation_France': [nation_France],
            'nation_Germany': [nation_Germany],
            'nation_Hungary': [nation_Hungary],
            'nation_Italy': [nation_Italy],
            'nation_Netherlands': [nation_Netherlands],
            'nation_Poland': [nation_Poland],
            'nation_Portugal': [nation_Portugal],
            'nation_Rare': [nation_Rare],
            'nation_Romania': [nation_Romania],
            'nation_Slovak_Republic': [nation_Slovak_Republic],
            'nation_Spain': [nation_Spain],
            'nation_USA': [nation_USA],
            'continent_America': [continent_America],
            'continent_Asia': [continent_Asia],
            'continent_Europe': [continent_Europe],
            'continent_Rare': [continent_Rare],
            'europa_third_country_EU_17': [europa_third_country_EU_17],
            'europa_third_country_EU_2': [europa_third_country_EU_2],
            'europa_third_country_EU_8': [europa_third_country_EU_8],
            'europa_third_country_Rare': [europa_third_country_Rare],
            'europa_third_country_third_countries': [europa_third_country_third_countries]
        })

        prediction = model.predict(input_data)[0]
        if int(prediction) < 0:
            st.write('According to the provided conditions, the number of migrant potential to be able to come to Switzerland is 0.')
        else:
            st.write(f'Prediction Result: According to the provided conditions, the number of migrant potential to be able to come to Switzerland is {int(prediction)}.')
# endregion