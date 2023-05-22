# region 1. IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from helpers import Preprocessing

shap.initjs()
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import joblib
import optuna
import warnings

warnings.filterwarnings('ignore')
from lazypredict.Supervised import LazyRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
# endregion
# region 2. IMPORTING DATASETS
train1 = pd.read_csv('data/immigration_2019.csv', encoding='latin1', delimiter=';')
train2 = pd.read_csv('data/immigration_2020.csv', encoding='latin1', delimiter=';')
test = pd.read_csv('data/immigration_2021.csv', encoding='latin1', delimiter=';')
# endregion
# region 3. MERGING DATASETS
df_ = pd.concat([train1, train2, test], axis=0, ignore_index=True)
df = df_.drop(
    ['StaeNichtStaeBev_CD', 'Auslaendergruppe_CD', 'Sektor_CD', 'Geschlecht_CD', 'Arbeitskanton_CD', 'Nation_CD',
     'Kontinent_CD', 'EuropaDrittstaat_CD'], axis=1)
df.columns = [col.lower() for col in df.columns]
df.head()
df.tail()
# endregion
# region 4. HANDLING DATASET
# staenichtstaebev
df.loc[(df['staenichtstaebev'] == 'Ständige ausl. Bevölkerung'), 'staenichtstaebev'] = 'permanent'
df.loc[(df['staenichtstaebev'] == 'Nicht ständige ausl. Bevölkerung'), 'staenichtstaebev'] = 'temporal'

# auslaendergruppe
df.loc[df['auslaendergruppe'] == 'Aufenthalter (B)', 'auslaendergruppe'] = 'resident_B'
df.loc[
    df['auslaendergruppe'] == 'Kurzaufenthalter > 4 bis < 12 Monate', 'auslaendergruppe'] = 'short_resident_4_12_month'
df.loc[df['auslaendergruppe'] == 'Kurzaufenthalter (L) >= 12 Monate', 'auslaendergruppe'] = 'short_resident_L_12_month'
df.loc[df['auslaendergruppe'] == 'Kurzaufenthalter <= 4 Monate', 'auslaendergruppe'] = 'short_resident_4_month'
df.loc[
    df['auslaendergruppe'] == 'Dienstleistungserbringern <= 4 Monate', 'auslaendergruppe'] = 'service_providers_4_month'
df.loc[df['auslaendergruppe'] == 'Musiker und Künstler <= 8 Monate', 'auslaendergruppe'] = 'musician_artist_8_month'
df.loc[df['auslaendergruppe'] == 'Niedergelassene (C)', 'auslaendergruppe'] = 'settled_foreigners_C'

# sektor
df.loc[df['sektor'] == 'Sektor Dienstleistungen', 'sektor'] = 'services_sector'
df.loc[df['sektor'] == 'Sektor Industrie und Handwerk', 'sektor'] = 'industry_craft_sector'
df.loc[df['sektor'] == 'Sektor Landwirtschaft', 'sektor'] = 'agriculture_sector'
df.loc[df['sektor'] == 'Sektor unbekannt', 'sektor'] = 'unknown_sector'

# arbeitskanton
df.loc[df['arbeitskanton'] == 'Basel-Stadt', 'arbeitskanton'] = 'Basel_City'
df.loc[df['arbeitskanton'] == 'St. Gallen', 'arbeitskanton'] = 'St_Gallen'
df.loc[df['arbeitskanton'] == 'Basel-Land', 'arbeitskanton'] = 'Basel_Country'
df.loc[df['arbeitskanton'] == 'Appenzell A. Rh.', 'arbeitskanton'] = 'Appenzell_A_Rh'
df.loc[df['arbeitskanton'] == 'Appenzell I. Rh.', 'arbeitskanton'] = 'Appenzell_I_Rh'
df.loc[df['arbeitskanton'] == 'Zürich', 'arbeitskanton'] = 'Zurich'
df.loc[df['arbeitskanton'] == 'Graubünden', 'arbeitskanton'] = 'Grisons'
df.loc[df['arbeitskanton'] == 'Unbekannt', 'arbeitskanton'] = 'Unknown'

# kontinent
df.loc[df['kontinent'] == 'Europa', 'kontinent'] = 'Europe'
df.loc[df['kontinent'] == 'Asien', 'kontinent'] = 'Asia'
df.loc[df['kontinent'] == 'Amerika', 'kontinent'] = 'America'
df.loc[df['kontinent'] == 'Afrika', 'kontinent'] = 'Africa'
df.loc[df['kontinent'] == 'Ozeanien', 'kontinent'] = 'Oceania'
df.loc[df['kontinent'] == 'Herkunft unbek.', 'kontinent'] = 'unknown_origin'

# europadrittstaat
df.loc[df['europadrittstaat'] == 'EU-17', 'europadrittstaat'] = 'EU_17'
df.loc[df['europadrittstaat'] == 'EU-8', 'europadrittstaat'] = 'EU_8'
df.loc[df['europadrittstaat'] == 'EU-2', 'europadrittstaat'] = 'EU_2'
df.loc[df['europadrittstaat'] == 'EU-1', 'europadrittstaat'] = 'EU_1'
df.loc[df['europadrittstaat'] == 'Drittstaaten', 'europadrittstaat'] = 'third_countries'

# geschlecht
df.loc[df['geschlecht'] == 'Männlich', 'geschlecht'] = 'male'
df.loc[df['geschlecht'] == 'Weiblich', 'geschlecht'] = 'female'

# nation
df_ = pd.DataFrame(df.nation.value_counts()).reset_index()
ger_nations = list(df_['index'].values)
eng_nations = ['Germany', 'Italy', 'Poland', 'Romania', 'Portugal',
               'France', 'Spain', 'Hungary', 'Austria',
               'Slovak_Republic', 'Bulgaria', 'USA',
               'Czech_Republic', 'Netherlands', 'Croatia', 'Greece',
               'Belgium', 'India', 'China', 'Russia',
               'Slovenia', 'Ukraine', 'UnitedKingdom', 'Brazil',
               'Sweden', 'Lithuania', 'Canada', 'Ireland', 'Turkey', 'Finland',
               'Serbia', 'Latvia', 'Denmark', 'Japan', 'Great_Britain',
               'Kosovo', 'Mexico', 'Australia', 'Iran', 'Afghanistan', 'Eritrea',
               'Syria', 'South_Africa', 'Norway', 'Colombia', 'Philippines',
               'North_Macedonia', 'Luxembourg', 'South_Korea', 'Argentina',
               'Estonia', 'Malaysia', 'Bosnia_Herzegovina', 'Egypt',
               'Tunisia', 'Singapore', 'SriLanka', 'Morocco', 'Thailand',
               'Israel', 'Somalia', 'Albania', 'Indonesia', 'Lebanon', 'Taiwan',
               'Pakistan', 'New Zealand', 'Ethiopia', 'Iraq', 'Liechtenstein',
               'Cyprus', 'Chile', 'Nigeria', 'Vietnam', 'Belarus', 'Algeria',
               'CongoDR', 'Peru', 'Kazakhstan', 'Ecuador', 'Iceland', 'Cameroon',
               'Kenya', 'Bolivia', 'Georgia', "CôtedIvoire", 'Jordan',
               'Ghana', 'Venezuela', 'Senegal', 'Malta', 'Mongolia', 'Zimbabwe',
               'Armenia', 'Nepal', 'Cuba', 'CostaRica', 'StateUnknown',
               'Bangladesh', 'DominicanRepublic', 'SaudiArabia', 'Angola',
               'Azerbaijan', 'Montenegro', 'Uruguay', 'Moldova', 'Sudan',
               'Madagascar', 'BurkinaFaso', 'Guatemala', 'Kyrgyzstan',
               'Paraguay', 'Togo', 'Guinea', 'Rwanda', 'WithoutNationality',
               'Tanzania', 'Yemen', 'Mali', 'Benin', 'Uzbekistan', 'Libya',
               'Honduras', 'ElSalvador', 'Mauritius', 'Zambia', 'Namibia',
               'Uganda', 'Burundi', 'Haiti', 'Panama', 'Gambia', 'Tajikistan',
               'TrinidadTobago', 'Stateless', 'Congo', 'Nicaragua',
               'Myanmar', 'GuineaBissau', 'Malawi', 'Mozambique', 'Liberia',
               'Kuwait', 'Gabon', 'Jamaica', 'Cambodia', 'Bahrain', 'Laos',
               'CapeVerde', 'Botswana', 'SierraLeone', 'Mauritania',
               'stKitts and Nevis', 'Bahamas', 'Bhutan', 'Grenada', 'Chad',
               'Maldives', 'Dominica', 'V_A_Emirates', 'Seychelles',
               'CentralAfriRepublic', 'Monaco', 'Djibouti', 'SouthSudan',
               'SaoTomePrincipe', 'Niger', 'Comoros', 'Oman', 'SanMarino',
               'Barbados', 'Fiji', 'Turkmenistan', 'Andorra', 'Brunei']
for ger_nation_index, ger_nation_value in enumerate(ger_nations):
    # print(ger_nation_index, ger_nation_value)
    for eng_nation_index, eng_nations_value in enumerate(eng_nations):
        if ger_nation_index == eng_nation_index:
            df.loc[df['nation'] == ger_nation_value, 'nation'] = eng_nations_value
# endregion
# region 5. CHANGING COLUMN NAMES
df.columns = ['year', 'permanent_temporal_population', 'foreign_group', 'sector', 'sex', 'working_canton', 'nation',
              'continent', 'europa_third_country', 'n_people']
# endregion
# region 6. SAVING THE NEW DATASET AFTER HANDLING
df.to_csv('immigration_to_switzerland.csv', index=False)
# endregion
# region 7. IMPORTING NEW DATASET
df = pd.read_csv('immigration_to_switzerland.csv')
# endregion
# region 8. DIVIDING THE DATA SET AS TRAIN AND TEST SETS
def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:22682], all_data.loc[22683:]


train, test = divide_df(df)
# endregion
# region 9. TESTSET FOR PREDICTION
df_people = pd.DataFrame(test['n_people'])


# endregion
# region 10. EDA AND PREPROCESSING FUNCTIONS
prep_clas = Preprocessing(df)
# endregion
# region 11. EDA
prep_clas.check_df()
cat_cols, num_cols, cat_but_car = prep_clas.grab_col_names()

for col in cat_cols:
    prep_clas.cat_summary(col)

prep_clas.cat_summary('nation', False)
# endregion
# region 12. SIMPLE FEATURE ENGINEERING
# Rare analyzing
prep_clas.rare_analyzer('n_people', cat_cols)

prep_clas.rare_analyzer('n_people', cat_but_car)

new_df = prep_clas.rare_encoder(0.02)
new_df.to_csv('immigration.csv', index=False)

prep_clas1 = Preprocessing(new_df)

# Label encoding
binary_cols = [col for col in new_df.columns if new_df[col].dtypes == 'O' and new_df[col].nunique() == 2]
for col in binary_cols:
    prep_clas1.label_encoder(col)

# One-Hot encoding
ohe_cols = [col for col in new_df.columns if 35 >= new_df[col].nunique() > 2]
df = prep_clas1.one_hot_encoder(ohe_cols, True)
df.to_csv('one_hot_encodered_df.csv', index=False)
# endregion
# region 13. BASE MODELING
def base_models_kfold(X, y, n_splits=5, random_state=12345, save=True):
    print('Base models with cross-validation...')

    all_models = []

    models = [("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('ET', ExtraTreeRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor()),
              ("LightGBM", LGBMRegressor()),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, model in models:
        rmse_train_scores = []
        rmse_val_scores = []
        mae_train_scores = []
        mae_val_scores = []
        r2_train_scores = []
        r2_val_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            y_pred_train = model.predict(X_train)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_val = mean_absolute_error(y_val, y_pred_val)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            r2_val = r2_score(y_val, y_pred_val)
            r2_train = r2_score(y_train, y_pred_train)

            rmse_train_scores.append(rmse_train)
            rmse_val_scores.append(rmse_val)
            mae_train_scores.append(mae_train)
            mae_val_scores.append(mae_val)
            r2_train_scores.append(r2_train)
            r2_val_scores.append(r2_val)

        values = dict(MODEL=name,
                      RMSE_TRAIN=np.mean(rmse_train_scores),
                      RMSE_VAL=np.mean(rmse_val_scores),
                      MAE_TRAIN=np.mean(mae_train_scores),
                      MAE_VAL=np.mean(mae_val_scores),
                      R2_TRAIN=np.mean(r2_train_scores),
                      R2_VAL=np.mean(r2_val_scores))

        all_models.append(values)

    sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)

    # Set up the subplots
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(8, 9))
    fig.suptitle('Performance Metrics')

    # Plot the bar charts
    sns.barplot(x='RMSE_TRAIN', y='MODEL', data=all_models_df, ax=axs[0, 0])
    sns.barplot(x='RMSE_VAL', y='MODEL', data=all_models_df, ax=axs[0, 1])
    sns.barplot(x='MAE_TRAIN', y='MODEL', data=all_models_df, ax=axs[1, 0])
    sns.barplot(x='MAE_VAL', y='MODEL', data=all_models_df, ax=axs[1, 1])
    sns.barplot(x='R2_TRAIN', y='MODEL', data=all_models_df, ax=axs[2, 0])
    sns.barplot(x='R2_VAL', y='MODEL', data=all_models_df, ax=axs[2, 1])

    # Set the subplot titles
    axs[0, 0].set_title('RMSE_TRAIN')
    axs[0, 1].set_title('RMSE_VAL')
    axs[1, 0].set_title('MAE_TRAIN')
    axs[1, 1].set_title('MAE_VAL')
    axs[2, 0].set_title('R2_TRAIN')
    axs[2, 1].set_title('R2_VAL')

    # Adjust the layout
    plt.tight_layout()
    plt.show(block=True)

    if save:
        plt.savefig('image/01basic_train_models.png')

    print(all_models_df)


# calling the dataset
df = pd.read_csv('one_hot_encodered_df.csv')

# dividing the dataset
train, test = divide_df(df)

# for the train dataset
y = train['n_people']
X = train.drop('n_people', axis=1)

# calling the function
base_models_kfold(X, y)


#          MODEL  RMSE_TRAIN  RMSE_VAL  MAE_TRAIN  MAE_VAL  R2_TRAIN  R2_VAL
# 6           RF      13.155    24.103      4.093    7.922     0.939   0.784
# 11    CatBoost      18.746    24.334      7.660    8.770     0.877   0.781
# 9      XGBoost      17.428    24.971      6.920    8.608     0.893   0.770
# 4           ET      10.104    26.824      2.056    8.016     0.964   0.731
# 5         CART      10.104    26.898      2.056    8.029     0.964   0.728
# 10    LightGBM      32.145    37.155      9.598   10.395     0.638   0.509
# 3          KNN      30.499    38.894      8.373   10.643     0.673   0.458
# 8          GBM      43.401    45.407     13.073   13.365     0.340   0.267
# 0        Ridge      49.771    49.603     18.093   18.145     0.133   0.130
# 1        Lasso      51.651    51.412     15.534   15.549     0.066   0.065
# 7          SVR      52.130    51.884     10.347   10.471     0.049   0.048
# 2   ElasticNet      52.462    52.192     15.875   15.881     0.036   0.037

# endregion
# region 14. BASIC RF, XGBOOST, and CATBOOST
def evaluate_models(X, y, model, n_splits=5, random_state=12345):
    # Split data into training and test sets
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_metrics = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # RF basic model
        model_basic = model
        model_basic.fit(X_train, y_train)

        # Calculate evaluation metrics for RF final model
        y_pred_test = model_basic.predict(X_test)
        y_pred_train = model_basic.predict(X_train)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        values = dict(MODEL=type(model), RMSE_TRAIN=np.mean(rmse_train), RMSE_VAL=np.mean(rmse_test),
                      MAE_TRAIN=np.mean(mae_train), MAE_VAL=np.mean(mae_test),
                      R2_TRAIN=np.mean(r2_train), R2_VAL=np.mean(r2_test))
        all_metrics.append(values)

    # Create a DataFrame with all evaluation metrics
    all_metrics_df = pd.DataFrame(all_metrics)

    return all_metrics_df.mean()


models = [('RandomForest', RandomForestRegressor(random_state=12345)), ('XGBoost', XGBRegressor(random_state=12345)),
          ('CatBoost', CatBoostRegressor(random_state=12345, verbose=False))]
for name, model in models:
    all_metrics_df = evaluate_models(X, y, model)
    print(name, '\n', all_metrics_df)


# RandomForest
#  RMSE_TRAIN   13.306
# RMSE_VAL     24.063
# MAE_TRAIN     4.100
# MAE_VAL       7.884
# R2_TRAIN      0.937
# R2_VAL        0.785
# dtype: float64
# XGBoost
#  RMSE_TRAIN   17.428
# RMSE_VAL     24.971
# MAE_TRAIN     6.920
# MAE_VAL       8.608
# R2_TRAIN      0.893
# R2_VAL        0.770
# dtype: float64
# CatBoost
#  RMSE_TRAIN   18.783
# RMSE_VAL     24.225
# MAE_TRAIN     7.648
# MAE_VAL       8.742
# R2_TRAIN      0.876
# R2_VAL        0.783
# dtype: float64
# endregion
# region 15. PLOT IMPORTANCE OF BASIC MODELS
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'image/importance_{type(model).__name__}.png')


rf_basic = models[0][1]
xgboost_basic = models[1][1]
catboost_basic = models[2][1]
models = [rf_basic, xgboost_basic, catboost_basic]
for model in models:
    plot_importance(model, X, save=True)
# endregion
# region 16. SAVING and CALLING BASIC MODELS
# region saving models
joblib.dump(rf_basic, 'models/01rf_basic.pkl')
joblib.dump(xgboost_basic, 'models/02xgboost_basic.pkl')
joblib.dump(catboost_basic, 'models/03catboost_basic.pkl')
# endregion

# region calling models
basic_rf = joblib.load('models/01rf_basic.pkl')
basic_xgboost = joblib.load('models/02xgboost_basic.pkl')
basic_catboost = joblib.load('models/03catboost_basic.pkl')
# endregion
# endregion
# # region 17. SHAP ANALYSIS FOR FEATURE IMPORTANCE
# models = [basic_rf, basic_catboost, basic_xgboost]
# names = ['basic_rf', 'basic_catboost', 'basic_xgboost']
# for i, model in enumerate(models):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     shap.summary_plot(shap_values, X)
#     plt.savefig('image/shap_summary_{}.png'.format(names[i]))
# # endregion
#
# # region 18. LAZY PREDICTION FOR BASIC DATASET
# X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.25, random_state=12345)
# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(X_train_, X_test_, y_train_, y_test_)
# models
# # endregion
# region 19. VOTING REGRESSOR FOR BASIC MODELS
def voting_regressor(best_models, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=12345)
    rmse_scores, mae_scores, r2_scores = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
        voting_regressor.fit(X_train, y_train)

        y_pred = voting_regressor.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        r2_scores.append(r2_score(y_val, y_pred))

    rmse = np.mean(rmse_scores)
    mae = np.mean(mae_scores)
    r2 = np.mean(r2_scores)
    print(f'RMSE={round(rmse, 3)}, MAE={round(mae, 3)}, R_SQUARE={round(r2, 3)}')


best_models = {'catboost': basic_catboost, 'xgboost': basic_xgboost, 'random_forest': basic_rf}
basic_voting_regressor = voting_regressor(best_models, X, y)

# RMSE=23.156, MAE=7.796, R_SQUARE=0.802
# endregion
# region 20. SAVING and CALLING THE VOTING MODELS AS PKL FILE
# saving
joblib.dump(basic_voting_regressor, 'models/basic_voting.pkl')

# calling
basic_voting = VotingRegressor([('catboost', basic_catboost),
                                ('xgboost', basic_xgboost),
                                ('random_forest', basic_rf)])
basic_voting.fit(X, y)
# endregion
# region 21. PREDICTION WITH BASIC MODELS
test.drop('n_people', axis=1, inplace=True)
df1 = test.copy()
df1['number_people'] = basic_rf.predict(df1)

df2 = test.copy()
df2['number_people'] = basic_xgboost.predict(df2)

df3 = test.copy()
df3['number_people'] = basic_catboost.predict(df3)

df4 = test.copy()
df4['number_people'] = basic_voting.predict(df4)
# df1['number_people'] = np.log1p(df1['number_people'])
# df2['number_people'] = np.log1p(df2['number_people'])
# df3['number_people'] = np.log1p(df3['number_people'])

df_new1 = pd.concat([df1, df_people], axis=1)
df_new2 = pd.concat([df2, df_people], axis=1)
df_new3 = pd.concat([df3, df_people], axis=1)
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

    plt.savefig(f'image/08Prediction with Basic Models')
    plt.show(block=True)


plot_dataframes(df_new1, df_new2, df_new3, df_new4)


# endregion
# region 22. HYPERPARAMETER OPTIMIZATION FOR BASIC MODELING
# region CATBOOST REGRESSOR
def objective_catboost(trial, X, y, n_splits=5):
    # param = {
    #     'iterations': trial.suggest_int("iterations", 100, 1000),
    #     'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
    #     'depth': trial.suggest_int("depth", 4, 10),
    #     'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
    #     'bootstrap_type': trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
    #     'random_strength': trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
    #     'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 10.0),
    #     'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
    #     'od_wait': trial.suggest_int("od_wait", 10, 50),
    #     'random_state': 12345
    # }

    best_params = {'iterations': 999,
                   'learning_rate': 0.07739418741960805,
                   'depth': 10,
                   'l2_leaf_reg': 0.0015274514470779742,
                   'bootstrap_type': 'Bayesian',
                   'random_strength': 0.03888213726127089,
                   'bagging_temperature': 0.026710982977804443,
                   'od_type': 'Iter',
                   'od_wait': 35,
                   'random_state': 12345}

    mae = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostRegressor(verbose=False, **best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae)

# Create the study
study_catboost = optuna.create_study(direction='minimize', study_name='regression')
study_catboost.optimize(lambda trial: objective_catboost(trial, X, y), n_trials=1, show_progress_bar=True)

# Print the best parameters
print(f'best_params = {study_catboost.best_params}')
# endregion

# region XGBOOST REGRESSOR
def objective_xgboost(trial, X, y, n_splits=5):
    # param = {
    #     'max_depth': trial.suggest_int('max_depth', 1, 10),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
    #     'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    #     'gamma': trial.suggest_float('gamma', 0.01, 1.0),
    #     'subsample': trial.suggest_float('subsample', 0.01, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
    #     'random_state': 12345
    # }

    best_params = {'max_depth': 7,
                   'learning_rate': 0.18852428804431892,
                   'n_estimators': 801,
                   'min_child_weight': 1,
                   'gamma': 0.3155522440479948,
                   'subsample': 0.6013061915476853,
                   'colsample_bytree': 0.9979459121019403,
                   'reg_alpha': 0.20484121586886916,
                   'reg_lambda': 0.53404115911961,
                   'random_state': 12345}


    mae = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae)

# Create the study
study_xgboost = optuna.create_study(direction='minimize', study_name='regression')
study_xgboost.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=1, show_progress_bar=True)

# Print the best parameters
print(f'best_params = {study_xgboost.best_params}')
# endregion

# region RANDOM FOREST REGRESSOR
def objective_rf(trial, X, y, n_splits=5):
    # param = {'n_estimators': trial.suggest_int('n_estimators', 10, 2000),
    #          'max_depth': trial.suggest_int('max_depth', 1, 50),
    #          'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    #          'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    #          'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    #          'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    #          'random_state': 12345
    #          }

    best_params = {'n_estimators': 10,
                   'max_depth': 28,
                   'min_samples_split': 8,
                   'min_samples_leaf': 1,
                  'max_features': 'sqrt',
                   'bootstrap': False,
                   'random_state': 12345}

    mae = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae)

# Create the study
study_rf = optuna.create_study(direction='minimize', study_name='regression')
study_rf.optimize(lambda trial: objective_rf(trial, X, y), n_trials=1, show_progress_bar=True)

# Print the best parameters
print(f'best_params = {study_rf.best_params}')
# endregion
# endregion
# region 23. COMPARISON OF SELECTED AND OPTIMIZED MODELS
catboost_params = study_catboost.best_params
xgboost_params = study_xgboost.best_params
rf_params = study_rf.best_params
models = [('CATBOOST', CatBoostRegressor(verbose=False, **catboost_params)),
          ('XGBOOST', XGBRegressor(**xgboost_params)),
          ('RandomForest', RandomForestRegressor(**rf_params))]

all_metrics = []
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
for name, model in models:
    rmse_train_list = []
    rmse_test_list = []
    mae_train_list = []
    mae_test_list = []
    r2_train_list = []
    r2_test_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
        mae_train_list.append(mae_train)
        mae_test_list.append(mae_test)
        r2_train_list.append(r2_train)
        r2_test_list.append(r2_test)
    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_test = np.mean(rmse_test_list)
    mean_mae_train = np.mean(mae_train_list)
    mean_mae_test = np.mean(mae_test_list)
    mean_r2_train = np.mean(r2_train_list)
    mean_r2_test = np.mean(r2_test_list)
    values = dict(MODEL=name, RMSE_TRAIN=mean_rmse_train, RMSE_TEST=mean_rmse_test, MAE_TRAIN=mean_mae_train,
                  MAE_TEST=mean_mae_test, R2_TRAIN=mean_r2_train, R2_TEST=mean_r2_test)
    all_metrics.append(values)

sort_method = True
all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df = all_metrics_df.sort_values(all_metrics_df.columns[2], ascending=sort_method)

#           MODEL  RMSE_TRAIN  RMSE_TEST  MAE_TRAIN  MAE_TEST  R2_TRAIN  R2_TEST
# 0      CATBOOST      11.376     19.868      3.894     6.744     0.954    0.851
# 1       XGBOOST      11.214     21.534      3.968     7.622     0.956    0.832
# 2  RandomForest      22.862     33.465      6.359     8.887     0.816    0.600
# endregion
# region 24. VISUALIZATION OF THE SELECTED MODELS
# Set up the subplots
fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(8, 9))
fig.suptitle('Performance Metrics')

# Plot the bar charts
sns.barplot(x='RMSE_TRAIN', y='MODEL', data=all_metrics_df, ax=axs[0, 0])
sns.barplot(x='RMSE_TEST', y='MODEL', data=all_metrics_df, ax=axs[0, 1])
sns.barplot(x='MAE_TRAIN', y='MODEL', data=all_metrics_df, ax=axs[1, 0])
sns.barplot(x='MAE_TEST', y='MODEL', data=all_metrics_df, ax=axs[1, 1])
sns.barplot(x='R2_TRAIN', y='MODEL', data=all_metrics_df, ax=axs[2, 0])
sns.barplot(x='R2_TEST', y='MODEL', data=all_metrics_df, ax=axs[2, 1])

# Adjust the layout
plt.tight_layout()
plt.savefig('image/performance_metrics.png')
plt.show(block=True)

# endregionc
# region 25. PLOT IMPORTANCE FOR OPTIMIZED MODELS
catboost_optimized = CatBoostRegressor(verbose=False, **catboost_params).fit(X, y)
xgboost_optimized = XGBRegressor(**xgboost_params).fit(X, y)
rf_optimized = RandomForestRegressor(**rf_params).fit(X, y)
models = [catboost_optimized, xgboost_optimized, rf_optimized]
for model in models:
    plot_importance(model, X, save=True)
# endregion
# region 26. SAVING AND CALLING THE OPTIMIZED MODELS AS PKL FILES
# saving the models
joblib.dump(catboost_optimized, 'models/04catboost_optimized.pkl')
joblib.dump(xgboost_optimized, 'models/05xgboost_optimized.pkl')
joblib.dump(rf_optimized, 'models/06rf_optimized.pkl')

# calling the models
optimized_catboost = joblib.load('models/04catboost_optimized.pkl')
optimized_xgboost = joblib.load('models/05xgboost_optimized.pkl')
optimized_rf = joblib.load('models/06rf_optimized.pkl')
# endregion
# region 27. VOTING REGRESSOR FOR OPTIMIZED MODELS
def voting_regressor(best_models, X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        voting_regressor.fit(X_train, y_train)

        y_pred = voting_regressor.predict(X_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    rmse = np.mean(rmse_scores)
    mae = np.mean(mae_scores)
    r2 = np.mean(r2_scores)

    print(f'RMSE={rmse}, MAE={mae}, R_SQUARE={r2}')

# Example usage:
best_models = {'catboost': catboost_optimized, 'xgboost': xgboost_optimized}
best_models1 = {'catboost': catboost_optimized, 'xgboost': xgboost_optimized, 'random_forest': rf_optimized}
voting_regressor(best_models, X, y)
voting_regressor(best_models1, X, y)

# RMSE=17.440386108669646, MAE=6.441983958957268, R_SQUARE=0.8757280668254269
# RMSE=20.615978557120854, MAE=6.677255947250641, R_SQUARE=0.834620647276531
# endregion
# region 28. SAVING AND CALLING THE VOTING REGRESSOR
# Optimized voting regressor
voting_optimized = VotingRegressor([('catboost', catboost_optimized), ('xgboost', xgboost_optimized)]).fit(X, y)

# saving the voting regressor
joblib.dump(voting_optimized, 'models/08voting_optimized.pkl')

# calling the voting regressor
optimized_voting = joblib.load('models/08voting_optimized.pkl')
# endregion
# region 29. PREDICTION WITH OPTIMIZED MODELS
df5 = test.copy()
df5['number_people'] = optimized_catboost.predict(df5)

df6 = test.copy()
df6['number_people'] = optimized_xgboost.predict(df6)

df7 = test.copy()
df7['number_people'] = optimized_rf.predict(df7)

df8 = test.copy()
df8['number_people'] = voting_optimized.predict(df8)

df_new5 = pd.concat([df5, df_people], axis=1)
df_new6 = pd.concat([df6, df_people], axis=1)
df_new7 = pd.concat([df7, df_people], axis=1)
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
        f"R2 values:\nOptimized_{type(optimized_catboost).__name__} = {r2_1}\nOptimized_{type(optimized_xgboost).__name__} = {r2_2}\nOptimized_{type(optimized_rf).__name__} = {r2_3}\nOptimized_{type(voting_optimized).__name__} = {r2_4}")
    axs[0].set(xlabel='Observed number of people', ylabel='Predicted number of people')
    axs[1].set(xlabel='Observed number of people', ylabel='Predicted number of people')
    axs[2].set(xlabel='Observed number of people', ylabel='Predicted number of people')
    axs[3].set(xlabel='Observed number of people', ylabel='Predicted number of people')

    plt.savefig(f'image/Figure_10_Prediction with Optimized Models')
    plt.show(block=True)

plot_dataframes(df_new5, df_new6, df_new7, df_new8)
# endregion
# # region 30. FEATURE ENGINEERING
# # region importing dataframe
# df = pd.read_csv('immigration_to_switzerland.csv')
# # endregion
#
# # region VARIABLE PRODUCTION WITH YEAR
# # region year and foreign group
# df.loc[(df['year'] == 2019) & (df['foreign_group'] == 'resident_B'), 'NEW_YEAR_RESIDENT'] = 'firstyear_RB'
# df.loc[(df['year'] == 2019) & (
#             df['foreign_group'] == 'short_resident_4_12_month'), 'NEW_YEAR_RESIDENT'] = 'firstyear_R4_12M'
# df.loc[(df['year'] == 2019) & (
#             df['foreign_group'] == 'short_resident_L_12_month'), 'NEW_YEAR_RESIDENT'] = 'firstyear_RL12M'
# df.loc[(df['year'] == 2019) & (df['foreign_group'] == 'short_resident_4_month'), 'NEW_YEAR_RESIDENT'] = 'firstyear_R4M'
# df.loc[
#     (df['year'] == 2019) & (df['foreign_group'] == 'service_providers_4_month'), 'NEW_YEAR_RESIDENT'] = 'firstyear_SP4M'
# df.loc[
#     (df['year'] == 2019) & (df['foreign_group'] == 'musician_artist_8_month'), 'NEW_YEAR_RESIDENT'] = 'firstyear_MA8M'
# df.loc[(df['year'] == 2019) & (
#             df['foreign_group'] == 'settled_foreigners_C'), 'NEW_YEAR_RESIDENT'] = 'firstyear_settled_foreigners_C'
#
# df.loc[(df['year'] == 2020) & (df['foreign_group'] == 'resident_B'), 'NEW_YEAR_RESIDENT'] = 'secondyear_RB'
# df.loc[(df['year'] == 2020) & (
#             df['foreign_group'] == 'short_resident_4_12_month'), 'NEW_YEAR_RESIDENT'] = 'secondyear_R4_12M'
# df.loc[(df['year'] == 2020) & (
#             df['foreign_group'] == 'short_resident_L_12_month'), 'NEW_YEAR_RESIDENT'] = 'secondyear_RL12M'
# df.loc[(df['year'] == 2020) & (df['foreign_group'] == 'short_resident_4_month'), 'NEW_YEAR_RESIDENT'] = 'secondyear_R4M'
# df.loc[(df['year'] == 2020) & (
#             df['foreign_group'] == 'service_providers_4_month'), 'NEW_YEAR_RESIDENT'] = 'secondyear_SP4M'
# df.loc[
#     (df['year'] == 2020) & (df['foreign_group'] == 'musician_artist_8_month'), 'NEW_YEAR_RESIDENT'] = 'secondyear_MA8M'
# df.loc[(df['year'] == 2020) & (
#             df['foreign_group'] == 'settled_foreigners_C'), 'NEW_YEAR_RESIDENT'] = 'secondyear_settled_foreigners_C'
#
# df.loc[(df['year'] == 2021) & (df['foreign_group'] == 'resident_B'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_RB'
# df.loc[(df['year'] == 2021) & (
#             df['foreign_group'] == 'short_resident_4_12_month'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_R4_12M'
# df.loc[(df['year'] == 2021) & (
#             df['foreign_group'] == 'short_resident_L_12_month'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_RL12M'
# df.loc[(df['year'] == 2021) & (df['foreign_group'] == 'short_resident_4_month'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_R4M'
# df.loc[
#     (df['year'] == 2021) & (df['foreign_group'] == 'service_providers_4_month'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_SP4M'
# df.loc[
#     (df['year'] == 2021) & (df['foreign_group'] == 'musician_artist_8_month'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_MA8M'
# df.loc[(df['year'] == 2021) & (
#             df['foreign_group'] == 'settled_foreigners_C'), 'NEW_YEAR_RESIDENT'] = 'thirdyear_settled_foreigners_C'
# # endregion
#
# # region year permanent and temporal population
# df.loc[(df['year'] == 2019) & (df['permanent_temporal_population'] == 'permanent'), 'IS_PERMANENT'] = '2019_yes'
# df.loc[(df['year'] == 2019) & (df['permanent_temporal_population'] == 'temporal'), 'IS_PERMANENT'] = '2019_no'
# df.loc[(df['year'] == 2020) & (df['permanent_temporal_population'] == 'permanent'), 'IS_PERMANENT'] = '2020_yes'
# df.loc[(df['year'] == 2020) & (df['permanent_temporal_population'] == 'temporal'), 'IS_PERMANENT'] = '2020_no'
# df.loc[(df['year'] == 2021) & (df['permanent_temporal_population'] == 'permanent'), 'IS_PERMANENT'] = '2021_yes'
# df.loc[(df['year'] == 2021) & (df['permanent_temporal_population'] == 'temporal'), 'IS_PERMANENT'] = '2021_no'
# # endregion
#
# # region year and sector
# df.loc[(df['year'] == 2019) & (df['sector'] == 'services_sector'), 'NEW_YEAR_SECTOR'] = '2019_service'
# df.loc[(df['year'] == 2020) & (df['sector'] == 'services_sector'), 'NEW_YEAR_SECTOR'] = '2020_service'
# df.loc[(df['year'] == 2021) & (df['sector'] == 'services_sector'), 'NEW_YEAR_SECTOR'] = '2021_service'
#
# df.loc[(df['year'] == 2019) & (df['sector'] == 'industry_craft_sector'), 'NEW_YEAR_SECTOR'] = '2019_industry_craft'
# df.loc[(df['year'] == 2020) & (df['sector'] == 'industry_craft_sector'), 'NEW_YEAR_SECTOR'] = '2020_industry_craft'
# df.loc[(df['year'] == 2021) & (df['sector'] == 'industry_craft_sector'), 'NEW_YEAR_SECTOR'] = '2021_industry_craft'
#
# df.loc[(df['year'] == 2019) & (df['sector'] == 'agriculture_sector'), 'NEW_YEAR_SECTOR'] = '2019_agriculture'
# df.loc[(df['year'] == 2020) & (df['sector'] == 'agriculture_sector'), 'NEW_YEAR_SECTOR'] = '2020_agriculture'
# df.loc[(df['year'] == 2021) & (df['sector'] == 'agriculture_sector'), 'NEW_YEAR_SECTOR'] = '2021_agriculture'
#
# df.loc[(df['year'] == 2019) & (df['sector'] == 'unknown_sector'), 'NEW_YEAR_SECTOR'] = '2019_unknown_sector'
# df.loc[(df['year'] == 2020) & (df['sector'] == 'unknown_sector'), 'NEW_YEAR_SECTOR'] = '2020_unknown_sector'
# df.loc[(df['year'] == 2021) & (df['sector'] == 'unknown_sector'), 'NEW_YEAR_SECTOR'] = '2021_unknown_sector'
# # endregion
#
# # region year and sex
# df.loc[(df['sex'] == 'male') & (df['year'] == 2019), 'YEAR_SEX'] = '2019_male'
# df.loc[(df['sex'] == 'female') & (df['year'] == 2019), 'YEAR_SEX'] = '2019_female'
# df.loc[(df['sex'] == 'male') & (df['year'] == 2020), 'YEAR_SEX'] = '2020_male'
# df.loc[(df['sex'] == 'female') & (df['year'] == 2020), 'YEAR_SEX'] = '2020_female'
# df.loc[(df['sex'] == 'male') & (df['year'] == 2021), 'YEAR_SEX'] = '2021_male'
# df.loc[(df['sex'] == 'female') & (df['year'] == 2021), 'YEAR_SEX'] = '2021_female'
# # endregion
#
# # region working canton
# df.loc[df['working_canton'] == 'Zurich', 'NEW_SUBDIVISION'] = 'Zurich'
# df.loc[df['working_canton'].isin(['St_Gallen', 'Thurgau', 'Appenzell_A_Rh', 'Appenzell_I_Rh', 'Glarus', 'Schaffhausen',
#                                   'Grisons']), 'NEW_SUBDIVISION'] = 'Eastern_Switzerland'
# df.loc[df['working_canton'].isin(
#     ['Luzern', 'Uri', 'Obwalden', 'Nidwalden', 'Schwyz', 'Zug']), 'NEW_SUBDIVISION'] = 'Central_Switzerland'
# df.loc[df['working_canton'].isin(
#     ['Aargau', 'Basel_Country', 'Basel_City']), 'NEW_SUBDIVISION'] = 'Northwestern_Switzerland'
# df.loc[df['working_canton'].isin(
#     ['Bern', 'Solothurn', 'Freiburg', 'Jura', 'Neuenburg']), 'NEW_SUBDIVISION'] = 'Espace_Mittelland'
# df.loc[df['working_canton'].isin(['Wallis', 'Genf', 'Waadt']), 'NEW_SUBDIVISION'] = 'Lake_Geneva'
# df.loc[df['working_canton'] == 'Tessin', 'NEW_SUBDIVISION'] = 'Tessin'
# df.loc[df['working_canton'] == 'Unknown', 'NEW_SUBDIVISION'] = 'Unknown'
# # endregion
#
# # region year working canton
# df.loc[(df['year'] == 2019) & (df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_YEAR_SUBDIVISION'] = '2019_Lake_Geneva'
# df.loc[(df['year'] == 2019) & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2019_Eastern_Switzerland'
# df.loc[(df['year'] == 2019) & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_YEAR_SUBDIVISION'] = '2019_Espace_Mittelland'
# df.loc[(df['year'] == 2019) & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2019_Central_Switzerland'
# df.loc[(df['year'] == 2019) & (df[
#                                    'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2019_Northwestern_Switzerland'
# df.loc[(df['year'] == 2019) & (df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_YEAR_SUBDIVISION'] = '2019_Zurich'
# df.loc[(df['year'] == 2019) & (df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_YEAR_SUBDIVISION'] = '2019_Tessin'
# df.loc[(df['year'] == 2019) & (df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_YEAR_SUBDIVISION'] = '2019_Unknown'
#
# df.loc[(df['year'] == 2020) & (df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_YEAR_SUBDIVISION'] = '2020_Lake_Geneva'
# df.loc[(df['year'] == 2020) & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2020_Eastern_Switzerland'
# df.loc[(df['year'] == 2020) & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_YEAR_SUBDIVISION'] = '2020_Espace_Mittelland'
# df.loc[(df['year'] == 2020) & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2020_Central_Switzerland'
# df.loc[(df['year'] == 2020) & (df[
#                                    'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2020_Northwestern_Switzerland'
# df.loc[(df['year'] == 2020) & (df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_YEAR_SUBDIVISION'] = '2020_Zurich'
# df.loc[(df['year'] == 2020) & (df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_YEAR_SUBDIVISION'] = '2020_Tessin'
# df.loc[(df['year'] == 2020) & (df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_YEAR_SUBDIVISION'] = '2020_Unknown'
#
# df.loc[(df['year'] == 2021) & (df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_YEAR_SUBDIVISION'] = '2021_Lake_Geneva'
# df.loc[(df['year'] == 2021) & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2021_Eastern_Switzerland'
# df.loc[(df['year'] == 2021) & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_YEAR_SUBDIVISION'] = '2021_Espace_Mittelland'
# df.loc[(df['year'] == 2021) & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2021_Central_Switzerland'
# df.loc[(df['year'] == 2021) & (df[
#                                    'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_YEAR_SUBDIVISION'] = '2021_Northwestern_Switzerland'
# df.loc[(df['year'] == 2021) & (df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_YEAR_SUBDIVISION'] = '2021_Zurich'
# df.loc[(df['year'] == 2021) & (df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_YEAR_SUBDIVISION'] = '2021_Tessin'
# df.loc[(df['year'] == 2021) & (df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_YEAR_SUBDIVISION'] = '2021_Unknown'
# # endregion
#
# # region year continent
# df.loc[(df['year'] == 2019) & (df['continent'] == 'Europe'), 'NEW_YEAR_CONTINENT'] = '2019_Europe'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'Europe'), 'NEW_YEAR_CONTINENT'] = '2020_Europe'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'Europe'), 'NEW_YEAR_CONTINENT'] = '2021_Europe'
#
# df.loc[(df['year'] == 2019) & (df['continent'] == 'Asia'), 'NEW_YEAR_CONTINENT'] = '2019_Asia'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'Asia'), 'NEW_YEAR_CONTINENT'] = '2020_Asia'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'Asia'), 'NEW_YEAR_CONTINENT'] = '2021_Asia'
#
# df.loc[(df['year'] == 2019) & (df['continent'] == 'America'), 'NEW_YEAR_CONTINENT'] = '2019_America'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'America'), 'NEW_YEAR_CONTINENT'] = '2020_America'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'America'), 'NEW_YEAR_CONTINENT'] = '2021_America'
#
# df.loc[(df['year'] == 2019) & (df['continent'] == 'Africa'), 'NEW_YEAR_CONTINENT'] = '2019_Africa'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'Africa'), 'NEW_YEAR_CONTINENT'] = '2020_Africa'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'Africa'), 'NEW_YEAR_CONTINENT'] = '2021_Africa'
#
# df.loc[(df['year'] == 2019) & (df['continent'] == 'Oceania'), 'NEW_YEAR_CONTINENT'] = '2019_Oceania'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'Oceania'), 'NEW_YEAR_CONTINENT'] = '2020_Oceania'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'Oceania'), 'NEW_YEAR_CONTINENT'] = '2021_Oceania'
#
# df.loc[(df['year'] == 2019) & (df['continent'] == 'unknown_origin'), 'NEW_YEAR_CONTINENT'] = '2019_unknown_origin'
# df.loc[(df['year'] == 2020) & (df['continent'] == 'unknown_origin'), 'NEW_YEAR_CONTINENT'] = '2020_unknown_origin'
# df.loc[(df['year'] == 2021) & (df['continent'] == 'unknown_origin'), 'NEW_YEAR_CONTINENT'] = '2021_unknown_origin'
# # endregion
#
# # region year europa_third_country
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EU_17'), 'NEW_YEAR_EUROPA'] = '2019_EU_17'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EU_17'), 'NEW_YEAR_EUROPA'] = '2020_EU_17'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EU_17'), 'NEW_YEAR_EUROPA'] = '2021_EU_17'
#
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EU'), 'NEW_YEAR_EUROPA'] = '2019_EU'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EU'), 'NEW_YEAR_EUROPA'] = '2020_EU'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EU'), 'NEW_YEAR_EUROPA'] = '2021_EU'
#
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EU_8'), 'NEW_YEAR_EUROPA'] = '2019_EU_8'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EU_8'), 'NEW_YEAR_EUROPA'] = '2020_EU_8'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EU_8'), 'NEW_YEAR_EUROPA'] = '2021_EU_8'
#
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EU_2'), 'NEW_YEAR_EUROPA'] = '2019_EU_2'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EU_2'), 'NEW_YEAR_EUROPA'] = '2020_EU_2'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EU_2'), 'NEW_YEAR_EUROPA'] = '2021_EU_2'
#
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EU_1'), 'NEW_YEAR_EUROPA'] = '2019_EU_1'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EU_1'), 'NEW_YEAR_EUROPA'] = '2020_EU_1'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EU_1'), 'NEW_YEAR_EUROPA'] = '2021_EU_1'
#
# df.loc[(df['year'] == 2019) & (df['europa_third_country'] == 'EFTA'), 'NEW_YEAR_EUROPA'] = '2019_EFTA'
# df.loc[(df['year'] == 2020) & (df['europa_third_country'] == 'EFTA'), 'NEW_YEAR_EUROPA'] = '2020_EFTA'
# df.loc[(df['year'] == 2021) & (df['europa_third_country'] == 'EFTA'), 'NEW_YEAR_EUROPA'] = '2021_EFTA'
#
# df.loc[(df['year'] == 2019) & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_YEAR_EUROPA'] = '2019_third_countries'
# df.loc[(df['year'] == 2020) & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_YEAR_EUROPA'] = '2020_third_countries'
# df.loc[(df['year'] == 2021) & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_YEAR_EUROPA'] = '2021_third_countries'
# # endregion
#
# # endregion
#
# # region VARIABLE PRODUCTION WITH PERMANENT AND TEMPORAL POPULATION
# # region permanent_temporal_population and foreign_group
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['foreign_group'] == 'resident_B'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_resident_B'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'short_resident_4_12_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_short_resident_4_12_month'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'short_resident_L_12_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_short_resident_L_12_month'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'short_resident_4_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_short_resident_4_month'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'service_providers_4_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_service_providers_4_month'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'musician_artist_8_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_musician_artist_8_month'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'foreign_group'] == 'settled_foreigners_C'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'permanent_settled_foreigners_C'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['foreign_group'] == 'resident_B'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_resident_B'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'short_resident_4_12_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_short_resident_4_12_month'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'short_resident_L_12_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_short_resident_L_12_month'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'short_resident_4_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_short_resident_4_month'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'service_providers_4_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_service_providers_4_month'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'musician_artist_8_month'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_musician_artist_8_month'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'foreign_group'] == 'settled_foreigners_C'), 'NEW_PERMANENT_TEMPORAL_FOREIGN_GROUP'] = 'temporal_settled_foreigners_C'
# # endregion
#
# # region permanent_temporal_population and sector
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['sector'] == 'services_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'permanent_services_sector'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'sector'] == 'industry_craft_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'permanent_industry_craft_sector'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'permanent_agriculture_sector'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['sector'] == 'unknown_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'permanent_unknown_sector'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['sector'] == 'services_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'temporal_services_sector'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'sector'] == 'industry_craft_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'temporal_industry_craft_sector'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'temporal_agriculture_sector'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['sector'] == 'unknown_sector'), 'NEW_PERMANENT_TEMPORAL_SECTOR'] = 'temporal_unknown_sector'
# # endregion
#
# # region permanent_temporal_population and sex
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['sex'] == 'male'), 'NEW_PERMANENT_TEMPORAL_SEX'] = 'permanent_male'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['sex'] == 'female'), 'NEW_PERMANENT_TEMPORAL_SEX'] = 'permanent_female'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['sex'] == 'male'), 'NEW_PERMANENT_TEMPORAL_SEX'] = 'temporal_male'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['sex'] == 'female'), 'NEW_PERMANENT_TEMPORAL_SEX'] = 'temporal_female'
# # endregion
#
# # region permanent_temporal_population and coontinent
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'Europe'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_Europe'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'Asia'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_Asia'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'America'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_America'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'Africa'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_Africa'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'Oceania'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_Oceania'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['continent'] == 'unknown_origin'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'permanent_unknown_origin'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'Europe'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_Europe'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'Asia'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_Asia'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'America'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_America'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'Africa'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_Africa'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'Oceania'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_Oceania'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['continent'] == 'unknown_origin'), 'NEW_PERMANENT_TEMPORAL_CONTINENT'] = 'temporal_unknown_origin'
# # endregion
#
# # region permanent_temporal_population and europa_third_country
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'europa_third_country'] == 'third_countries'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_third_countries'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EU_17'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EU'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EU'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EU_8'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EU_2'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EU_1'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'permanent_EFTA'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'europa_third_country'] == 'third_countries'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_third_countries'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EU_17'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EU'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EU'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EU_8'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EU_2'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EU_1'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_PERMANENT_TEMPORAL_CUNTRIES'] = 'temporal_EFTA'
# # endregion
# # endregion
#
# # region VARIABLE PRODUCTION WITH FOREIGN GROUP
# # region foreign group and sector
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'resident_B_services_sector'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'resident_B_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'resident_B_agriculture_sector'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'resident_B_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_12_month_services_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (df[
#                                                                    'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_12_month_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (df[
#                                                                    'sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_12_month_agriculture_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_12_month_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_L_12_month_services_sector'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (df[
#                                                                    'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_L_12_month_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (df[
#                                                                    'sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_L_12_month_agriculture_sector'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_L_12_month_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_month_services_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (df[
#                                                                 'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_month_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_month_agriculture_sector'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'short_resident_4_month_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'service_providers_4_month_services_sector'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (df[
#                                                                    'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'service_providers_4_month_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (df[
#                                                                    'sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'service_providers_4_month_agriculture_sector'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'service_providers_4_month_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'musician_artist_8_month_services_sector'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (df[
#                                                                  'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'musician_artist_8_month_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'musician_artist_8_month_agriculture_sector'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'musician_artist_8_month_unknown_sector'
#
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['sector'] == 'services_sector'), 'NEW_FOREIGN_SECTOR'] = 'settled_foreigners_C_services_sector'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (df[
#                                                               'sector'] == 'industry_craft_sector'), 'NEW_FOREIGN_SECTOR'] = 'settled_foreigners_C_industry_craft_sector'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['sector'] == 'agriculture_sector'), 'NEW_FOREIGN_SECTOR'] = 'settled_foreigners_C_agriculture_sector'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['sector'] == 'unknown_sector'), 'NEW_FOREIGN_SECTOR'] = 'settled_foreigners_C_unknown_sector'
# # endregion
#
# # region foreign group and sex
# df.loc[(df['sex'] == 'male') & (df['foreign_group'] == 'resident_B'), 'NEW_FOREIGN_SEX'] = 'male_resident_B'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'short_resident_4_12_month'), 'NEW_FOREIGN_SEX'] = 'male_short_resident_4_12_month'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'short_resident_L_12_month'), 'NEW_FOREIGN_SEX'] = 'male_short_resident_L_12_month'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'short_resident_4_month'), 'NEW_FOREIGN_SEX'] = 'male_short_resident_4_month'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'service_providers_4_month'), 'NEW_FOREIGN_SEX'] = 'male_service_providers_4_month'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'musician_artist_8_month'), 'NEW_FOREIGN_SEX'] = 'male_musician_artist_8_month'
# df.loc[(df['sex'] == 'male') & (
#             df['foreign_group'] == 'settled_foreigners_C'), 'NEW_FOREIGN_SEX'] = 'male_settled_foreigners_C'
#
# df.loc[(df['sex'] == 'female') & (df['foreign_group'] == 'resident_B'), 'NEW_FOREIGN_SEX'] = 'female_resident_B'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'short_resident_4_12_month'), 'NEW_FOREIGN_SEX'] = 'female_short_resident_4_12_month'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'short_resident_L_12_month'), 'NEW_FOREIGN_SEX'] = 'female_short_resident_L_12_month'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'short_resident_4_month'), 'NEW_FOREIGN_SEX'] = 'female_short_resident_4_month'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'service_providers_4_month'), 'NEW_FOREIGN_SEX'] = 'female_service_providers_4_month'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'musician_artist_8_month'), 'NEW_FOREIGN_SEX'] = 'female_musician_artist_8_month'
# df.loc[(df['sex'] == 'female') & (
#             df['foreign_group'] == 'settled_foreigners_C'), 'NEW_FOREIGN_SEX'] = 'female_settled_foreigners_C'
# # endregion
#
# # region foreign group and continent
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_Europe'
# df.loc[(df['foreign_group'] == 'resident_B') & (df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_Asia'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_America'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_Africa'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_Oceania'
# df.loc[(df['foreign_group'] == 'resident_B') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'resident_B_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_Europe'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_Asia'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_America'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_Africa'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_Oceania'
# df.loc[(df['foreign_group'] == 'short_resident_4_12_month') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_12_month_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_Europe'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_Asia'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_America'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_Africa'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_Oceania'
# df.loc[(df['foreign_group'] == 'short_resident_L_12_month') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_L_12_month_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_Europe'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_Asia'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_America'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_Africa'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_Oceania'
# df.loc[(df['foreign_group'] == 'short_resident_4_month') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'short_resident_4_month_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_Europe'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_Asia'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_America'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_Africa'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_Oceania'
# df.loc[(df['foreign_group'] == 'service_providers_4_month') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'service_providers_4_month_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_Europe'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_Asia'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_America'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_Africa'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_Oceania'
# df.loc[(df['foreign_group'] == 'musician_artist_8_month') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'musician_artist_8_month_unknown_origin'
#
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'Europe'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_Europe'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'Asia'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_Asia'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'America'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_America'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'Africa'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_Africa'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'Oceania'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_Oceania'
# df.loc[(df['foreign_group'] == 'settled_foreigners_C') & (
#             df['continent'] == 'unknown_origin'), 'NEW_FOREIGN_CONTINENT'] = 'settled_foreigners_C_unknown_origin'
# # endregion
# # endregion
#
# # region VARIABLE PRODUCTION WITH SECTOR
# # region sector and sex
# df.loc[(df['sex'] == 'male') & (df['sector'] == 'services_sector'), 'NEW_SECTOR_SEX'] = 'male_services_sector'
# df.loc[
#     (df['sex'] == 'male') & (df['sector'] == 'industry_craft_sector'), 'NEW_SECTOR_SEX'] = 'male_industry_craft_sector'
# df.loc[(df['sex'] == 'male') & (df['sector'] == 'agriculture_sector'), 'NEW_SECTOR_SEX'] = 'male_agriculture_sector'
# df.loc[(df['sex'] == 'male') & (df['sector'] == 'unknown_sector'), 'NEW_SECTOR_SEX'] = 'male_unknown_sector'
#
# df.loc[(df['sex'] == 'female') & (df['sector'] == 'services_sector'), 'NEW_SECTOR_SEX'] = 'female_services_sector'
# df.loc[(df['sex'] == 'female') & (
#             df['sector'] == 'industry_craft_sector'), 'NEW_SECTOR_SEX'] = 'female_industry_craft_sector'
# df.loc[(df['sex'] == 'female') & (df['sector'] == 'agriculture_sector'), 'NEW_SECTOR_SEX'] = 'female_agriculture_sector'
# df.loc[(df['sex'] == 'female') & (df['sector'] == 'unknown_sector'), 'NEW_SECTOR_SEX'] = 'female_unknown_sector'
# # endregion
#
# # region sector and continent
# df.loc[(df['sector'] == 'services_sector') & (
#             df['continent'] == 'Europe'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_Europe'
# df.loc[
#     (df['sector'] == 'services_sector') & (df['continent'] == 'Asia'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_Asia'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['continent'] == 'America'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_America'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['continent'] == 'Africa'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_Africa'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['continent'] == 'Oceania'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_Oceania'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['continent'] == 'unknown_origin'), 'NEW_SECTOR_CONTINENT'] = 'services_sector_unknown_origin'
#
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'Europe'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_Europe'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'Asia'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_Asia'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'America'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_America'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'Africa'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_Africa'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'Oceania'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_Oceania'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['continent'] == 'unknown_origin'), 'NEW_SECTOR_CONTINENT'] = 'industry_craft_sector_unknown_origin'
#
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'Europe'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_Europe'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'Asia'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_Asia'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'America'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_America'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'Africa'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_Africa'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'Oceania'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_Oceania'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['continent'] == 'unknown_origin'), 'NEW_SECTOR_CONTINENT'] = 'agriculture_sector_unknown_origin'
#
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['continent'] == 'Europe'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_Europe'
# df.loc[(df['sector'] == 'unknown_sector') & (df['continent'] == 'Asia'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_Asia'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['continent'] == 'America'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_America'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['continent'] == 'Africa'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_Africa'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['continent'] == 'Oceania'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_Oceania'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['continent'] == 'unknown_origin'), 'NEW_SECTOR_CONTINENT'] = 'unknown_sector_unknown_origin'
# # endregion
#
# # region sector and europa_third_country
# df.loc[(df['sector'] == 'services_sector') & (df[
#                                                   'europa_third_country'] == 'third_countries'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_third_countries'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EU_17'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EU'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EU'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EU_8'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EU_2'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EU_1'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_SECTOR_COUNTRIES'] = 'services_sector_EFTA'
#
# df.loc[(df['sector'] == 'industry_craft_sector') & (df[
#                                                         'europa_third_country'] == 'third_countries'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_third_countries'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EU_17'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EU'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EU'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EU_8'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EU_2'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EU_1'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_SECTOR_COUNTRIES'] = 'industry_craft_sector_EFTA'
#
# df.loc[(df['sector'] == 'agriculture_sector') & (df[
#                                                      'europa_third_country'] == 'third_countries'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_third_countries'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EU_17'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EU'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EU'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EU_8'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EU_2'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EU_1'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_SECTOR_COUNTRIES'] = 'agriculture_sector_EFTA'
#
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_third_countries'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EU_17'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EU_17'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EU'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EU'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EU_8'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EU_8'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EU_2'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EU_2'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EU_1'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EU_1'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['europa_third_country'] == 'EFTA'), 'NEW_SECTOR_COUNTRIES'] = 'unknown_sector_EFTA'
# # endregion
# # endregion
#
# # region VARIABLE PRODUCTION WITH SEX
# # region sex and continent
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'Europe'), 'NEW_SEX_CONTINENT'] = 'male_Europe'
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'Asia'), 'NEW_SEX_CONTINENT'] = 'male_Asia'
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'America'), 'NEW_SEX_CONTINENT'] = 'male_America'
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'Africa'), 'NEW_SEX_CONTINENT'] = 'male_Africa'
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'Oceania'), 'NEW_SEX_CONTINENT'] = 'male_Oceania'
# df.loc[(df['sex'] == 'male') & (df['continent'] == 'unknown_origin'), 'NEW_SEX_CONTINENT'] = 'male_unknown_origin'
#
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'Europe'), 'NEW_SEX_CONTINENT'] = 'female_Europe'
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'Asia'), 'NEW_SEX_CONTINENT'] = 'female_Asia'
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'America'), 'NEW_SEX_CONTINENT'] = 'female_America'
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'Africa'), 'NEW_SEX_CONTINENT'] = 'female_Africa'
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'Oceania'), 'NEW_SEX_CONTINENT'] = 'female_Oceania'
# df.loc[(df['sex'] == 'female') & (df['continent'] == 'unknown_origin'), 'NEW_SEX_CONTINENT'] = 'female_unknown_origin'
# # endregion
#
# # region sex and europa_third_country
# df.loc[(df['sex'] == 'male') & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_SEX_COUNTRY'] = 'male_third_countries'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EU_17'), 'NEW_SEX_COUNTRY'] = 'male_EU_17'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EU'), 'NEW_SEX_COUNTRY'] = 'male_EU'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EU_8'), 'NEW_SEX_COUNTRY'] = 'male_EU_8'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EU_2'), 'NEW_SEX_COUNTRY'] = 'male_EU_2'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EU_1'), 'NEW_SEX_COUNTRY'] = 'male_EU_1'
# df.loc[(df['sex'] == 'male') & (df['europa_third_country'] == 'EFTA'), 'NEW_SEX_COUNTRY'] = 'male_EFTA'
#
# df.loc[(df['sex'] == 'female') & (
#             df['europa_third_country'] == 'third_countries'), 'NEW_SEX_COUNTRY'] = 'female_third_countries'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EU_17'), 'NEW_SEX_COUNTRY'] = 'female_EU_17'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EU'), 'NEW_SEX_COUNTRY'] = 'female_EU'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EU_8'), 'NEW_SEX_COUNTRY'] = 'female_EU_8'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EU_2'), 'NEW_SEX_COUNTRY'] = 'female_EU_2'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EU_1'), 'NEW_SEX_COUNTRY'] = 'female_EU_1'
# df.loc[(df['sex'] == 'female') & (df['europa_third_country'] == 'EFTA'), 'NEW_SEX_COUNTRY'] = 'female_EFTA'
# # endregion
# # endregion
#
# # region VARIABLE PRODUCTION WITH SUBDIVISION
# # region sex and new subdivision
# df.loc[(df['sex'] == 'male') & (df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SEX_SUBDIVISION'] = 'male_Lake_Geneva'
# df.loc[(df['sex'] == 'male') & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'male_Eastern_Switzerland'
# df.loc[(df['sex'] == 'male') & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SEX_SUBDIVISION'] = 'male_Espace_Mittelland'
# df.loc[(df['sex'] == 'male') & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'male_Central_Switzerland'
# df.loc[(df['sex'] == 'male') & (df[
#                                     'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'male_Northwestern_Switzerland'
# df.loc[(df['sex'] == 'male') & (df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SEX_SUBDIVISION'] = 'male_Zurich'
# df.loc[(df['sex'] == 'male') & (df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SEX_SUBDIVISION'] = 'male_Tessin'
# df.loc[(df['sex'] == 'male') & (df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SEX_SUBDIVISION'] = 'male_Unknown'
#
# df.loc[(df['sex'] == 'female') & (df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SEX_SUBDIVISION'] = 'female_Lake_Geneva'
# df.loc[(df['sex'] == 'female') & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'female_Eastern_Switzerland'
# df.loc[(df['sex'] == 'female') & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SEX_SUBDIVISION'] = 'female_Espace_Mittelland'
# df.loc[(df['sex'] == 'female') & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'female_Central_Switzerland'
# df.loc[(df['sex'] == 'female') & (df[
#                                       'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SEX_SUBDIVISION'] = 'female_Northwestern_Switzerland'
# df.loc[(df['sex'] == 'female') & (df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SEX_SUBDIVISION'] = 'female_Zurich'
# df.loc[(df['sex'] == 'female') & (df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SEX_SUBDIVISION'] = 'female_Tessin'
# df.loc[(df['sex'] == 'female') & (df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SEX_SUBDIVISION'] = 'female_Unknown'
# # endregion
#
# # region permanent_temporal_population and new subdivision
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Lake_Geneva'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Eastern_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Espace_Mittelland'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Central_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (df[
#                                                                    'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Northwestern_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Zurich'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Tessin'
# df.loc[(df['permanent_temporal_population'] == 'permanent') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_PTP_SUBDIVISION'] = 'permanent_Unknown'
#
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Lake_Geneva'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Eastern_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Espace_Mittelland'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Central_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (df[
#                                                                   'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Northwestern_Switzerland'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Zurich'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Tessin'
# df.loc[(df['permanent_temporal_population'] == 'temporal') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_PTP_SUBDIVISION'] = 'temporal_Unknown'
# # endregion
#
# # region sector and new subdivision
# df.loc[(df['sector'] == 'services_sector') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Lake_Geneva'
# df.loc[(df['sector'] == 'services_sector') & (df[
#                                                   'NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Eastern_Switzerland'
# df.loc[(df['sector'] == 'services_sector') & (df[
#                                                   'NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Espace_Mittelland'
# df.loc[(df['sector'] == 'services_sector') & (df[
#                                                   'NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Central_Switzerland'
# df.loc[(df['sector'] == 'services_sector') & (df[
#                                                   'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Northwestern_Switzerland'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Zurich'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Tessin'
# df.loc[(df['sector'] == 'services_sector') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SECTOR_SUBDIVISION'] = 'services_sector_Unknown'
#
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Lake_Geneva'
# df.loc[(df['sector'] == 'industry_craft_sector') & (df[
#                                                         'NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Eastern_Switzerland'
# df.loc[(df['sector'] == 'industry_craft_sector') & (df[
#                                                         'NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Espace_Mittelland'
# df.loc[(df['sector'] == 'industry_craft_sector') & (df[
#                                                         'NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Central_Switzerland'
# df.loc[(df['sector'] == 'industry_craft_sector') & (df[
#                                                         'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Northwestern_Switzerland'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Zurich'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Tessin'
# df.loc[(df['sector'] == 'industry_craft_sector') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SECTOR_SUBDIVISION'] = 'industry_craft_sector_Unknown'
#
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Lake_Geneva'
# df.loc[(df['sector'] == 'agriculture_sector') & (df[
#                                                      'NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Eastern_Switzerland'
# df.loc[(df['sector'] == 'agriculture_sector') & (df[
#                                                      'NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Espace_Mittelland'
# df.loc[(df['sector'] == 'agriculture_sector') & (df[
#                                                      'NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Central_Switzerland'
# df.loc[(df['sector'] == 'agriculture_sector') & (df[
#                                                      'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Northwestern_Switzerland'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Zurich'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Tessin'
# df.loc[(df['sector'] == 'agriculture_sector') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SECTOR_SUBDIVISION'] = 'agriculture_sector_Unknown'
#
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['NEW_SUBDIVISION'] == 'Lake_Geneva'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Lake_Geneva'
# df.loc[(df['sector'] == 'unknown_sector') & (df[
#                                                  'NEW_SUBDIVISION'] == 'Eastern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Eastern_Switzerland'
# df.loc[(df['sector'] == 'unknown_sector') & (df[
#                                                  'NEW_SUBDIVISION'] == 'Espace_Mittelland'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Espace_Mittelland'
# df.loc[(df['sector'] == 'unknown_sector') & (df[
#                                                  'NEW_SUBDIVISION'] == 'Central_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Central_Switzerland'
# df.loc[(df['sector'] == 'unknown_sector') & (df[
#                                                  'NEW_SUBDIVISION'] == 'Northwestern_Switzerland'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Northwestern_Switzerland'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['NEW_SUBDIVISION'] == 'Zurich'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Zurich'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['NEW_SUBDIVISION'] == 'Tessin'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Tessin'
# df.loc[(df['sector'] == 'unknown_sector') & (
#             df['NEW_SUBDIVISION'] == 'Unknown'), 'NEW_SECTOR_SUBDIVISION'] = 'unknown_sector_Unknown'
# # endregion
# # endregion
# # endregion
# # region 31. SAVING THE NEW DATAFRAME
# df.to_csv('new_immigration.csv', index=False)
# # endregion
# # region 32. IMPORTING NEW DATAFRAME
# df = pd.read_csv('new_immigration.csv')
# df.columns = [col.lower() for col in df.columns]
# df.head()
# # endregion
# # region 33. DIVIDING NEW DATASET
# def divide_df(all_data):
#     # Returns divided dfs of training and test set
#     return all_data.loc[:22682], all_data.loc[22683:]
# # endregion
# # region 34. EDA AND PREPROCESSING FUNCTIONS
# class preprocessing():
#     def __init__(self, dataframe):
#         self.dataframe = dataframe
#
#     # EXPLORATORY DATA ANALYSIS
#
#     # check dataframe
#     def check_df(self, head=10):
#         print('#' * 20, 'Head', '#' * 20)
#         print(self.dataframe.head(head))
#         print('#' * 20, 'Shape', '#' * 20)
#         print(self.dataframe.shape)
#         print('#' * 20, 'Data Info', '#' * 20)
#         print(self.dataframe.info())
#         print('#' * 20, 'Data Types', '#' * 20)
#         print(self.dataframe.dtypes)
#         print('#' * 20, 'Missing Values', '#' * 20)
#         print(self.dataframe.isnull().sum().sort_values(ascending=False))
#         print('#' * 20, 'Descriptive Statistics', '#' * 20)
#         print(self.dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
#
#     # categorical, numerical and cardinal variables
#     def grab_col_names(self, cat_th=10, car_th=30):
#         # cat_cols, cat_but_car
#         cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
#         num_but_cat = [col for col in self.dataframe.columns if self.dataframe[col].nunique() < cat_th and
#                        self.dataframe[col].dtypes != "O"]
#         cat_but_car = [col for col in self.dataframe.columns if self.dataframe[col].nunique() > car_th and
#                        self.dataframe[col].dtypes == "O"]
#         cat_cols = cat_cols + num_but_cat
#         cat_cols = [col for col in cat_cols if col not in cat_but_car]
#
#         # num_cols
#         num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
#         num_cols = [col for col in num_cols if col not in num_but_cat]
#
#         print(f"Observations: {self.dataframe.shape[0]}")
#         print(f"Variables: {self.dataframe.shape[1]}")
#         print(f'cat_cols: {len(cat_cols)}')
#         print(f'num_cols: {len(num_cols)}')
#         print(f'cat_but_car: {len(cat_but_car)}')
#         print(f'num_but_cat: {len(num_but_cat)}')
#         return cat_cols, num_cols, cat_but_car
#
#     # summary of categorical variables
#     def cat_summary(self, col_name, plot=False):
#         print(f'\n{col_name.upper()} Summary:')
#         counts = self.dataframe[col_name].value_counts()
#         percentages = counts / len(self.dataframe) * 100
#         print(pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']))
#
#         if plot:
#             self.dataframe[col_name].value_counts().plot(kind='bar', rot=90, color='#40596D')
#             plt.xlabel(col_name.upper())
#             plt.ylabel('COUNT')
#             plt.show(block=True)
#
#     # summary of numerical variables
#     def num_summary(self, numerical_col, plot=False):
#         print(f'\n{numerical_col.upper()} Summary:')
#         print(self.dataframe[numerical_col].describe().T.round(2))
#
#         if plot:
#             fig = plt.figure(figsize=(10, 4))
#             plt.subplot(1, 2, 1)
#             sns.boxplot(data=self.dataframe, y=numerical_col, color='#40596D')
#             plt.xlabel(numerical_col.upper())
#             plt.subplot(1, 2, 2)
#             sns.histplot(data=self.dataframe, x=numerical_col, color='#40596D')
#             plt.xlabel(numerical_col.upper())
#             plt.show(block=True)
#
#     # Analysis of target variable with categorical variables
#     def target_summary_with_cat(self, target, categorical_col, plot=False):
#         print('\n', '#' * 10, categorical_col.upper(), '#' * 10)
#         target_mean = round(self.dataframe.groupby(categorical_col)[target].mean(), 2)
#         print(pd.DataFrame({'TARGET_MEAN': target_mean}))
#
#         if plot:
#             sns.barplot(x=self.dataframe[categorical_col], y=self.dataframe[target], color='#40596D')
#             plt.xlabel(categorical_col.upper())
#             plt.ylabel(target.upper())
#             plt.show(block=True)
#
#     # Analysis of target variable with numerical variables
#     def target_summary_with_num(self, target, numerical_col, plot=False):
#         target_mean = self.dataframe.groupby(target)[numerical_col].mean()
#         print(target_mean)
#
#         if plot:
#             sns.barplot(x=self.dataframe[target], y=self.dataframe[numerical_col], color='#40596D')
#             plt.ylabel(numerical_col.upper())
#             plt.xlabel(target.upper())
#             plt.show(block=True)
#
#     # high_correlated_cols' function to delete the variables with high correlation
#     def high_correlated_cols(self, corr_threshold=0.90, plot=False):
#         corr = self.dataframe.corr()
#         cor_matrix = corr.abs()
#         upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
#         drop_list = [col for col in upper_triangle_matrix.columns if
#                      any(upper_triangle_matrix[col] > corr_threshold)]
#
#         if plot:
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(corr, cmap='RdBu', annot=True, linewidths=0.5, linecolor='w')
#             plt.show(block=True)
#
#         return drop_list
#
#     # DATA PREPROCESSING
#
#     # finding outlier thresholds
#     def outlier_thresholds(self, col_name, lower_quantile=0.05, upper_quantile=0.95):
#         lower_quartile = self.dataframe[col_name].quantile(lower_quantile)
#         upper_quartile = self.dataframe[col_name].quantile(upper_quantile)
#         iqr = upper_quartile - lower_quartile
#         upper_limit = round(upper_quartile + 1.5 * iqr, 2)
#         lower_limit = round(lower_quartile - 1.5 * iqr, 2)
#         return lower_limit, upper_limit
#
#     # replace outliers with thresholds
#     def replace_with_thresholds(self, variable, lower_quantile=0.05, upper_quantile=0.95):
#         lower_quartile = self.dataframe[variable].quantile(lower_quantile)
#         upper_quartile = self.dataframe[variable].quantile(upper_quantile)
#         iqr = upper_quartile - lower_quartile
#         upper_limit = round(upper_quartile + 1.5 * iqr, 2)
#         lower_limit = round(lower_quartile - 1.5 * iqr, 2)
#         self.dataframe.loc[(self.dataframe[variable] < lower_limit), variable] = lower_limit
#         self.dataframe.loc[(self.dataframe[variable] > upper_limit), variable] = upper_limit
#
#     # is there any outliers?
#     def has_outliers(self, col_name, lower_quantile=0.05, upper_quantile=0.95):
#         lower_quartile = self.dataframe[col_name].quantile(lower_quantile)
#         upper_quartile = self.dataframe[col_name].quantile(upper_quantile)
#         iqr = upper_quartile - lower_quartile
#         upper_limit = round(upper_quartile + 1.5 * iqr, 2)
#         lower_limit = round(lower_quartile - 1.5 * iqr, 2)
#         return self.dataframe[
#             ((self.dataframe[col_name] < lower_limit) | (self.dataframe[col_name] > upper_limit))].any(
#             axis=None)
#
#     # printing outliers
#     def print_outliers(self, column_name, lower_quantile=0.05, upper_quantile=0.95, show_index=False):
#         lower_quartile = self.dataframe[column_name].quantile(lower_quantile)
#         upper_quartile = self.dataframe[column_name].quantile(upper_quantile)
#         iqr = upper_quartile - lower_quartile
#         upper_limit = round(upper_quartile + 1.5 * iqr, 2)
#         lower_limit = round(lower_quartile - 1.5 * iqr, 2)
#         if self.dataframe[((self.dataframe[column_name] < lower_limit) | (
#                 self.dataframe[column_name] > upper_limit))].shape[0] > 10:
#             print(self.dataframe[((self.dataframe[column_name] < lower_limit) | (
#                     self.dataframe[column_name] > upper_limit))].head())
#         else:
#             print(self.dataframe[((self.dataframe[column_name] < lower_limit) | (
#                     self.dataframe[column_name] > upper_limit))])
#
#         if show_index:
#             outlier_index = self.dataframe[((self.dataframe[column_name] < lower_limit) | (
#                     self.dataframe[column_name] > upper_limit))].index
#             return outlier_index
#
#     # removing outliers
#     def remove_outliers(self, column_name, lower_quantile=0.05, upper_quantile=0.95):
#         lower_quartile = self.dataframe[column_name].quantile(lower_quantile)
#         upper_quartile = self.dataframe[column_name].quantile(upper_quantile)
#         iqr = upper_quartile - lower_quartile
#         upper_limit = round(upper_quartile + 1.5 * iqr, 2)
#         lower_limit = round(lower_quartile - 1.5 * iqr, 2)
#         df_without_outliers = self.dataframe[
#             ~((self.dataframe[column_name] < lower_limit) | (self.dataframe[column_name] > upper_limit))]
#         return df_without_outliers
#
#     # missing values
#     def missing_values_table(self, return_cols=False):
#         missing_cols = self.dataframe.columns[self.dataframe.isnull().any()]
#         missing_count = self.dataframe[missing_cols].isnull().sum().sort_values(ascending=False)
#         missing_ratio = (missing_count / self.dataframe.shape[0]) * 100
#         missing_data = pd.concat([missing_count, missing_ratio], axis=1,
#                                  keys=['Missing Count', 'Missing Ratio (%)'])
#         print(missing_data, end="\n")
#         if return_cols:
#             return missing_cols
#
#     # missing values vs target
#     def missing_vs_target(self, target, missing_cols):
#         temp_df = self.dataframe.copy()
#         for col in missing_cols:
#             temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
#         na_flags = temp_df.columns[temp_df.columns.str.contains("_NA_FLAG")]
#         for col in na_flags:
#             print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
#                                 "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
#
#     # label encoding
#     def label_encoder(self, binary_col):
#         labelencoder = LabelEncoder()
#         self.dataframe[binary_col] = labelencoder.fit_transform(self.dataframe[binary_col])
#         return self.dataframe
#
#     # one-hot encoding
#     def one_hot_encoder(self, categorical_cols, drop_first=False):
#         self.dataframe = pd.get_dummies(self.dataframe, columns=categorical_cols, drop_first=drop_first)
#         return self.dataframe
#
#     # rare analyzer
#     def rare_analyzer(self, target, categorical_columns, threshold=0.05):
#         for col in categorical_columns:
#             counts = self.dataframe[col].value_counts(normalize=True)
#             rare_labels = counts[counts < threshold].index
#             print(f'{col} : {len(rare_labels)}')
#             print(pd.DataFrame({'COUNT': self.dataframe[col].value_counts(),
#                                 'RATIO': self.dataframe[col].value_counts(normalize=True),
#                                 'TARGET_MEAN': self.dataframe.groupby(col)[target].mean()}), end='\n\n\n')
#
#     # encoding rare colums
#     def rare_encoder(self, rare_perc=0.05):
#         temp_df = self.dataframe.copy()
#         rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O' and (
#                 temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
#         for col in rare_columns:
#             counts = temp_df[col].value_counts() / len(temp_df)
#             rare_labels = counts[counts < rare_perc].index
#             temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
#         return temp_df
#
#
# # endregion
# # region 35. EDA & PREPROCESSING FOR NEW DATAFRAME
# prep = preprocessing(df)
#
# prep.check_df()
#
# cat_cols, num_cols, cat_but_car = prep.grab_col_names()
#
# for col in cat_cols:
#     prep.cat_summary(col, False)
#
# for col in cat_but_car:
#     prep.cat_summary(col, False)
#
#
# class Rare_analyzing():
#     def __init__(self, dataframe):
#         self.dataframe = dataframe
#         # rare analyzer
#
#     def rare_analyzer(self, target, categorical_columns, threshold=0.05):
#         for col in categorical_columns:
#             counts = self.dataframe[col].value_counts(normalize=True)
#             rare_labels = counts[counts < threshold].index
#             print(f'{col} : {len(rare_labels)}')
#             print(pd.DataFrame({'COUNT': self.dataframe[col].value_counts(),
#                                 'RATIO': self.dataframe[col].value_counts(normalize=True),
#                                 'TARGET_MEAN': self.dataframe.groupby(col)[target].mean()}), end='\n\n\n')
#
#     # encoding rare colums
#     def rare_encoder(self, rare_perc=0.05):
#         temp_df = self.dataframe.copy()
#         rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O']
#         for col in rare_columns:
#             counts = temp_df[col].value_counts(normalize=True)
#             rare_labels = counts[counts < rare_perc].index
#             temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare_columns', temp_df[col])
#
#         return temp_df
#
#
# rare = Rare_analyzing(df)
#
# rare.rare_analyzer('n_people', cat_cols)
#
# df_new = rare.rare_encoder(0.025)
#
# df_new.to_csv('rare_encoded_immigration.csv', index=False)
#
# df = pd.read_csv('rare_encoded_immigration.csv')
#
# prep1 = preprocessing(df)
#
# prep1.check_df()
#
# cat_cols, num_cols, cat_but_car = prep1.grab_col_names()
#
# for col in cat_cols:
#     prep1.cat_summary(col, False)
#
# for col in cat_but_car:
#     prep1.cat_summary(col, False)
# # endregion
# # region 36. LABEL ENCODING
# binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]
# for col in binary_cols:
#     prep1.label_encoder(col)
# # endregion
# # region 37. ONEHOT ENCODING
# ohe_cols = [col for col in df.columns if 15 >= df[col].nunique() > 2]
# df = prep1.one_hot_encoder(ohe_cols, True)
# df.drop(['working_canton', 'new_year_subdivision'], axis=1, inplace=True)
# # endregion
# # region 38. MODELING USING FEATURE ENGINEERED DATASET-2
# def base_models_kfold2(X, y, n_splits=10, random_state=12345, save=False):
#     print('Base models with cross-validation...')
#
#     all_models = []
#
#     models = [("XGBoost", XGBRegressor()),
#               ("CatBoost", CatBoostRegressor(verbose=False))]
#
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#
#     for name, model in models:
#         rmse_train_scores = []
#         rmse_val_scores = []
#         mae_train_scores = []
#         mae_val_scores = []
#         r2_train_scores = []
#         r2_val_scores = []
#
#         for train_index, test_index in kf.split(X):
#             X_train, X_val = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_val = y.iloc[train_index], y.iloc[test_index]
#
#             model.fit(X_train, y_train)
#             y_pred_val = model.predict(X_val)
#             y_pred_train = model.predict(X_train)
#             rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
#             rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
#             mae_val = mean_absolute_error(y_val, y_pred_val)
#             mae_train = mean_absolute_error(y_train, y_pred_train)
#             r2_val = r2_score(y_val, y_pred_val)
#             r2_train = r2_score(y_train, y_pred_train)
#
#             rmse_train_scores.append(rmse_train)
#             rmse_val_scores.append(rmse_val)
#             mae_train_scores.append(mae_train)
#             mae_val_scores.append(mae_val)
#             r2_train_scores.append(r2_train)
#             r2_val_scores.append(r2_val)
#
#         values = dict(MODEL=name,
#                       RMSE_TRAIN=np.mean(rmse_train_scores),
#                       RMSE_VAL=np.mean(rmse_val_scores),
#                       MAE_TRAIN=np.mean(mae_train_scores),
#                       MAE_VAL=np.mean(mae_val_scores),
#                       R2_TRAIN=np.mean(r2_train_scores),
#                       R2_VAL=np.mean(r2_val_scores))
#
#         all_models.append(values)
#
#     sort_method = True
#     all_models_df = pd.DataFrame(all_models)
#     all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
#
#     # Set up the subplots
#     fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(8, 9))
#     fig.suptitle('Performance Metrics')
#
#     # Plot the bar charts
#     sns.barplot(x='RMSE_TRAIN', y='MODEL', data=all_models_df, ax=axs[0, 0])
#     sns.barplot(x='RMSE_VAL', y='MODEL', data=all_models_df, ax=axs[0, 1])
#     sns.barplot(x='MAE_TRAIN', y='MODEL', data=all_models_df, ax=axs[1, 0])
#     sns.barplot(x='MAE_VAL', y='MODEL', data=all_models_df, ax=axs[1, 1])
#     sns.barplot(x='R2_TRAIN', y='MODEL', data=all_models_df, ax=axs[2, 0])
#     sns.barplot(x='R2_VAL', y='MODEL', data=all_models_df, ax=axs[2, 1])
#
#     # Set the subplot titles
#     axs[0, 0].set_title('RMSE_TRAIN')
#     axs[0, 1].set_title('RMSE_VAL')
#     axs[1, 0].set_title('MAE_TRAIN')
#     axs[1, 1].set_title('MAE_VAL')
#     axs[2, 0].set_title('R2_TRAIN')
#     axs[2, 1].set_title('R2_VAL')
#
#     # Adjust the layout
#     plt.tight_layout()
#     plt.show(block=True)
#
#     if save:
#         plt.savefig('image/11models_with_featured_engineered.png')
#
#     print(all_models_df)
#
# train, test = divide_df(df)
# y = train['n_people']
# X = train.drop('n_people', axis=1)
# base_models_kfold2(X, y, save=True)
#
# # Base models with cross-validation...
# #       MODEL  RMSE_TRAIN  RMSE_VAL  MAE_TRAIN  MAE_VAL  R2_TRAIN  R2_VAL
# # 1  CatBoost      29.114    35.521      9.470   10.859     0.703   0.495
# # 0   XGBoost      28.777    36.501      9.108   11.037     0.710   0.469
# # endregion