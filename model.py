import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import pickle
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv("mobile_prices_2023.csv")

# Data preprocessing
df["Price in INR"] = df["Price in INR"].str.replace("₹", "").str.replace(",", "").astype("int")
df["Number of Ratings"] = df["Number of Ratings"].str.replace(",", "").astype("int")
df.rename(columns={'Rating ?/5': 'Rating 0/5'}, inplace=True)
df["Battery"] = df["Battery"].str.replace("mAh", "")
df.fillna({'Battery': 4000}, inplace=True)
replacement_dict = {'A15 ': 4000, 'A13 ': 4000, 'A14 ': 4000, 'A16 ': 4000, 'A12 ': 4000, 'A9 ': 4000,
                    'Apple ': 4000, '1 ': 4000, '0 ': 4000, 'MediaTek ': 4000, 'Brand ': 4000, 'Unisoc ': 4000,
                    '2 ': 4000}

# Replace values in the "Battery" column using replace function
df['Battery'] = df['Battery'].replace(replacement_dict)

# Convert the column to integer type
df['Battery'] = pd.to_numeric(df['Battery'], errors='coerce')
# Filling missing values in 'Battery' column with 4000
df["Battery"] = df["Battery"].astype('int')
df.fillna({'Front Camera': 'No Value Available',
           'ROM/Storage': 'No Value Available',
           'Processor': 'No Value Available',
           'Back/Rare Camera': 'No Value Available'}, inplace=True)
df['BRAND'] = df['Phone Name'].str.split().str[0]
replacement_dict = {'No': '0', '0': '0', 'Expandable': '0', 'NA': '0'}

# Replace values in the "ROM(GB)" column using replace function
df['ROM/Storage'] = df['ROM/Storage'].replace(replacement_dict)

# Convert the column to float type
df['ROM/Storage'] = pd.to_numeric(df['ROM/Storage'], errors='coerce')
df['ROM/Storage'] = df['ROM/Storage'].astype(str)
df['ROM(GB)'] = df['ROM/Storage'].str.split().str[0].astype(float)
df['RAM(GB)'] = df['RAM'].str.split().str[0]
rom_list = ['No', '0', 'Expandable', 'NA']
df['ROM(GB)'] = df['ROM(GB)'].apply(lambda x: 0 if x in rom_list else x)
ram_list = ['NA', 'Expandable', '0.53', '3.81', 'cm', '0.046875', '1.5']
df['RAM(GB)'] = df['RAM(GB)'].apply(lambda x: 0 if x in ram_list else x)
df['RAM(GB)'] = df['RAM(GB)'].replace({'0.53': '1', '3.81': '4', '0.046875': '0', '1.5': '2'}).astype(int)
df['Main_Back_Camera'] = df['Back/Rare Camera'].str.split().str[0].str.replace("MP", "").str.replace("Mp", "").replace('No', 0).astype(float)
df['Main_Front_Camera'] = df['Front Camera'].str.split().str[0].str.replace("MP", "").str.replace("Mp", "").replace('No', 0).astype(float)
BackCamera_list = []

# Split values in the 'Back/Rare Camera' column by '+'
for camera in df['Back/Rare Camera']:
    s = camera.split("+")
    BackCamera_list.append(s)

# Calculate the length of each sublist in 'BackCamera_list' and store the lengths in 'score'
score = [len(sublist) for sublist in BackCamera_list]
df['Back_camera_Score'] = score
frontCamera_list = []

# Split values in the 'Front Camera' column by '+'
for camera in df['Front Camera']:
    s = camera.split("+")
    frontCamera_list.append(s)
score1 = [len(sublist) for sublist in frontCamera_list]
df['front_camera_Score']=score1
df['BRAND'] = df['BRAND'].replace({"APPLE": "Apple", "apple": "Apple", "Nexus": "Google",
                                   "realme": "Realme", "vivo": "Vivo", "SAMSUNG": "Samsung",
                                   "10A": "Samsung", "Mi": "Xiaomi", "MOTOROLA": "Motorola","Redmi": "Xiaomi","REDMI": "Xiaomi",
                                   "Moto": "Motorola", "A10E": "Samsung", "a": "Samsung", "�9A": "Samsung", "10A": "Samsung"})

# Encoding categorical variable 'BRAND'
le = LabelEncoder()
df['Encoding_Brand'] = le.fit_transform(df['BRAND'])

# Prepare data for modeling
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
train_data_model = train_data[['Encoding_Brand', 'Number of Ratings','Battery', 'Rating 0/5', 'ROM(GB)',
                               'RAM(GB)', 'Back_camera_Score', 'front_camera_Score', 'Main_Back_Camera',
                               'Main_Front_Camera', 'Price in INR']]

train_data_model_X = train_data_model.iloc[:, :-1]
train_data_model_Y = train_data_model.iloc[:, -1]


# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(train_data_model_X, train_data_model_Y, test_size=0.3, random_state=0)

# Hyperparameter tuning for Random Forest Regressor
n_estimators = [100, 200, 300, 400, 500]
max_features = ['auto', 'sqrt']
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': [None, 50, 60, 70],
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf_random_search_CV = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                         n_iter=10, cv=10, random_state=0, n_jobs=-1)
rf_random_search_CV.fit(x_train, y_train)

# Model evaluation
y_pred_rf_train_Randomsearch = rf_random_search_CV.predict(x_train)
y_pred_rf_test_Randomsearch = rf_random_search_CV.predict(x_test)

print('Accuracy for training set: ', r2_score(y_train, y_pred_rf_train_Randomsearch) * 100, '%')
print('Accuracy for testing set: ', r2_score(y_test, y_pred_rf_test_Randomsearch) * 100, '%')

filename='rf_for_Mobie_Price_prediction'
pickle.dump(rf_random_search_CV,open(filename,'wb'))
loaded_model=pickle.load(open('rf_for_Mobie_Price_prediction','rb'))


