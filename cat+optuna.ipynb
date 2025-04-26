import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
import torch # Untuk pengecekan CUDA

# Load data dan preprocessing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['id']

train.drop(columns=['unique_session_id'], inplace=True)
test.drop(columns=['unique_session_id'], inplace=True)

categorical_cols = train.select_dtypes(include='object').columns.tolist()
if 'will_buy_on_return_visit' in categorical_cols:
    categorical_cols.remove('will_buy_on_return_visit')
numerical_cols = train.select_dtypes(exclude='object').columns.tolist()
if 'will_buy_on_return_visit' in numerical_cols:
    numerical_cols.remove('will_buy_on_return_visit')

def cap_outliers_iqr(df, columns, multiplier=1.5):
    df_capped = df.copy()
    for col in columns:
        if col in df_capped.columns:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_capped[col] = np.where(df_capped[col] < lower_bound, lower_bound, df_capped[col])
            df_capped[col] = np.where(df_capped[col] > upper_bound, upper_bound, df_capped[col])
    return df_capped

train_capped = cap_outliers_iqr(train, numerical_cols)

for df in [train_capped, test]:
    df[categorical_cols] = df[categorical_cols].fillna('missing')
    df[df.select_dtypes(exclude='object').columns] = df.select_dtypes(exclude='object').fillna(-1)

X = train_capped.drop(columns=['id', 'will_buy_on_return_visit'])
y = train_capped['will_buy_on_return_visit']
X_test = test.drop(columns=['id'])

# Label Encoding untuk target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Cek ketersediaan CUDA
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print("CUDA tersedia, menggunakan GPU untuk CatBoost.")
    task_type = 'GPU'
else:
    print("CUDA tidak tersedia, menggunakan CPU untuk CatBoost.")
    task_type = 'CPU'

# Fungsi objektif untuk Optuna
def objective(trial):
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'random_state': 42,
        'verbose': 0,
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50),
        'task_type': task_type # Gunakan GPU jika tersedia
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    preds = model.predict(val_pool)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

# Melakukan optimasi dengan Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params_optuna = study.best_params
print(f"Parameter CatBoost terbaik dari Optuna: {best_params_optuna}")

# Melatih model final dengan parameter terbaik
cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]
full_train_pool = Pool(X, y, cat_features=cat_features_indices)
final_model_catboost = CatBoostClassifier(
    **best_params_optuna,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_state=42,
    verbose=50,
    task_type=task_type # Gunakan GPU jika tersedia
)
final_model_catboost.fit(full_train_pool)

# Prediksi pada test set
test_pool = Pool(X_test, cat_features=cat_features_indices)
probabilities_catboost = final_model_catboost.predict_proba(test_pool)[:, 1]
predictions_catboost_binary = (probabilities_catboost > 0.5).astype(int)

# Buat file submission
submission_catboost_optuna_gpu = pd.DataFrame({
    'id': test_ids,
    'will_buy_on_return_visit': predictions_catboost_binary
})
submission_catboost_optuna_gpu.to_csv('submission_catboost_optuna_gpu.csv', index=False)

print("\nFile submission_catboost_optuna_gpu.csv berhasil dibuat.")
