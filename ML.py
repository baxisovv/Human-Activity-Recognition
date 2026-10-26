import pandas as pd
import numpy as np
import time
import warnings
from scipy import stats, signal
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings('ignore')

CSV_FILE_PATH = 'Time_domain_subsamples.csv'

CLASS_NAMES = [
    "Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", 
    "Lay", "Lay-stand", "Pick", "Jump", "Push-up", 
    "Sit-up", "Walk", "Walk-backward", "Circle-walk", 
    "Run", "Stair-up", "Stair-down", "Table-tennis"
]

def calculate_entropy(signal):
    """Sinyal entropisi hesaplar"""
    counts, _ = np.histogram(signal, bins=10, density=True)
    counts = counts[counts > 0]
    return stats.entropy(counts)

def get_fft_features(sig, fs=100):
    """Frekans alanı özelliklerini hesaplar (FFT)"""
    f_values = np.fft.rfft(sig)
    f_mag = np.abs(f_values)
    f_freqs = np.fft.rfftfreq(len(sig), 1/fs)

    f_mag_no_dc = f_mag[1:]
    f_freqs_no_dc = f_freqs[1:]
    
    if len(f_mag_no_dc) == 0: return [0, 0, 0, 0, 0] # Güvenlik

    # MaxInds: En baskın frekans (Dominant Frequency)
    max_idx = np.argmax(f_mag_no_dc)
    dominant_freq = f_freqs_no_dc[max_idx]
    
    # meanFreq: Ağırlıklı ortalama frekans
    mean_freq = np.sum(f_freqs_no_dc * f_mag_no_dc) / (np.sum(f_mag_no_dc) + 1e-9)
    
    # Frekans Skewness ve Kurtosis
    freq_skew = stats.skew(f_mag_no_dc)
    freq_kurt = stats.kurtosis(f_mag_no_dc)
    
    # Frekans Enerjisi (Spectral Energy)
    freq_energy = np.sum(f_mag_no_dc**2) / len(f_mag_no_dc)
    
    return [dominant_freq, mean_freq, freq_skew, freq_kurt, freq_energy]

def get_ar_coeffs(sig, order=4):
    try:
        N = len(sig)
        if N <= order: return [0]*order
        
        correlation = np.correlate(sig, sig, mode='full')
        correlation = correlation[len(correlation)//2:]
        if correlation[0] == 0: return [0]*order
        
        ar_coeffs = correlation[1:order+1] / correlation[0]
        if len(ar_coeffs) < order:
            ar_coeffs = np.pad(ar_coeffs, (0, order-len(ar_coeffs)))
        return list(ar_coeffs)
    except:
        return [0] * order

def extract_features(data):
    print("⚙️  Advanced Feature Extraction started...")
    features_list = []
    
    data_reshaped = data.reshape(data.shape[0], 6, 300)

    start_feat_time = time.time()
    
    for i, sample in enumerate(data_reshaped):
        sample_features = []
        
        for axis_idx, axis in enumerate(sample):
            sample_features.extend([
                np.mean(axis),              # Ortalama
                np.max(axis),               # Maksimum
                np.min(axis),               # Minimum
                np.median(axis),            # Medyan
                np.std(axis),               # Standart Sapma
                stats.skew(axis),           # Çarpıklık
                stats.kurtosis(axis),       # Basıklık
                np.sum(np.abs(axis)),       # SMA
                stats.iqr(axis),            # IQR
                np.sum(axis**2),            # Enerji
                calculate_entropy(axis)     # Entropi
            ])
            
            sample_features.extend(get_ar_coeffs(axis, order=4))
            
            sample_features.extend(get_fft_features(axis))

        # --- 2. KANALLAR ARASI ÖZELLİKLER (Korelasyon & Açı) ---
        # Acc Eksenleri: 0, 1, 2 | Gyro Eksenleri: 3, 4, 5
        
        # Korelasyon (Acc X-Y, X-Z, Y-Z)
        acc_corr_xy = np.corrcoef(sample[0], sample[1])[0,1]
        acc_corr_xz = np.corrcoef(sample[0], sample[2])[0,1]
        acc_corr_yz = np.corrcoef(sample[1], sample[2])[0,1]
        
        # Korelasyon (Gyro X-Y, X-Z, Y-Z)
        gyro_corr_xy = np.corrcoef(sample[3], sample[4])[0,1]
        gyro_corr_xz = np.corrcoef(sample[3], sample[5])[0,1]
        gyro_corr_yz = np.corrcoef(sample[4], sample[5])[0,1]
        
        sample_features.extend([
            acc_corr_xy, acc_corr_xz, acc_corr_yz,
            gyro_corr_xy, gyro_corr_xz, gyro_corr_yz
        ])
        
        acc_mean_vec = np.array([np.mean(sample[0]), np.mean(sample[1]), np.mean(sample[2])])
        norm_acc = np.linalg.norm(acc_mean_vec) + 1e-9
        
        angle_x = np.arccos(acc_mean_vec[0] / norm_acc)
        angle_y = np.arccos(acc_mean_vec[1] / norm_acc)
        angle_z = np.arccos(acc_mean_vec[2] / norm_acc)
        
        sample_features.extend([angle_x, angle_y, angle_z])

        # NaN kontrolü 
        sample_features = [0 if np.isnan(x) else x for x in sample_features]
        
        features_list.append(sample_features)
        
        # İlerleme Bilgisi
        if (i+1) % 2000 == 0:
            print(f"   -> {i+1} samples processed...")

    print(f"✅ Feature extraction completed ({time.time() - start_feat_time:.2f} sn).")
    print(f"   New Feature Vector Size: {len(features_list[0])}")
    return np.array(features_list)

# ---------------------------------------------------------
# 3. VERİ YÜKLEME VE ÖN İŞLEME
# ---------------------------------------------------------
def load_and_preprocess(csv_path):
    print(f"📂 Dosya okunuyor: {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print("❌ Error: File is not found.")
        return None, None, None, None

    X_raw = df.iloc[:, 0:1800].values
    y_raw = df.iloc[:, 1800].values
    
    X_features = extract_features(X_raw)
    
    # Etiket Kodlama (Label Encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    # Eğitim/Test Ayrımı (%70 Eğitim, %30 Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Temizlik (Imputation)
    print("🧹 Data is being imputed and scaled...")
    X_train = np.where(np.isfinite(X_train), X_train, 0)
    X_test = np.where(np.isfinite(X_test), X_test, 0)
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Ölçekleme (Standard Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ---------------------------------------------------------
# 4. MODELLERİN EĞİTİMİ VE KIYASLANMASI
# ---------------------------------------------------------
def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}
    trained_models = {} 

    print("\n🚀 Training Models...")
    print("-" * 85)
    print(f"{'Model':<20} | {'Time (s)':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 85)

    for name, model in models.items():
        try:
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            elapsed_time = time.time() - start_time
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1
            }
            trained_models[name] = model
            
            print(f"{name:<20} | {elapsed_time:<10.2f} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
            
        except Exception as e:
            print(f"❌ {name} Error: {e}")

    return results, trained_models

def tune_hyperparameters(X_train, y_train, model_name):
    print(f"\n🔍 Hyperparameter Tuning for {model_name}...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }
        
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        param_dist = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }
        
    elif model_name == "LightGBM":
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_dist = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 60, 100],
            'feature_fraction': [0.7, 0.8, 0.9]
        }
    
    else:
        print(f"⚠️ Tuning Parameters are not defined for {model_name}.")
        return None

    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, 
        n_iter=10, cv=cv, scoring='f1_macro', 
        n_jobs=-1, verbose=1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"✅ En İyi Parametreler ({model_name}): {random_search.best_params_}")
    return random_search.best_estimator_
# ---------------------------------------------------------
# 5. GÖRSELLEŞTİRME FONKSİYONLARI
# ---------------------------------------------------------
def visualize_results(results, trained_models, X_test, y_test):
    results_df = pd.DataFrame(results).T.sort_values(by='F1 Score', ascending=False)
    
    print("\n" + "="*50)
    print("🏆 Overall Performance Table")
    print("="*50)
    print(results_df)

    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df, annot=True, cmap='RdYlGn', fmt='.4f', linewidths=.5)
    plt.title('Performance metrics of all models')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=results_df.index, y=results_df['F1 Score'], palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('F1-score board of all models')
    plt.ylim(0, 1.1)
    for i, v in enumerate(results_df['F1 Score']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    print(f"\n🌟 Best performance: {best_model_name}")

    y_pred_best = best_model.predict(X_test)
    
    unique_classes = np.unique(y_test)
    current_names = [CLASS_NAMES[i] for i in unique_classes]

    cm = confusion_matrix(y_test, y_pred_best)

# Metrics
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)

    recall = np.nan_to_num(recall)
    precision = np.nan_to_num(precision)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)

    f1_macro = f1_score(y_test, y_pred_best, average='macro')

    n = cm.shape[0]

# Extended matrix
    cm_ext = np.zeros((n+1, n+1))
    cm_ext[:n, :n] = cm
    cm_ext[:n, -1] = recall
    cm_ext[-1, :n] = precision
    cm_ext[-1, -1] = f1_macro

# Labels
    labels_x = current_names + ['Recall']
    labels_y = current_names + ['Precision']

# STYLE
    plt.figure(figsize=(14, 12))
    sns.set(style="white")

# --- BASE CM (Blues) ---
    sns.heatmap(cm_ext[:n, :n],
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            xticklabels=current_names,
            yticklabels=current_names)

# --- RECALL (RIGHT SIDE) ---
    sns.heatmap(cm_ext[:n, -1].reshape(-1,1),
            annot=np.array([f"{r*100:.1f}%" for r in recall]).reshape(-1,1),
            fmt='',
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar=False,
            xticklabels=['Recall'],
            yticklabels=False)

# --- PRECISION (BOTTOM) ---
    sns.heatmap(cm_ext[-1, :n].reshape(1,-1),
            annot=np.array([f"{p*100:.1f}%" for p in precision]).reshape(1,-1),
            fmt='',
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar=False,
            xticklabels=False,
            yticklabels=['Precision'])

# --- F1 CELL ---
    sns.heatmap(np.array([[f1_macro]]),
            annot=np.array([[f"F1\n{f1_macro*100:.1f}%"]]),
            fmt='',
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar=False,
            xticklabels=False,
            yticklabels=False)

# Axis düzenleme
    plt.xticks(np.arange(n+1)+0.5, labels_x, rotation=45, ha='right')
    plt.yticks(np.arange(n+1)+0.5, labels_y, rotation=0)

    plt.title(f'{best_model_name} - Confusion Matrix (Paper Style)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()
    

    report = classification_report(y_test, y_pred_best, target_names=current_names, output_dict=True)
    
    class_metrics = []
    for class_name in current_names:
        metrics = report[class_name]
        class_metrics.append({
            'Class': class_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1-score']
        })
    
    df_class = pd.DataFrame(class_metrics)
    
    print("\n" + "="*50)
    print(f"📊 {best_model_name} - Detailed Activity Prediction Performance")
    print("="*50)
    print(df_class.sort_values(by='F1-Score', ascending=False).to_string(index=False))

    # Matplotlib ile Tablo Görselleştirme
    plt.figure(figsize=(12, len(df_class) * 0.6 + 1))
    ax = plt.subplot(111, frame_on=False) # Eksenleri kapat
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Tablo verilerini hazırla (Sayıları yuvarla)
    table_data = df_class.sort_values(by='F1-Score', ascending=False).copy()
    table_data[['Precision', 'Recall', 'F1-Score']] = table_data[['Precision', 'Recall', 'F1-Score']].round(4)

    table = plt.table(cellText=table_data.values,
                      colLabels=table_data.columns,
                      loc='center',
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8) # Satır yüksekliği ve genişliği ayarla

    plt.title(f"For {best_model_name} Success Score of Each Activity (F1-Score)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    # -------------------------------------------------------------

    df_melted = df_class.melt(id_vars='Class', var_name='Metric', value_name='Score')

    plt.figure(figsize=(16, 8))
    sns.barplot(data=df_melted, x='Class', y='Score', hue='Metric', palette='magma')
    plt.title(f'{best_model_name} - Class Based Activity Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 6. ANA PROGRAM (MAIN)
# ---------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(CSV_FILE_PATH)
    
    if X_train is not None:
        # 1. Aşama: Tüm modelleri default ayarlarla test et
        results, trained_models = train_evaluate_models(X_train, X_test, y_train, y_test)
        
        # 2. Aşama: En iyi modeli bul
        results_df = pd.DataFrame(results).T
        best_model_name = results_df.sort_values(by='F1 Score', ascending=False).index[0]
        print(f"\n🏆 Model with the Highest F1-score: {best_model_name}")

        # 3. Aşama: Sadece en iyi model üzerinde Tuning yap (Eğer tuning tanımlıysa)
        if best_model_name in ["Random Forest", "XGBoost", "LightGBM"]:
            best_tuned_model = tune_hyperparameters(X_train, y_train, best_model_name)
            
            if best_tuned_model:
                # Tuned modeli test et
                y_pred_tuned = best_tuned_model.predict(X_test)
                tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')
                tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
                print(f"✨ Tuned {best_model_name} F1 Score: {tuned_f1:.4f}")
                
                # Mevcut sonuçları görselleştirmek için en iyi modeli tuned versiyonla değiştir
                trained_models[best_model_name] = best_tuned_model
        
        # 4. Aşama: Görselleştirme
        visualize_results(results, trained_models, X_test, y_test)
        
        print("\n✅ PROGRAM COMPLETED SUCCESSFULLY.")