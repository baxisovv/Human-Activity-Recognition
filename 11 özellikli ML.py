import pandas as pd
import numpy as np
import time
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. AYARLAR VE SABİTLER
# ---------------------------------------------------------
CSV_FILE_PATH = 'Time_domain_subsamples.csv'

# KU-HAR Sınıf İsimleri (0-17)
CLASS_NAMES = [
    "Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", 
    "Lay", "Lay-stand", "Pick", "Jump", "Push-up", 
    "Sit-up", "Walk", "Walk-backward", "Circle-walk", 
    "Run", "Stair-up", "Stair-down", "Table-tennis"
]

# ---------------------------------------------------------
# 2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE EXTRACTION)
# ---------------------------------------------------------
def calculate_entropy(signal):
    """Sinyal entropisi hesaplar"""
    counts, _ = np.histogram(signal, bins=10, density=True)
    counts = counts[counts > 0]
    return stats.entropy(counts)

def extract_features(data):
    """
    Ham sensör verisinden istatistiksel özellikler çıkarır.
    Girdi: (N, 1800) -> Çıktı: (N, Özellik Sayısı)
    """
    print("⚙️  Özellik Çıkarımı (Feature Extraction) Başlıyor...")
    features_list = []
    
    # Veriyi (Sample, Kanal, Zaman) formatına çevir: (N, 6, 300)
    # 6 Kanal: AccX, AccY, AccZ, GyroX, GyroY, GyroZ
    data_reshaped = data.reshape(data.shape[0], 6, 300)

    start_feat_time = time.time()
    
    for i, sample in enumerate(data_reshaped):
        sample_features = []
        for axis in sample:
            # İstatistiksel Özellikler
            sample_features.extend([
                np.mean(axis),              # Ortalama
                np.max(axis),               # Maksimum
                np.min(axis),               # Minimum
                np.median(axis),            # Medyan
                np.std(axis),               # Standart Sapma
                stats.skew(axis),           # Çarpıklık
                stats.kurtosis(axis),       # Basıklık
                np.sum(np.abs(axis)),       # SMA (Signal Magnitude Area)
                stats.iqr(axis),            # Çeyrekler Açıklığı
                np.sum(axis**2),            # Enerji
                calculate_entropy(axis)     # Entropi
            ])
        features_list.append(sample_features)
        
        # İlerleme Bilgisi
        if (i+1) % 5000 == 0:
            print(f"   -> {i+1} örnek işlendi...")

    print(f"✅ Özellik çıkarımı tamamlandı ({time.time() - start_feat_time:.2f} sn).")
    return np.array(features_list)

# ---------------------------------------------------------
# 3. VERİ YÜKLEME VE ÖN İŞLEME
# ---------------------------------------------------------
def load_and_preprocess(csv_path):
    print(f"📂 Dosya okunuyor: {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None, None, None, None

    # Veriyi ve Etiketi Ayır
    # İlk 1800 sütun sensör verisi, 1800. sütun Class ID
    X_raw = df.iloc[:, 0:1800].values
    y_raw = df.iloc[:, 1800].values
    
    # Özellik Çıkarımı
    X_features = extract_features(X_raw)
    
    # Etiket Kodlama (Label Encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    # Eğitim/Test Ayrımı (%70 Eğitim, %30 Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Temizlik (Imputation) - Sonsuz ve NaN değerler için
    print("🧹 Veri temizleniyor ve ölçekleniyor...")
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    X_test = np.where(np.isfinite(X_test), X_test, np.nan)
    
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
    trained_models = {} # En iyi modeli sonradan kullanmak için sakla

    print("\n🚀 MODELLER EĞİTİLİYOR... (Lütfen bekleyin)")
    print("-" * 85)
    print(f"{'Model':<20} | {'Süre (s)':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 85)

    for name, model in models.items():
        try:
            # Süre Ölçümü Başlat
            start_time = time.time()
            
            # Eğitim
            model.fit(X_train, y_train)
            
            # Tahmin
            y_pred = model.predict(X_test)
            
            # Süre Ölçümü Bitir
            elapsed_time = time.time() - start_time
            
            # Metrikler (Weighted Average)
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
            
            print(f"{name:<20} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
            
        except Exception as e:
            print(f"❌ {name} Hatası: {e}")

    return results, trained_models

# ---------------------------------------------------------
# 5. GÖRSELLEŞTİRME FONKSİYONLARI
# ---------------------------------------------------------
def visualize_results(results, trained_models, X_test, y_test):
    results_df = pd.DataFrame(results).T.sort_values(by='F1 Score', ascending=False)
    
    print("\n" + "="*50)
    print("🏆 GENEL PERFORMANS TABLOSU")
    print("="*50)
    print(results_df)

    # 1. Isı Haritası (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df, annot=True, cmap='RdYlGn', fmt='.4f', linewidths=.5)
    plt.title('Performance Metrics of all Models')
    plt.tight_layout()
    plt.show()

    # 2. F1 Score Sıralaması (Bar Chart)
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=results_df.index, y=results_df['F1 Score'], palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Modellerin F1 Score Sıralaması')
    plt.ylim(0, 1.1)
    for i, v in enumerate(results_df['F1 Score']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- EN İYİ MODEL ANALİZİ ---
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    print(f"\n🌟 SEÇİLEN EN İYİ MODEL: {best_model_name}")

    y_pred_best = best_model.predict(X_test)
    
    # Sınıf isimlerini test setinde var olanlara göre filtrele
    unique_classes = np.unique(y_test)
    current_names = [CLASS_NAMES[i] for i in unique_classes]

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=current_names, yticklabels=current_names)
    plt.title(f'{best_model_name} - Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 4. Sınıf Bazlı Detaylı Grafik
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
    df_melted = df_class.melt(id_vars='Class', var_name='Metric', value_name='Score')

    plt.figure(figsize=(16, 8))
    sns.barplot(data=df_melted, x='Class', y='Score', hue='Metric', palette='magma')
    plt.title(f'{best_model_name} - Sınıf Bazlı Performans Analizi')
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
        results, trained_models = train_evaluate_models(X_train, X_test, y_train, y_test)
        
        visualize_results(results, trained_models, X_test, y_test)
        
        print("\n✅ PROGRAM BAŞARIYLA TAMAMLANDI.")