import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import shap

class RainfallClassificationModel:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Pra-pemrosesan data termasuk normalisasi dan penanganan nilai yang hilang
        """
        # Penanganan nilai yang hilang
        print("Memeriksa nilai yang hilang...")
        print(df.isnull().sum())
        
        # Mengisi nilai yang hilang
        # Untuk kolom numerik, gunakan median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Untuk kolom kategorik (jika ada), gunakan modus
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Normalisasi arah angin dalam bentuk sudut (0-360)
        if 'ddd_x' in df.columns:
            df['ddd_x'] = df['ddd_x'] % 360
        if 'ddd_car' in df.columns:
            df['ddd_car'] = df['ddd_car'] % 360
        
        return df
    
    def define_target(self, df, column='RR', bins=None, labels=None):
        """
        Mendefinisikan variabel target berdasarkan curah hujan
        Default: 
        - Tidak hujan (0 mm)
        - Hujan ringan (0-20 mm)
        - Hujan sedang (20-50 mm)
        - Hujan lebat (50-100 mm)
        - Hujan sangat lebat (>100 mm)
        """
        if bins is None:
            bins = [0, 0.1, 20, 50, 100, float('inf')]
        if labels is None:
            labels = ['Tidak Hujan', 'Ringan', 'Sedang', 'Lebat', 'Sangat Lebat']
        
        df['RainCategory'] = pd.cut(df[column], bins=bins, labels=labels)
        
        # Konversi kategori ke kode numerik
        df['RainClass'] = df['RainCategory'].cat.codes
        
        print("Distribusi kelas curah hujan:")
        print(df['RainCategory'].value_counts())
        
        return df
    
    def prepare_features(self, df, target_column='RainClass', scale=True):
        """
        Menyiapkan fitur dan variabel target
        """
        # Memisahkan fitur dan target
        X = df.drop([target_column, 'RainCategory'], axis=1, errors='ignore')
        
        # Menyimpan nama fitur
        self.feature_names = X.columns.tolist()
        
        # Hapus 'RR' dari fitur karena itu adalah target prediksi
        if 'RR' in X.columns:
            print("Menghapus 'RR' (curah hujan) dari fitur untuk menghindari kebocoran data.")
            X = X.drop('RR', axis=1)
            # Update feature names setelah drop RR
            self.feature_names = X.columns.tolist()
        
        y = df[target_column]
        
        # Membagi dataset menjadi data latih dan data uji
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Normalisasi fitur
        if scale:
            self.scaler = StandardScaler()
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns
            )
        
        print(f"Jumlah fitur: {self.X_train.shape[1]}")
        print(f"Jumlah sampel latih: {self.X_train.shape[0]}")
        print(f"Jumlah sampel uji: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def evaluate_feature_importance(self, method='random_forest'):
        """
        Mengevaluasi kepentingan fitur menggunakan beberapa metode
        """
        if method == 'random_forest':
            # Menggunakan Random Forest untuk mengevaluasi kepentingan fitur
            rf = RandomForestClassifier(random_state=42)
            rf.fit(self.X_train, self.y_train)
            
            # Mendapatkan kepentingan fitur
            importances = rf.feature_importances_
            self.feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=self.feature_importance)
            plt.title('Kepentingan Fitur (Random Forest)')
            plt.show()
            
            print("Kepentingan Fitur (Random Forest):")
            print(self.feature_importance)
            
        elif method == 'anova':
            # Menggunakan ANOVA F-test untuk seleksi fitur
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(self.X_train, self.y_train)
            
            # Mendapatkan skor
            scores = selector.scores_
            self.feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'F-Score': scores,
                'P-Value': selector.pvalues_
            }).sort_values(by='F-Score', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='F-Score', y='Feature', data=self.feature_importance)
            plt.title('Kepentingan Fitur (ANOVA F-test)')
            plt.show()
            
            print("Kepentingan Fitur (ANOVA F-test):")
            print(self.feature_importance)
        
        elif method == 'rfe':
            # Menggunakan Recursive Feature Elimination
            estimator = RandomForestClassifier(random_state=42)
            rfe = RFE(estimator, n_features_to_select=1, step=1)
            rfe.fit(self.X_train, self.y_train)
            
            # Mendapatkan peringkat fitur
            ranks = rfe.ranking_
            self.feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Rank': ranks
            }).sort_values(by='Rank')
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Rank', y='Feature', data=self.feature_importance)
            plt.title('Peringkat Fitur (RFE)')
            plt.show()
            
            print("Peringkat Fitur (RFE):")
            print(self.feature_importance)
        
        return self.feature_importance
    
    def select_features(self, k=None, threshold=None, method='importance'):
        """
        Memilih fitur berdasarkan kepentingan atau metode lain
        """
        if self.feature_importance is None:
            self.evaluate_feature_importance()
        
        if method == 'importance':
            if k is not None:
                # Pilih k fitur teratas
                selected_features = self.feature_importance.iloc[:k]['Feature'].tolist()
            elif threshold is not None:
                # Pilih fitur dengan kepentingan di atas threshold
                selected_features = self.feature_importance[
                    self.feature_importance['Importance'] >= threshold
                ]['Feature'].tolist()
            else:
                # Jika tidak ada k atau threshold yang ditentukan, gunakan semua fitur
                selected_features = self.feature_names
        
        # Membatasi fitur yang dipilih
        self.selected_features = selected_features
        print(f"Fitur yang dipilih ({len(selected_features)}):")
        print(selected_features)
        
        # Memperbarui data latih dan uji
        self.X_train = self.X_train[selected_features]
        self.X_test = self.X_test[selected_features]
        
        return selected_features
    
    def build_and_compare_models(self):
        """
        Membangun dan membandingkan berbagai model klasifikasi
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nMembangun model {name}...")
            
            # Fitting model
            model.fit(self.X_train, self.y_train)
            
            # Prediksi
            y_pred = model.predict(self.X_test)
            
            # Evaluasi
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring='accuracy'
            )
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores)
            }
            
            print(f"Akurasi: {accuracy:.4f}")
            print(f"Presisi: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"Cross-validation (mean): {np.mean(cv_scores):.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            plt.show()
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        # Memilih model terbaik berdasarkan skor cross-validation
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nModel terbaik: {best_model_name}")
        print(f"Akurasi CV rata-rata: {results[best_model_name]['cv_mean']:.4f}")
        
        # Visualisasi perbandingan model
        plt.figure(figsize=(12, 6))
        
        # Perbandingan akurasi
        plt.subplot(1, 2, 1)
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        sns.barplot(x=accuracies, y=model_names)
        plt.title('Perbandingan Akurasi Model')
        plt.xlabel('Akurasi')
        
        # Perbandingan cross-validation
        plt.subplot(1, 2, 2)
        cv_means = [results[model]['cv_mean'] for model in model_names]
        
        sns.barplot(x=cv_means, y=model_names)
        plt.title('Perbandingan Cross-Validation')
        plt.xlabel('Skor CV Rata-rata')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def tune_best_model(self, param_grid=None):
        """
        Melakukan hyperparameter tuning pada model terbaik
        """
        if self.best_model is None:
            print("Anda harus memanggil build_and_compare_models() terlebih dahulu")
            return None
        
        print(f"\nMelakukan tuning pada model {self.best_model_name}...")
        
        # Parameter grid default untuk model yang berbeda
        default_param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                'penalty': ['l2', 'none']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        }
        
        # Menggunakan param_grid yang diberikan atau default
        if param_grid is None:
            param_grid = default_param_grids.get(
                self.best_model_name, 
                {"n_estimators": [100, 200, 300]}  # Default fallback
            )
        
        # Membuat objek GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.best_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fitting GridSearchCV
        grid_search.fit(self.X_train, self.y_train)
        
        # Mendapatkan hasil terbaik
        print("\nParameter terbaik:")
        print(grid_search.best_params_)
        print(f"Skor terbaik: {grid_search.best_score_:.4f}")
        
        # Mengevaluasi model terbaik pada data uji
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        print(f"\nHasil evaluasi model terbaik:")
        print(f"Akurasi: {accuracy:.4f}")
        print(f"Presisi: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Memperbarui model terbaik
        self.best_model = best_model
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Tuned {self.best_model_name}')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.show()
        
        return grid_search
    
    def evaluate_dataset_efficiency(self):
        """
        Mengevaluasi efisiensi dan efektivitas dataset
        """
        if self.best_model is None:
            print("Anda harus memanggil build_and_compare_models() terlebih dahulu")
            return None
        
        print("\nMengevaluasi efisiensi dan efektivitas dataset...")
        
        # 1. Evaluasi pengaruh ukuran dataset
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            # Membagi dataset dengan ukuran yang berbeda
            X_train_sub, _, y_train_sub, _ = train_test_split(
                self.X_train, self.y_train, 
                train_size=size, 
                random_state=42,
                stratify=self.y_train
            )
            
            # Melatih model
            model = self.models[self.best_model_name]
            model.fit(X_train_sub, y_train_sub)
            
            # Mengevaluasi
            train_score = model.score(X_train_sub, y_train_sub)
            test_score = model.score(self.X_test, self.y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
            print(f"Ukuran data latih: {len(X_train_sub)} ({size*100:.0f}%)")
            print(f"Skor data latih: {train_score:.4f}")
            print(f"Skor data uji: {test_score:.4f}")
        
        # Visualisasi
        plt.figure(figsize=(10, 6))
        plt.plot([size*100 for size in train_sizes], train_scores, 'o-', label='Skor Latih')
        plt.plot([size*100 for size in train_sizes], test_scores, 'o-', label='Skor Uji')
        plt.xlabel('Persentase Data Latih (%)')
        plt.ylabel('Akurasi')
        plt.title('Kurva Belajar - Pengaruh Ukuran Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 2. Evaluasi pengaruh jumlah fitur
        feature_numbers = [1, 2, 3, 5, 7, 10, len(self.selected_features)]
        feature_numbers = [n for n in feature_numbers if n <= len(self.selected_features)]
        
        test_scores = []
        feature_sets = []
        
        for n in feature_numbers:
            # Memilih n fitur teratas
            if self.feature_importance is None:
                self.evaluate_feature_importance()
            
            top_features = self.feature_importance.iloc[:n]['Feature'].tolist()
            feature_sets.append(top_features)
            
            # Membuat subset data
            X_train_sub = self.X_train[top_features]
            X_test_sub = self.X_test[top_features]
            
            # Melatih model
            model = self.models[self.best_model_name]
            model.fit(X_train_sub, self.y_train)
            
            # Mengevaluasi
            score = model.score(X_test_sub, self.y_test)
            test_scores.append(score)
            
            print(f"\nJumlah fitur: {n}")
            print(f"Fitur yang digunakan: {top_features}")
            print(f"Akurasi: {score:.4f}")
        
        # Visualisasi
        plt.figure(figsize=(10, 6))
        plt.plot(feature_numbers, test_scores, 'o-')
        plt.xlabel('Jumlah Fitur')
        plt.ylabel('Akurasi')
        plt.title('Pengaruh Jumlah Fitur terhadap Performa Model')
        plt.grid(True)
        plt.show()
        
        # 3. Analisis fitur yang paling efisien
        if len(feature_numbers) > 1 and len(test_scores) > 1:
            # Hitung peningkatan akurasi per fitur
            accuracy_gains = []
            
            for i in range(1, len(feature_numbers)):
                prev_score = test_scores[i-1]
                curr_score = test_scores[i]
                feature_diff = feature_numbers[i] - feature_numbers[i-1]
                
                if feature_diff > 0:
                    gain = (curr_score - prev_score) / feature_diff
                    accuracy_gains.append(gain)
                    
                    print(f"\nPenambahan fitur dari {feature_numbers[i-1]} ke {feature_numbers[i]}:")
                    print(f"Fitur yang ditambahkan: {set(feature_sets[i]) - set(feature_sets[i-1])}")
                    print(f"Peningkatan akurasi: {curr_score - prev_score:.4f}")
                    print(f"Peningkatan per fitur: {gain:.4f}")
            
            # Visualisasi
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(accuracy_gains) + 1), accuracy_gains)
            plt.xlabel('Penambahan Fitur ke-n')
            plt.ylabel('Peningkatan Akurasi per Fitur')
            plt.title('Efisiensi Penambahan Fitur')
            plt.xticks(range(1, len(accuracy_gains) + 1))
            plt.grid(True)
            plt.show()
        
        # 4. Identifikasi fitur redundan
        if len(self.selected_features) > 1:
            # Menghitung korelasi antar fitur
            correlation = self.X_train[self.selected_features].corr()
            
            # Visualisasi
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Korelasi antar Fitur')
            plt.show()
            
            # Identifikasi pasangan fitur dengan korelasi tinggi
            high_corr_pairs = []
            
            for i in range(len(self.selected_features)):
                for j in range(i+1, len(self.selected_features)):
                    if abs(correlation.iloc[i, j]) >= 0.7:  # Threshold korelasi tinggi
                        high_corr_pairs.append((
                            self.selected_features[i],
                            self.selected_features[j],
                            correlation.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print("\nPasangan fitur dengan korelasi tinggi (potensial redundan):")
                for f1, f2, corr in high_corr_pairs:
                    print(f"{f1} - {f2}: {corr:.4f}")
            else:
                print("\nTidak ditemukan pasangan fitur dengan korelasi tinggi.")
        
        # 5. Analisis interpretabilitas model dengan SHAP
        if self.best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree', 'Gradient Boosting']:
            try:
                # Membuat SHAP explainer
                explainer = shap.Explainer(self.best_model)
                shap_values = explainer(self.X_test)
                
                # Visualisasi ringkasan SHAP
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, self.X_test)
                plt.title('SHAP Feature Importance')
                plt.show()
                
                # Visualisasi dependence plot untuk fitur teratas
                if len(self.selected_features) > 0:
                    top_feature = self.feature_importance.iloc[0]['Feature']
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        top_feature, 
                        shap_values.values, 
                        self.X_test,
                        feature_names=self.selected_features
                    )
                    plt.title(f'SHAP Dependence Plot - {top_feature}')
                    plt.show()
            except Exception as e:
                print(f"Error in SHAP analysis: {e}")
        
        # Kesimpulan dan rekomendasi
        print("\n=== Kesimpulan Evaluasi Dataset ===")
        
        # Efektivitas ukuran dataset
        if len(train_sizes) > 1 and len(train_scores) > 1 and len(test_scores) > 1:
            # Hitung perbedaan antara skor terakhir dan kedua dari terakhir
            last_diff = test_scores[-1] - test_scores[-2]
            
            if last_diff < 0.01:
                print("- Dataset sudah mencapai ukuran yang optimal. Penambahan data tidak memberikan peningkatan signifikan.")
            else:
                print("- Dataset masih dapat ditingkatkan dengan penambahan data lebih banyak.")
        
        # Efektivitas fitur
        if len(feature_numbers) > 1 and len(test_scores) > 1:
            max_score_idx = test_scores.index(max(test_scores))
            optimal_n_features = feature_numbers[max_score_idx]
            
            print(f"- Jumlah fitur optimal: {optimal_n_features}")
            print(f"- Fitur optimal: {feature_sets[max_score_idx]}")
            
            if optimal_n_features < len(self.selected_features):
                print("- Beberapa fitur dapat dihilangkan tanpa mengurangi performa model.")
            else:
                print("- Semua fitur yang dipilih berkontribusi positif terhadap performa model.")
        
        # Redundansi fitur
        if 'high_corr_pairs' in locals() and high_corr_pairs:
            print("- Terdapat fitur yang berpotensi redundan (korelasi tinggi).")
            print("  Pertimbangkan untuk menghilangkan salah satu dari pasangan berikut:")
            for f1, f2, corr in high_corr_pairs[:3]:  # Batasi output untuk 3 pasangan teratas
                print(f"  * {f1} - {f2}")
        
        # Rekomendasi akhir
        print("\n=== Rekomendasi untuk Peningkatan Dataset ===")
        
        if 'train_scores' in locals() and 'test_scores' in locals():
            if max(train_scores) - max(test_scores) > 0.1:
                print("- Model menunjukkan tanda overfitting. Pertimbangkan untuk:")
                print("  * Menambah lebih banyak data")
                print("  * Mengurangi kompleksitas model")
                print("  * Menerapkan regularisasi yang lebih kuat")
            
            if max(test_scores) < 0.7:
                print("- Performa model masih dapat ditingkatkan. Pertimbangkan untuk:")
                print("  * Menambah fitur baru yang lebih relevan")
                print("  * Mengeksplorasi transformasi fitur yang ada")
                print("  * Mencoba model yang lebih kompleks")
        
        return {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "test_scores": test_scores,
            "feature_numbers": feature_numbers,
            "feature_scores": test_scores,
            "feature_sets": feature_sets
        }
    
    def explain_model(self):
        """
        Menjelaskan model terbaik dengan analisis interpretatif
        """
        if self.best_model is None:
            print("Anda harus memanggil build_and_compare_models() terlebih dahulu")
            return None
        
        print(f"\nMenjelaskan model {self.best_model_name}...")
        
        # Prediksi pada data uji
        y_pred = self.best_model.predict(self.X_test)
        
        # 1. Contoh prediksi
        print("\nContoh prediksi:")
        n_samples = min(5, len(self.X_test))
        
        for i in range(n_samples):
            print(f"\nSampel {i+1}:")
            for feature, value in zip(self.X_test.columns, self.X_test.iloc[i]):
                print(f"- {feature}: {value:.4f}")
            print(f"Kelas aktual: {self.y_test.iloc[i]}")
            print(f"Kelas prediksi: {y_pred[i]}")
        
        # 2. Visualisasi prediksi vs aktual
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
        plt.xlabel('Nilai Aktual')
        plt.ylabel('Nilai Prediksi')
        plt.title('Perbandingan Nilai Aktual vs Prediksi')
        plt.grid(True)
        plt.show()
        
        # 3. Analisis kesalahan
        errors = (y_pred != self.y_test)
        error_indices = np.where(errors)[0]
        
        if len(error_indices) > 0:
            print(f"\nJumlah kesalahan klasifikasi: {len(error_indices)} dari {len(self.y_test)} sampel")
            print(f"Tingkat kesalahan: {len(error_indices) / len(self.y_test):.4f}")
            
            # Menganalisis pola kesalahan
            error_samples = self.X_test.iloc[error_indices]
            error_actual = self.y_test.iloc[error_indices]
            error_pred = y_pred[error_indices]
            
            # Confusion matrix untuk sampel yang salah
            cm_error = confusion_matrix(error_actual, error_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_error, annot=True, fmt='d', cmap='Reds')
            plt.title('Confusion Matrix - Kesalahan Klasifikasi')
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            plt.show()
            
            # Melihat beberapa contoh kesalahan
            print("\nContoh kesalahan klasifikasi:")
            n_error_samples = min(5, len(error_indices))
            
            for i in range(n_error_samples):
                idx = error_indices[i]
                print(f"\nSampel Error {i+1}:")
                for feature, value in zip(self.X_test.columns, self.X_test.iloc[idx]):
                    print(f"- {feature}: {value:.4f}")
                print(f"Kelas aktual: {self.y_test.iloc[idx]}")
                print(f"Kelas prediksi: {y_pred[idx]}")
        else:
            print("\nTidak ada kesalahan klasifikasi pada data uji!")
        
        # 4. Analisis model-specific
        if self.best_model_name == 'Random Forest' or self.best_model_name == 'Decision Tree':
            # Feature importance dari model
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), self.X_test.columns[indices], rotation=90)
            plt.tight_layout()
            plt.show()
            
            print("\nKepentingan Fitur:")
            for i in range(len(importances)):
                print(f"{self.X_test.columns[indices[i]]}: {importances[indices[i]]:.4f}")
        
        elif self.best_model_name == 'Logistic Regression':
            # Koefisien model
            if hasattr(self.best_model, 'coef_'):
                coefs = self.best_model.coef_[0]
                indices = np.argsort(np.abs(coefs))[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title('Koefisien Logistic Regression')
                plt.bar(range(len(coefs)), coefs[indices])
                plt.xticks(range(len(coefs)), self.X_test.columns[indices], rotation=90)
                plt.tight_layout()
                plt.show()
                
                print("\nKoefisien Model:")
                for i in range(len(coefs)):
                    print(f"{self.X_test.columns[indices[i]]}: {coefs[indices[i]]:.4f}")
        
        # 5. Visualisasi pembatas keputusan (untuk 2 fitur teratas)
        if len(self.X_test.columns) >= 2:
            # Mengambil 2 fitur teratas
            if self.feature_importance is not None:
                top_features = self.feature_importance.iloc[:2]['Feature'].tolist()
            else:
                top_features = self.X_test.columns[:2].tolist()
            
            X = self.X_test[top_features]
            y = self.y_test
            
            # Membuat mesh grid
            h = 0.02  # Ukuran step dalam mesh
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Visualisasi pembatas keputusan
            plt.figure(figsize=(10, 8))
            
            # Membuat titik-titik untuk prediksi
            Z = self.best_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot pembatas keputusan dan titik data
            plt.contourf(xx, yy, Z, alpha=0.3)
            scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', alpha=0.8)
            plt.xlabel(top_features[0])
            plt.ylabel(top_features[1])
            plt.title(f'Pembatas Keputusan - {self.best_model_name}')
            plt.colorbar(scatter)
            plt.show()
        
        return None

# Contoh penggunaan:
# ==================

# # 1. Memuat dataset
# df = pd.read_csv('data_curah_hujan.csv')

# # 2. Inisialisasi model
# model = RainfallClassificationModel()

# # 3. Pra-pemrosesan data
# df = model.preprocess_data(df)

# # 4. Mendefinisikan target
# df = model.define_target(df)

# # 5. Menyiapkan fitur
# X_train, X_test, y_train, y_test = model.prepare_features(df)

# # 6. Mengevaluasi kepentingan fitur
# feature_importance = model.evaluate_feature_importance(method='random_forest')

# # 7. Memilih fitur
# selected_features = model.select_features(k=5)

# # 8. Membangun dan membandingkan model
# results = model.build_and_compare_models()

# # 9. Melakukan tuning pada model terbaik
# grid_search = model.tune_best_model()

# # 10. Mengevaluasi efisiensi dan efektivitas dataset
# efficiency_metrics = model.evaluate_dataset_efficiency()

# # 11. Menjelaskan model
# model.explain_model()