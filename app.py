from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import numpy as np
import os
import pickle
import json
import io
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from model.rainfall_model import RainfallClassificationModel
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'rainfallanalysissecretkey'  # Needed for flash messages and session

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize global variables
global_model = None
global_df = None
global_X_train = None
global_X_test = None
global_y_train = None
global_y_test = None
global_feature_importance = None

# Function to create a base64 encoded image from matplotlib figure
def get_plot_as_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

# Function to create distribution plot
def create_distribution_plot(df, column):
    plt.figure(figsize=(10, 6))
    
    # Check if column exists and has valid data
    if column not in df.columns or df[column].empty:
        plt.text(0.5, 0.5, f"Tidak dapat membuat plot distribusi untuk kolom '{column}'.\nKolom tidak ditemukan atau tidak memiliki data.", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        try:
            sns.histplot(df[column], kde=True)
            plt.title(f'Distribusi {column}')
            plt.xlabel(column)
            plt.ylabel('Frekuensi')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error saat membuat plot distribusi untuk kolom '{column}':\n{str(e)}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
    
    img_str = get_plot_as_base64(plt.gcf())
    plt.close()
    return img_str

# Function to create correlation matrix
def create_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Check if there are enough numeric columns
    if numeric_df.shape[1] < 2:
        # Return a message if not enough numeric columns
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Tidak cukup kolom numerik untuk membuat matriks korelasi.\nMinimal diperlukan 2 kolom numerik.", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        img_str = get_plot_as_base64(plt.gcf())
        plt.close()
        return img_str
    
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Matriks Korelasi')
    img_str = get_plot_as_base64(plt.gcf())
    plt.close()
    return img_str

# Function to check if uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global global_df
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read the file based on its extension
                if filename.endswith('.csv'):
                    global_df = pd.read_csv(filepath)
                elif filename.endswith(('.xlsx', '.xls')):
                    global_df = pd.read_excel(filepath)
                
                # Check if dataframe is empty
                if global_df.empty:
                    flash('File yang diunggah tidak memiliki data.')
                    return redirect(request.url)
                
                # Store file info in session
                session['filename'] = filename
                
                # Preview the data
                preview = global_df.head(5).to_html(classes='table table-striped table-hover')
                
                # Generate data summary
                summary = {
                    'shape': global_df.shape,
                    'columns': global_df.columns.tolist(),
                    'dtypes': global_df.dtypes.astype(str).to_dict(),
                    'missing_values': global_df.isnull().sum().to_dict()
                }
                
                # Create distribution plots for numeric columns
                numeric_cols = global_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                distribution_plots = {}
                
                # Limit to first 5 numeric columns or fewer if not enough
                plot_cols = numeric_cols[:min(5, len(numeric_cols))]
                for col in plot_cols:
                    distribution_plots[col] = create_distribution_plot(global_df, col)
                
                # Create correlation matrix
                correlation_matrix = create_correlation_matrix(global_df)
                
                return render_template('analyze.html', 
                                      preview=preview, 
                                      summary=summary, 
                                      filename=filename,
                                      distribution_plots=distribution_plots,
                                      correlation_matrix=correlation_matrix)
            except Exception as e:
                flash(f'Error saat memproses file: {str(e)}')
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    global global_df, global_model
    
    if global_df is None:
        flash('No data uploaded yet')
        return redirect(url_for('upload_file'))
    
    # Get preprocessing parameters from form
    target_column = request.form.get('target_column', 'RR')
    bins = request.form.getlist('bins')
    labels = request.form.getlist('labels')
    
    # Convert bins to float
    bins = [float(bin_val) if bin_val else None for bin_val in bins if bin_val]
    
    # Initialize model
    global_model = RainfallClassificationModel()
    
    # Preprocess data
    global_df = global_model.preprocess_data(global_df)
    
    # Define target
    if bins and labels and len(bins) > 1 and len(labels) > 0:
        global_df = global_model.define_target(global_df, column=target_column, bins=bins, labels=labels)
    else:
        global_df = global_model.define_target(global_df, column=target_column)
    
    # Prepare features - RR will be automatically excluded as a feature
    global_X_train, global_X_test, global_y_train, global_y_test = global_model.prepare_features(global_df)
    
    # Show preprocessing results
    class_distribution = global_df['RainCategory'].value_counts().to_dict()
    
    # Create class distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='RainCategory', data=global_df)
    plt.title('Distribusi Kelas Curah Hujan')
    plt.xlabel('Kategori Curah Hujan')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    class_dist_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    return render_template('preprocess_results.html',
                          class_distribution=class_distribution,
                          class_dist_plot=class_dist_plot,
                          X_train_shape=global_X_train.shape,
                          X_test_shape=global_X_test.shape)

@app.route('/feature_importance', methods=['POST'])
def analyze_feature_importance():
    global global_model, global_feature_importance
    
    if global_model is None:
        flash('Model not initialized yet')
        return redirect(url_for('upload_file'))
    
    method = request.form.get('method', 'random_forest')
    
    # Evaluate feature importance
    global_feature_importance = global_model.evaluate_feature_importance(method=method)
    
    # Convert feature importance to dictionary for display
    importance_dict = global_feature_importance.to_dict('records')
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    if method == 'random_forest':
        sns.barplot(x='Importance', y='Feature', data=global_feature_importance)
        plt.title('Kepentingan Fitur (Random Forest)')
    elif method == 'anova':
        sns.barplot(x='F-Score', y='Feature', data=global_feature_importance)
        plt.title('Kepentingan Fitur (ANOVA F-test)')
    elif method == 'rfe':
        sns.barplot(x='Rank', y='Feature', data=global_feature_importance)
        plt.title('Peringkat Fitur (RFE)')
    
    feature_importance_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    return render_template('feature_importance.html',
                          importance_dict=importance_dict,
                          feature_importance_plot=feature_importance_plot,
                          method=method)

@app.route('/select_features', methods=['POST'])
def select_features():
    global global_model
    
    if global_model is None:
        flash('Model not initialized yet')
        return redirect(url_for('upload_file'))
    
    k = request.form.get('k')
    threshold = request.form.get('threshold')
    
    if k:
        k = int(k)
        selected_features = global_model.select_features(k=k)
    elif threshold:
        threshold = float(threshold)
        selected_features = global_model.select_features(threshold=threshold)
    else:
        selected_features = global_model.select_features()
    
    return render_template('selected_features.html',
                          selected_features=selected_features)

@app.route('/build_models', methods=['POST'])
def build_models():
    global global_model
    
    if global_model is None:
        flash('Model not initialized yet')
        return redirect(url_for('upload_file'))
    
    # Build and compare models
    results = global_model.build_and_compare_models()
    
    # Convert results to a format suitable for display
    model_results = {}
    for name, result in results.items():
        model_results[name] = {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'cv_mean': result['cv_mean']
        }
    
    # Create comparison plot
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
    model_comparison_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    # Save best model name in session
    session['best_model'] = global_model.best_model_name
    
    return render_template('model_results.html',
                          model_results=model_results,
                          best_model=global_model.best_model_name,
                          model_comparison_plot=model_comparison_plot)

@app.route('/tune_model', methods=['POST'])
def tune_model():
    global global_model
    
    if global_model is None or global_model.best_model is None:
        flash('Models not built yet')
        return redirect(url_for('build_models'))
    
    # Tune best model
    grid_search = global_model.tune_best_model()
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Evaluate tuned model
    y_pred = global_model.best_model.predict(global_model.X_test)
    accuracy = np.mean(y_pred == global_model.y_test)
    
    # Create confusion matrix plot
    cm = pd.crosstab(global_model.y_test, y_pred, 
                     rownames=['Aktual'], 
                     colnames=['Prediksi'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Tuned {global_model.best_model_name}')
    confusion_matrix_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    return render_template('tuned_model.html',
                          best_params=best_params,
                          best_score=best_score,
                          accuracy=accuracy,
                          confusion_matrix_plot=confusion_matrix_plot)

@app.route('/evaluate_dataset', methods=['POST'])
def evaluate_dataset():
    global global_model
    
    if global_model is None or global_model.best_model is None:
        flash('Models not built yet')
        return redirect(url_for('build_models'))
    
    # Evaluate dataset efficiency
    efficiency_metrics = global_model.evaluate_dataset_efficiency()
    
    # Create learning curve plot
    plt.figure(figsize=(10, 6))
    plt.plot([size*100 for size in efficiency_metrics['train_sizes']], 
             efficiency_metrics['train_scores'], 'o-', label='Skor Latih')
    plt.plot([size*100 for size in efficiency_metrics['train_sizes']], 
             efficiency_metrics['test_scores'], 'o-', label='Skor Uji')
    plt.xlabel('Persentase Data Latih (%)')
    plt.ylabel('Akurasi')
    plt.title('Kurva Belajar - Pengaruh Ukuran Dataset')
    plt.legend()
    plt.grid(True)
    learning_curve_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    # Create feature impact plot
    plt.figure(figsize=(10, 6))
    plt.plot(efficiency_metrics['feature_numbers'], 
             efficiency_metrics['feature_scores'], 'o-')
    plt.xlabel('Jumlah Fitur')
    plt.ylabel('Akurasi')
    plt.title('Pengaruh Jumlah Fitur terhadap Performa Model')
    plt.grid(True)
    feature_impact_plot = get_plot_as_base64(plt.gcf())
    plt.close()
    
    # Generate recommendations
    recommendations = []
    
    # Check for overfitting
    if max(efficiency_metrics['train_scores']) - max(efficiency_metrics['test_scores']) > 0.1:
        recommendations.append("Model menunjukkan tanda overfitting. Pertimbangkan untuk menambah data, mengurangi kompleksitas model, atau menerapkan regularisasi yang lebih kuat.")
    
    # Check for model performance
    if max(efficiency_metrics['test_scores']) < 0.7:
        recommendations.append("Performa model masih dapat ditingkatkan. Pertimbangkan untuk menambah fitur baru yang lebih relevan, mengeksplorasi transformasi fitur, atau mencoba model yang lebih kompleks.")
    
    # Check optimal number of features
    max_score_idx = efficiency_metrics['feature_scores'].index(max(efficiency_metrics['feature_scores']))
    optimal_n_features = efficiency_metrics['feature_numbers'][max_score_idx]
    optimal_features = efficiency_metrics['feature_sets'][max_score_idx]
    
    if optimal_n_features < len(global_model.selected_features):
        recommendations.append(f"Jumlah fitur optimal adalah {optimal_n_features}. Beberapa fitur dapat dihilangkan tanpa mengurangi performa model.")
    
    # Check if more data needed
    if efficiency_metrics['test_scores'][-1] > efficiency_metrics['test_scores'][-2]:
        recommendations.append("Dataset masih dapat ditingkatkan dengan penambahan data lebih banyak.")
    else:
        recommendations.append("Dataset sudah mencapai ukuran yang optimal. Penambahan data tidak memberikan peningkatan signifikan.")
    
    return render_template('dataset_evaluation.html',
                          learning_curve_plot=learning_curve_plot,
                          feature_impact_plot=feature_impact_plot,
                          recommendations=recommendations,
                          optimal_features=optimal_features,
                          optimal_n_features=optimal_n_features)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global global_model
    
    if request.method == 'GET':
        if global_model is None or global_model.best_model is None:
            flash('Model not trained yet')
            return redirect(url_for('index'))
        
        # Get feature names
        features = global_model.selected_features
        
        return render_template('predict.html', features=features)
    
    elif request.method == 'POST':
        if global_model is None or global_model.best_model is None:
            flash('Model not trained yet')
            return redirect(url_for('index'))
        
        # Get input values
        input_data = {}
        for feature in global_model.selected_features:
            input_data[feature] = float(request.form.get(feature, 0))
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale input data if scaler exists
        if global_model.scaler is not None:
            input_df = pd.DataFrame(
                global_model.scaler.transform(input_df),
                columns=input_df.columns
            )
        
        # Make prediction
        prediction = global_model.best_model.predict(input_df)[0]
        
        # Get prediction probabilities if available
        probabilities = {}
        if hasattr(global_model.best_model, 'predict_proba'):
            proba = global_model.best_model.predict_proba(input_df)[0]
            for i, p in enumerate(proba):
                probabilities[i] = p
        
        # Map prediction to category
        categories = global_df['RainCategory'].cat.categories.tolist()
        prediction_category = categories[prediction] if prediction < len(categories) else "Unknown"
        
        # Create probability plot if available
        prob_plot = None
        if probabilities:
            plt.figure(figsize=(10, 6))
            categories_names = [categories[i] if i < len(categories) else f"Class {i}" for i in probabilities.keys()]
            sns.barplot(x=categories_names, y=list(probabilities.values()))
            plt.title('Probabilitas Prediksi')
            plt.xlabel('Kategori Curah Hujan')
            plt.ylabel('Probabilitas')
            plt.xticks(rotation=45)
            prob_plot = get_plot_as_base64(plt.gcf())
            plt.close()
        
        return render_template('prediction_result.html',
                              input_data=input_data,
                              prediction=prediction,
                              prediction_category=prediction_category,
                              probabilities=probabilities,
                              prob_plot=prob_plot)

@app.route('/export_model', methods=['POST'])
def export_model():
    global global_model
    
    if global_model is None or global_model.best_model is None:
        flash('Model not trained yet')
        return redirect(url_for('index'))
    
    # Create model export directory if it doesn't exist
    export_dir = 'model_exports'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Export model using pickle
    model_filename = os.path.join(export_dir, 'rainfall_model.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(global_model.best_model, f)
    
    # Export scaler if it exists
    if global_model.scaler is not None:
        scaler_filename = os.path.join(export_dir, 'scaler.pkl')
        with open(scaler_filename, 'wb') as f:
            pickle.dump(global_model.scaler, f)
    
    # Export feature names
    feature_filename = os.path.join(export_dir, 'features.json')
    with open(feature_filename, 'w') as f:
        json.dump(global_model.selected_features, f)
    
    # Create download links
    downloads = {
        'model': model_filename,
        'scaler': scaler_filename if global_model.scaler is not None else None,
        'features': feature_filename
    }
    
    # Create model info
    model_info = {
        'type': global_model.best_model_name,
        'accuracy': accuracy_score(global_model.y_test, global_model.best_model.predict(global_model.X_test)),
        'features': global_model.selected_features,
        'export_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('export_model.html',
                          downloads=downloads,
                          model_info=model_info)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)