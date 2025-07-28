import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import io
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧠 Enhanced AI Personality Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .model-performance {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .data-overview {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Enhanced data preprocessing functions
def comprehensive_eda(df):
    """Perform comprehensive exploratory data analysis"""
    st.subheader("📊 Comprehensive Data Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📋 Total Features", f"{df.shape[1]-1}")  # Exclude target
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("❓ Missing Data", f"{missing_pct:.1f}%")
    with col4:
        if 'Personality' in df.columns:
            unique_targets = df['Personality'].nunique()
            st.metric("🎯 Classes", unique_targets)
    
    # Data distribution visualization
    if 'Personality' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Target distribution
            target_counts = df['Personality'].value_counts()
            fig = px.pie(
                values=target_counts.values, 
                names=target_counts.index,
                title="🎯 Target Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature correlation heatmap (numerical features only)
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="🔥 Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

def advanced_preprocessing(df):
    """Enhanced preprocessing with detailed logging"""
    
    st.write("🔄 **Processing Data Pipeline:**")
    
    # Create a copy
    df_processed = df.copy()
    
    # Step 1: Handle ID column
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
        st.write("✅ Removed ID column")
    
    # Step 2: Identify column types
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if 'Personality' in categorical_cols:
        categorical_cols.remove('Personality')
    
    st.write(f"📊 **Numerical features:** {len(numerical_cols)}")
    st.write(f"📋 **Categorical features:** {len(categorical_cols)}")
    
    # Step 3: Handle missing values with strategy logging
    missing_before = df_processed.isnull().sum().sum()
    
    # Numerical imputation
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
        st.write("🔢 Applied median imputation for numerical features")
    
    # Categorical imputation
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
        st.write("📝 Applied mode imputation for categorical features")
    
    missing_after = df_processed.isnull().sum().sum()
    st.write(f"✨ **Missing values reduced:** {missing_before} → {missing_after}")
    
    # Step 4: Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        st.write(f"🏷️ Encoded categorical feature: {col}")
    
    return df_processed, label_encoders, numerical_cols, categorical_cols

def train_enhanced_models(X, y):
    """Train models with enhanced tracking and performance metrics"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    performance = {}
    
    # Model configurations
    model_configs = [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42, 
                class_weight='balanced'
            ),
            'emoji': '🌲'
        },
        {
            'name': 'SVM (RBF)',
            'model': SVC(
                kernel='rbf', 
                C=1.0,
                gamma='scale',
                probability=True, 
                random_state=42, 
                class_weight='balanced'
            ),
            'emoji': '🎯'
        },
        {
            'name': 'Neural Network',
            'model': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'emoji': '🧠'
        }
    ]
    
    # Train individual models
    individual_models = {}
    
    for i, config in enumerate(model_configs):
        start_time = time.time()
        status_text.text(f"{config['emoji']} Training {config['name']}...")
        
        model = config['model']
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        auc_score = roc_auc_score(y_test, test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        # Store results
        models[config['name']] = {
            'model': model,
            'scaler': scaler,
            'type': 'individual'
        }
        
        performance[config['name']] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': test_pred,
            'probabilities': test_proba,
            'training_time': training_time
        }
        
        # Store for ensemble
        model_key = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        individual_models[model_key] = model
        
        # Update progress
        progress_bar.progress((i + 1) / (len(model_configs) + 1))
    
    # Create and train ensemble
    status_text.text("🎭 Training Ensemble Model...")
    start_time = time.time()
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', individual_models['random_forest']),
            ('svm', individual_models['svm_rbf']),
            ('nn', individual_models['neural_network'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # Ensemble predictions and metrics
    ens_train_pred = ensemble.predict(X_train_scaled)
    ens_test_pred = ensemble.predict(X_test_scaled)
    ens_test_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    ens_train_acc = accuracy_score(y_train, ens_train_pred)
    ens_test_acc = accuracy_score(y_test, ens_test_pred)
    ens_auc = roc_auc_score(y_test, ens_test_proba)
    
    ens_cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    ens_training_time = time.time() - start_time
    
    models['Ensemble'] = {
        'model': ensemble,
        'scaler': scaler,
        'type': 'ensemble'
    }
    
    performance['Ensemble'] = {
        'train_accuracy': ens_train_acc,
        'test_accuracy': ens_test_acc,
        'auc_score': ens_auc,
        'cv_mean': ens_cv_scores.mean(),
        'cv_std': ens_cv_scores.std(),
        'predictions': ens_test_pred,
        'probabilities': ens_test_proba,
        'training_time': ens_training_time
    }
    
    progress_bar.progress(1.0)
    status_text.text("✅ All models trained successfully!")
    
    # Store test data for evaluation
    st.session_state.X_test = X_test_scaled
    st.session_state.y_test = y_test
    st.session_state.feature_names = X.columns.tolist()
    
    return models, performance

def display_enhanced_results(performance):
    """Enhanced results display with comprehensive metrics"""
    
    st.subheader("🏆 Model Performance Comparison")
    
    # Create comprehensive performance DataFrame
    perf_data = []
    for model_name, perf in performance.items():
        perf_data.append({
            'Model': model_name,
            'Train Acc': f"{perf['train_accuracy']:.4f}",
            'Test Acc': f"{perf['test_accuracy']:.4f}",
            'AUC Score': f"{perf['auc_score']:.4f}",
            'CV Mean': f"{perf['cv_mean']:.4f}",
            'CV Std': f"{perf['cv_std']:.4f}",
            'Time (s)': f"{perf['training_time']:.2f}"
        })
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        models = list(performance.keys())
        test_accs = [performance[model]['test_accuracy'] for model in models]
        auc_scores = [performance[model]['auc_score'] for model in models]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Test Accuracy', x=models, y=test_accs, yaxis='y'))
        fig.add_trace(go.Bar(name='AUC Score', x=models, y=auc_scores, yaxis='y'))
        
        fig.update_layout(
            title='🎯 Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[min(min(test_accs), min(auc_scores)) - 0.05, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Training time comparison
        times = [performance[model]['training_time'] for model in models]
        
        fig = px.bar(
            x=models, 
            y=times,
            title='⏱️ Training Time Comparison',
            labels={'x': 'Models', 'y': 'Time (seconds)'},
            color=times,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_prediction_interface():
    """Enhanced prediction interface with better UX"""
    
    st.subheader("🔮 Make New Predictions")
    
    if st.session_state.selected_model and hasattr(st.session_state, 'feature_names') and len(st.session_state.feature_names) <= 15:
        
        with st.form("enhanced_prediction_form"):
            st.write("**Enter feature values for personality prediction:**")
            
            feature_values = {}
            
            # Organize features in columns
            num_cols = min(3, len(st.session_state.feature_names))
            cols = st.columns(num_cols)
            
            for i, feature in enumerate(st.session_state.feature_names):
                with cols[i % num_cols]:
                    # Determine appropriate input type based on feature name
                    if any(keyword in feature.lower() for keyword in ['time', 'hour', 'frequency', 'count']):
                        feature_values[feature] = st.number_input(
                            f"🕐 {feature.replace('_', ' ').title()}", 
                            value=0.0,
                            min_value=0.0,
                            max_value=24.0 if 'time' in feature.lower() else 100.0,
                            step=0.1,
                            key=f"input_{feature}",
                            help=f"Enter value for {feature}"
                        )
                    elif any(keyword in feature.lower() for keyword in ['size', 'circle', 'attendance']):
                        feature_values[feature] = st.number_input(
                            f"👥 {feature.replace('_', ' ').title()}", 
                            value=0.0,
                            min_value=0.0,
                            max_value=50.0,
                            step=1.0,
                            key=f"input_{feature}",
                            help=f"Enter value for {feature}"
                        )
                    else:
                        feature_values[feature] = st.number_input(
                            f"📊 {feature.replace('_', ' ').title()}", 
                            value=0.0,
                            step=0.1,
                            key=f"input_{feature}",
                            help=f"Enter value for {feature}"
                        )
            
            # Prediction button
            predict_button = st.form_submit_button("🎯 Predict Personality Type", use_container_width=True)
            
            if predict_button:
                try:
                    # Prepare input data
                    input_data = np.array(list(feature_values.values())).reshape(1, -1)
                    
                    # Get model and scaler
                    selected = st.session_state.selected_model
                    model = st.session_state.models[selected]['model']
                    scaler = st.session_state.models[selected]['scaler']
                    
                    # Scale input
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                    
                    # Get class labels
                    if hasattr(st.session_state, 'le_target'):
                        predicted_label = st.session_state.le_target.inverse_transform([prediction])[0]
                    else:
                        predicted_label = "Extrovert" if prediction == 1 else "Introvert"
                    
                    confidence = max(probability) * 100
                    
                    # Enhanced result display
                    if predicted_label.lower() == "introvert":
                        emoji = "🧘"
                        color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        description = "Thoughtful, reflective, and energized by solitude"
                    else:
                        emoji = "🌟"
                        color = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                        description = "Outgoing, social, and energized by interaction"
                    
                    st.markdown(f"""
                    <div style="background: {color}; padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin: 2rem 0; box-shadow: 0 15px 35px rgba(0,0,0,0.15);">
                        <h1>{emoji} Predicted Personality: {predicted_label} {emoji}</h1>
                        <h2>Confidence Level: {confidence:.1f}%</h2>
                        <p style="font-size: 1.1em; margin-top: 1rem;">{description}</p>
                        <p><strong>Model Used:</strong> {selected}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show prediction probabilities
                    st.subheader("📊 Prediction Probabilities")
                    prob_data = {
                        'Personality Type': ['Introvert', 'Extrovert'],
                        'Probability': [probability[0]*100, probability[1]*100]
                    }
                    
                    fig = px.bar(
                        prob_data, 
                        x='Personality Type', 
                        y='Probability',
                        title='Prediction Confidence Breakdown',
                        color='Probability',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("💡 Please check your input values and try again.")

# CORRECTED: Better model selection logic
def select_best_model(performance):
    """Intelligent model selection with multiple criteria"""
    
    # Calculate composite score considering accuracy, speed, and stability
    best_score = -1
    best_model = None
    
    for model_name, perf in performance.items():
        # Normalize metrics (0-1 scale)
        accuracy_score = perf['test_accuracy']
        speed_score = 1 / (1 + perf['training_time'])  # Faster = higher score
        stability_score = 1 / (1 + perf['cv_std'])    # More stable = higher score
        
        # Weighted composite score (accuracy is most important)
        composite_score = (0.7 * accuracy_score + 0.2 * speed_score + 0.1 * stability_score)
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = model_name
    
    return best_model, best_score

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Enhanced AI Personality Prediction System</h1>
        <h3>Advanced Machine Learning • Real Data Analysis • Comprehensive Insights</h3>
        <p>Upload your dataset • Train multiple models • Get accurate predictions with detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 System Controls")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
        <h3>📁 Data Upload</h3>
        <p>Upload your CSV file with personality features and target variable</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file", 
        type="csv",
        help="Upload a CSV file with personality data including a 'Personality' target column"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            st.session_state.data_loaded = True
            st.session_state.original_data = df
            
            # Success message
            st.markdown(f"""
            <div class="success-card">
                <h3>✅ Data Loaded Successfully!</h3>
                <p>Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comprehensive EDA
            comprehensive_eda(df)
            
            # Data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Missing values analysis
            if df.isnull().sum().sum() > 0:
                st.subheader("❓ Missing Values Analysis")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Feature': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df)) * 100
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            
            # Model training section
            if 'Personality' in df.columns:
                st.subheader("🤖 Advanced Model Training")
                
                if st.button("🚀 Train Enhanced Models", use_container_width=True):
                    with st.spinner("🔄 Processing data and training advanced models..."):
                        
                        # Enhanced preprocessing
                        df_processed, label_encoders, numerical_cols, categorical_cols = advanced_preprocessing(df)
                        
                        # Prepare features and target
                        X = df_processed.drop('Personality', axis=1)
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(df_processed['Personality'])
                        
                        # Store for later use
                        st.session_state.le_target = le_target
                        st.session_state.feature_names = X.columns.tolist()
                        
                        # Train enhanced models
                        models, performance = train_enhanced_models(X, y)
                        
                        st.session_state.models = models
                        st.session_state.model_performance = performance
                        st.session_state.model_trained = True
                        
                        # CORRECTED: Better best model selection
                        best_model_name, best_score = select_best_model(performance)
                        
                        # Store training history
                        st.session_state.training_history.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'dataset_shape': df.shape,
                            'best_model': best_model_name,
                            'best_score': best_score
                        })
                    
                    st.success("🎉 All enhanced models trained successfully!")
                
                # Model results and selection
                if st.session_state.model_trained:
                    display_enhanced_results(st.session_state.model_performance)
                    
                    # Model selection
                    st.subheader("🎯 Select Your Preferred Model")
                    
                    # CORRECTED: Find and display best model using improved logic
                    best_model_name, best_score = select_best_model(st.session_state.model_performance)
                    best_auc = st.session_state.model_performance[best_model_name]['auc_score']
                    
                    st.info(f"🏆 **Recommended Model**: {best_model_name} (AUC: {best_auc:.4f}, Composite Score: {best_score:.3f})")
                    
                    # Model selection buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("🌲 Random Forest", use_container_width=True):
                            st.session_state.selected_model = "Random Forest"
                    with col2:
                        if st.button("🎯 SVM (RBF)", use_container_width=True):
                            st.session_state.selected_model = "SVM (RBF)"
                    with col3:
                        if st.button("🧠 Neural Network", use_container_width=True):
                            st.session_state.selected_model = "Neural Network"
                    with col4:
                        if st.button("🎭 Ensemble", use_container_width=True):
                            st.session_state.selected_model = "Ensemble"
                    
                    # Auto-select best model button
                    if st.button("✨ Auto-Select Best Model", use_container_width=True):
                        st.session_state.selected_model = best_model_name
                        st.success(f"🎉 Selected: {best_model_name}")
                    
                    # Display selected model performance
                    if st.session_state.selected_model:
                        selected = st.session_state.selected_model
                        perf = st.session_state.model_performance[selected]
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🎯 Selected Model: {selected}</h2>
                            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                                <div>
                                    <h3>Test Accuracy</h3>
                                    <h2>{perf['test_accuracy']:.4f}</h2>
                                </div>
                                <div>
                                    <h3>AUC Score</h3>
                                    <h2>{perf['auc_score']:.4f}</h2>
                                </div>
                                <div>
                                    <h3>CV Score</h3>
                                    <h2>{perf['cv_mean']:.4f} ± {perf['cv_std']:.3f}</h2>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed model evaluation
                        st.subheader(f"📈 {selected} Model Detailed Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Confusion Matrix
                            if hasattr(st.session_state, 'y_test'):
                                y_pred = st.session_state.model_performance[selected]['predictions']
                                cm = confusion_matrix(st.session_state.y_test, y_pred)
                                
                                fig = px.imshow(
                                    cm,
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=['Introvert', 'Extrovert'],
                                    y=['Introvert', 'Extrovert'],
                                    color_continuous_scale='Blues',
                                    title=f'Confusion Matrix - {selected}'
                                )
                                fig.update_layout(width=400, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # ROC Curve
                            if hasattr(st.session_state, 'y_test'):
                                y_proba = st.session_state.model_performance[selected]['probabilities']
                                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=fpr, y=tpr,
                                    mode='lines',
                                    name=f'{selected} (AUC = {st.session_state.model_performance[selected]["auc_score"]:.3f})',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='red', dash='dash')
                                ))
                                fig.update_layout(
                                    title=f'ROC Curve - {selected}',
                                    xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate',
                                    width=400, height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance for tree-based models
                        if selected == "Random Forest":
                            st.subheader("🎯 Feature Importance Analysis")
                            
                            model = st.session_state.models[selected]['model']
                            importance = model.feature_importances_
                            feature_names = st.session_state.feature_names
                            
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importance
                            }).sort_values('Importance', ascending=True)
                            
                            fig = px.bar(
                                importance_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title="Feature Importance Ranking",
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Classification Report
                        if hasattr(st.session_state, 'y_test'):
                            st.subheader("📊 Detailed Classification Report")
                            y_pred = st.session_state.model_performance[selected]['predictions']
                            report = classification_report(
                                st.session_state.y_test, 
                                y_pred, 
                                target_names=['Introvert', 'Extrovert'],
                                output_dict=True
                            )
                            
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(4), use_container_width=True)
                        
                        # Prediction interface
                        create_prediction_interface()
            
            else:
                st.error("❌ **Missing Target Column**: Please ensure your dataset contains a 'Personality' column.")
        
        except Exception as e:
            st.error(f"❌ **Error loading data**: {str(e)}")
            st.info("💡 **Please ensure your CSV file has the correct format with a 'Personality' column.**")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="data-overview">
            <h2>📁 Upload Your Personality Dataset</h2>
            <p><strong>To get started with advanced personality prediction:</strong></p>
            <div style="margin: 1rem 0;">
                <p>📊 <strong>Upload Requirements:</strong></p>
                <ul>
                    <li>CSV file format with personality behavioral data</li>
                    <li>Must include a 'Personality' column (target variable)</li>
                    <li>Behavioral features (numerical or categorical)</li>
                    <li>At least 100+ samples for reliable training</li>
                </ul>
            </div>
            <div style="margin: 1rem 0;">
                <p>🤖 <strong>What You'll Get:</strong></p>
                <ul>
                    <li>4 advanced ML models (Random Forest, SVM, Neural Network, Ensemble)</li>
                    <li>Comprehensive model performance comparison</li>
                    <li>Cross-validation and statistical analysis</li>
                    <li>Interactive prediction interface</li>
                    <li>Feature importance and model interpretability</li>
                    <li>ROC curves and confusion matrices</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data format
        st.subheader("📋 Expected Data Format")
        
        sample_data = {
            'Time_spent_Alone': [8.5, 2.1, 6.3, 9.2, 3.7],
            'Stage_fear': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Social_event_attendance': [2, 8, 5, 1, 7],
            'Going_outside': [1.5, 6.8, 4.2, 2.1, 5.9],
            'Drained_after_socializing': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Friends_circle_size': [3, 12, 8, 2, 10],
            'Post_frequency': [1.2, 7.5, 4.1, 2.3, 6.8],
            'Personality': ['Introvert', 'Extrovert', 'Extrovert', 'Introvert', 'Extrovert']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.info("💡 **Pro Tip**: Your dataset can have any number of behavioral features, but must include a 'Personality' column with binary labels (Introvert/Extrovert).")
        
        # Training history if available
        if st.session_state.training_history:
            st.subheader("📈 Training History")
            history_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(history_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin-top: 2rem;">
        <h3>🧠 Enhanced AI Personality Prediction System</h3>
        <p><strong>Built with Advanced Machine Learning & Statistical Analysis</strong></p>
        <p>🚀 <em>Features: Multi-model Training • Cross-Validation • Feature Importance • Interactive Predictions</em></p>
        <p>🎯 <em>Models: Random Forest • SVM • Neural Networks • Ensemble Learning</em></p>
        <p>💡 <em>Version 2.0 - Enhanced with Smart Model Selection & Error Handling</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    
