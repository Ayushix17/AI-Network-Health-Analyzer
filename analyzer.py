import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import sqlite3
from datetime import datetime, timedelta
import time
import random
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkDataGenerator:
    """Generates realistic network performance data for demonstration"""
    
    @staticmethod
    def generate_realtime_data():
        """Generate real-time network metrics"""
        base_time = datetime.now()
        
        # Simulate different network conditions
        conditions = ['normal', 'high_load', 'anomaly']
        condition = np.random.choice(conditions, p=[0.7, 0.2, 0.1])
        
        if condition == 'normal':
            latency = np.random.normal(50, 10)
            bandwidth = np.random.normal(85, 5)
            packet_loss = np.random.exponential(0.1)
            cpu_usage = np.random.normal(45, 8)
            memory_usage = np.random.normal(60, 10)
        elif condition == 'high_load':
            latency = np.random.normal(120, 20)
            bandwidth = np.random.normal(65, 10)
            packet_loss = np.random.exponential(0.5)
            cpu_usage = np.random.normal(75, 5)
            memory_usage = np.random.normal(80, 5)
        else:  # anomaly
            latency = np.random.normal(200, 30)
            bandwidth = np.random.normal(30, 15)
            packet_loss = np.random.exponential(2.0)
            cpu_usage = np.random.normal(90, 5)
            memory_usage = np.random.normal(95, 3)
        
        return {
            'timestamp': base_time,
            'latency_ms': max(0, latency),
            'bandwidth_mbps': max(0, bandwidth),
            'packet_loss_pct': min(100, max(0, packet_loss)),
            'cpu_usage_pct': min(100, max(0, cpu_usage)),
            'memory_usage_pct': min(100, max(0, memory_usage)),
            'active_connections': np.random.poisson(100),
            'error_rate_pct': max(0, np.random.exponential(0.5))
        }
    
    @staticmethod
    def generate_historical_data(days=30):
        """Generate historical network data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24 * 6):  # Every 10 minutes
            timestamp = start_date + timedelta(minutes=i*10)
            
            # Add daily and weekly patterns
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Business hours effect
            if 9 <= hour <= 17 and day_of_week < 5:
                load_multiplier = 1.5
            elif 18 <= hour <= 22:
                load_multiplier = 1.2
            else:
                load_multiplier = 0.8
            
            # Weekend effect
            if day_of_week >= 5:
                load_multiplier *= 0.7
            
            base_latency = 50 * load_multiplier
            base_bandwidth = 90 / load_multiplier
            
            # Add some random anomalies
            if np.random.random() < 0.02:  # 2% chance of anomaly
                base_latency *= 3
                base_bandwidth *= 0.3
            
            data.append({
                'timestamp': timestamp,
                'latency_ms': max(0, np.random.normal(base_latency, 10)),
                'bandwidth_mbps': max(0, np.random.normal(base_bandwidth, 8)),
                'packet_loss_pct': max(0, np.random.exponential(0.2)),
                'cpu_usage_pct': max(0, min(100, np.random.normal(50 * load_multiplier, 10))),
                'memory_usage_pct': max(0, min(100, np.random.normal(55 * load_multiplier, 12))),
                'active_connections': np.random.poisson(80 * load_multiplier),
                'error_rate_pct': max(0, np.random.exponential(0.3))
            })
        
        return pd.DataFrame(data)

class AnomalyDetector:
    """ML-based anomaly detection for network metrics"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        features = [
            'latency_ms', 'bandwidth_mbps', 'packet_loss_pct', 
            'cpu_usage_pct', 'memory_usage_pct', 'active_connections', 'error_rate_pct'
        ]
        return df[features].fillna(0)
    
    def fit(self, df):
        """Train the anomaly detection models"""
        features = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest.fit(features_scaled)
        self.dbscan.fit(features_scaled)
        self.is_fitted = True
        
        logger.info("Anomaly detection models trained successfully")
    
    def detect_anomalies(self, df):
        """Detect anomalies in the data"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before detecting anomalies")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Isolation Forest predictions (-1 for anomaly, 1 for normal)
        iso_predictions = self.isolation_forest.predict(features_scaled)
        iso_scores = self.isolation_forest.score_samples(features_scaled)
        
        # DBSCAN predictions (-1 for noise/anomaly)
        dbscan_predictions = self.dbscan.fit_predict(features_scaled)
        
        df['iso_anomaly'] = (iso_predictions == -1)
        df['iso_score'] = iso_scores
        df['dbscan_anomaly'] = (dbscan_predictions == -1)
        df['combined_anomaly'] = df['iso_anomaly'] | df['dbscan_anomaly']
        
        return df

class AlertSystem:
    """System for generating and managing alerts"""
    
    def __init__(self):
        self.alert_thresholds = {
            'latency_ms': 150,
            'bandwidth_mbps': 40,
            'packet_loss_pct': 2.0,
            'cpu_usage_pct': 85,
            'memory_usage_pct': 90,
            'error_rate_pct': 3.0
        }
        
    def check_threshold_alerts(self, data):
        """Check for threshold-based alerts"""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in data:
                value = data[metric]
                if (metric == 'bandwidth_mbps' and value < threshold) or \
                   (metric != 'bandwidth_mbps' and value > threshold):
                    alerts.append({
                        'type': 'threshold',
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'severity': self.get_severity(metric, value, threshold),
                        'timestamp': data.get('timestamp', datetime.now())
                    })
        
        return alerts
    
    def check_anomaly_alerts(self, data):
        """Check for ML-detected anomaly alerts"""
        alerts = []
        
        if data.get('combined_anomaly', False):
            alerts.append({
                'type': 'anomaly',
                'severity': 'medium',
                'iso_score': data.get('iso_score', 0),
                'timestamp': data.get('timestamp', datetime.now()),
                'description': 'ML model detected anomalous network behavior'
            })
        
        return alerts
    
    def get_severity(self, metric, value, threshold):
        """Determine alert severity"""
        if metric == 'bandwidth_mbps':
            ratio = threshold / value if value > 0 else float('inf')
        else:
            ratio = value / threshold
        
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        else:
            return 'medium'

class NetworkHealthAnalyzer:
    """Main application class"""
    
    def __init__(self):
        self.data_generator = NetworkDataGenerator()
        self.anomaly_detector = AnomalyDetector()
        self.alert_system = AlertSystem()
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect('network_metrics.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                latency_ms REAL,
                bandwidth_mbps REAL,
                packet_loss_pct REAL,
                cpu_usage_pct REAL,
                memory_usage_pct REAL,
                active_connections INTEGER,
                error_rate_pct REAL,
                iso_anomaly INTEGER,
                iso_score REAL,
                dbscan_anomaly INTEGER,
                combined_anomaly INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_metrics(self, data):
        """Store metrics in database"""
        conn = sqlite3.connect('network_metrics.db')
        
        columns = [
            'timestamp', 'latency_ms', 'bandwidth_mbps', 'packet_loss_pct',
            'cpu_usage_pct', 'memory_usage_pct', 'active_connections', 'error_rate_pct',
            'iso_anomaly', 'iso_score', 'dbscan_anomaly', 'combined_anomaly'
        ]
        
        values = [data.get(col, None) for col in columns]
        placeholders = ','.join(['?' for _ in columns])
        
        conn.execute(f'INSERT INTO metrics ({",".join(columns)}) VALUES ({placeholders})', values)
        conn.commit()
        conn.close()
    
    def load_historical_data(self, hours=24):
        """Load historical data from database"""
        conn = sqlite3.connect('network_metrics.db')
        
        query = f'''
            SELECT * FROM metrics 
            WHERE timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

def main():
    st.set_page_config(
        page_title="AI Network Health Analyzer",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌐 AI Network Health Analyzer")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = NetworkHealthAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Historical data range
    history_hours = st.sidebar.slider("Historical Data Range (hours)", 1, 168, 24)
    
    # Training controls
    st.sidebar.header("ML Model Training")
    if st.sidebar.button("Train Anomaly Detection Models"):
        with st.spinner("Training models..."):
            # Generate training data
            training_data = analyzer.data_generator.generate_historical_data(days=7)
            analyzer.anomaly_detector.fit(training_data)
            st.sidebar.success("Models trained successfully!")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Real-time Monitoring", "Historical Analysis", "Anomaly Detection", "Alert Management"])
    
    with tab1:
        st.header("Real-time Network Metrics")
        
        # Create containers for real-time data
        metrics_container = st.container()
        charts_container = st.container()
        
        # Get real-time data
        current_data = analyzer.data_generator.generate_realtime_data()
        
        # Check for anomalies if model is trained
        if analyzer.anomaly_detector.is_fitted:
            temp_df = pd.DataFrame([current_data])
            temp_df = analyzer.anomaly_detector.detect_anomalies(temp_df)
            current_data.update(temp_df.iloc[0].to_dict())
        
        # Store current data
        analyzer.store_metrics(current_data)
        
        # Display key metrics
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Latency",
                    f"{current_data['latency_ms']:.1f} ms",
                    delta=f"{current_data['latency_ms'] - 50:.1f}"
                )
            
            with col2:
                st.metric(
                    "Bandwidth",
                    f"{current_data['bandwidth_mbps']:.1f} Mbps",
                    delta=f"{current_data['bandwidth_mbps'] - 85:.1f}"
                )
            
            with col3:
                st.metric(
                    "Packet Loss",
                    f"{current_data['packet_loss_pct']:.2f}%",
                    delta=f"{current_data['packet_loss_pct'] - 0.1:.2f}"
                )
            
            with col4:
                st.metric(
                    "CPU Usage",
                    f"{current_data['cpu_usage_pct']:.1f}%",
                    delta=f"{current_data['cpu_usage_pct'] - 45:.1f}"
                )
        
        # Real-time charts
        with charts_container:
            # Load recent data for trending
            recent_data = analyzer.load_historical_data(hours=1)
            
            if not recent_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(recent_data, x='timestamp', y='latency_ms', 
                                title='Latency Trend (Last Hour)')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(recent_data, x='timestamp', y='bandwidth_mbps',
                                title='Bandwidth Trend (Last Hour)')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Check for alerts
        threshold_alerts = analyzer.alert_system.check_threshold_alerts(current_data)
        anomaly_alerts = analyzer.alert_system.check_anomaly_alerts(current_data)
        
        all_alerts = threshold_alerts + anomaly_alerts
        
        if all_alerts:
            st.header("🚨 Current Alerts")
            for alert in all_alerts:
                severity_color = {
                    'critical': 'red',
                    'high': 'orange',
                    'medium': 'yellow'
                }.get(alert.get('severity', 'medium'), 'yellow')
                
                st.error(f"**{alert.get('severity', 'medium').upper()}**: {alert}")
    
    with tab2:
        st.header("Historical Network Analysis")
        
        # Load historical data
        historical_data = analyzer.load_historical_data(hours=history_hours)
        
        if historical_data.empty:
            st.info("No historical data available. Please wait for data collection or run the real-time monitoring.")
        else:
            # Summary statistics
            st.subheader("Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Latency", f"{historical_data['latency_ms'].mean():.1f} ms")
                st.metric("Max Latency", f"{historical_data['latency_ms'].max():.1f} ms")
            
            with col2:
                st.metric("Avg Bandwidth", f"{historical_data['bandwidth_mbps'].mean():.1f} Mbps")
                st.metric("Min Bandwidth", f"{historical_data['bandwidth_mbps'].min():.1f} Mbps")
            
            with col3:
                st.metric("Avg Packet Loss", f"{historical_data['packet_loss_pct'].mean():.2f}%")
                st.metric("Max Packet Loss", f"{historical_data['packet_loss_pct'].max():.2f}%")
            
            with col4:
                st.metric("Avg CPU Usage", f"{historical_data['cpu_usage_pct'].mean():.1f}%")
                st.metric("Max CPU Usage", f"{historical_data['cpu_usage_pct'].max():.1f}%")
            
            # Time series charts
            st.subheader("Network Performance Trends")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Latency', 'Bandwidth', 'Packet Loss', 'CPU Usage'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['latency_ms'],
                          name='Latency (ms)', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['bandwidth_mbps'],
                          name='Bandwidth (Mbps)', line=dict(color='green')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['packet_loss_pct'],
                          name='Packet Loss (%)', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['cpu_usage_pct'],
                          name='CPU Usage (%)', line=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Metric Correlations")
            numeric_cols = ['latency_ms', 'bandwidth_mbps', 'packet_loss_pct', 
                          'cpu_usage_pct', 'memory_usage_pct', 'error_rate_pct']
            corr_matrix = historical_data[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Network Metrics Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("ML-Based Anomaly Detection")
        
        if not analyzer.anomaly_detector.is_fitted:
            st.warning("⚠️ Anomaly detection models are not trained yet. Please train them first.")
            if st.button("Quick Train (7 days of simulated data)"):
                with st.spinner("Training anomaly detection models..."):
                    training_data = analyzer.data_generator.generate_historical_data(days=7)
                    analyzer.anomaly_detector.fit(training_data)
                    st.success("Models trained successfully!")
                    st.rerun()
        else:
            # Load and analyze historical data
            historical_data = analyzer.load_historical_data(hours=history_hours)
            
            if not historical_data.empty and 'combined_anomaly' in historical_data.columns:
                # Anomaly statistics
                total_points = len(historical_data)
                anomaly_count = historical_data['combined_anomaly'].sum()
                anomaly_rate = (anomaly_count / total_points) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data Points", total_points)
                
                with col2:
                    st.metric("Anomalies Detected", int(anomaly_count))
                
                with col3:
                    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                
                # Anomaly visualization
                st.subheader("Anomaly Detection Results")
                
                # Time series with anomalies highlighted
                fig = go.Figure()
                
                # Normal data points
                normal_data = historical_data[~historical_data['combined_anomaly']]
                fig.add_trace(go.Scatter(
                    x=normal_data['timestamp'],
                    y=normal_data['latency_ms'],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=4)
                ))
                
                # Anomalous data points
                anomaly_data = historical_data[historical_data['combined_anomaly']]
                if not anomaly_data.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['latency_ms'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig.update_layout(
                    title='Latency with Anomaly Detection',
                    xaxis_title='Time',
                    yaxis_title='Latency (ms)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details
                if not anomaly_data.empty:
                    st.subheader("Recent Anomalies")
                    
                    # Display recent anomalies
                    recent_anomalies = anomaly_data.tail(10)
                    
                    for _, anomaly in recent_anomalies.iterrows():
                        with st.expander(f"Anomaly at {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Latency**: {anomaly['latency_ms']:.1f} ms")
                                st.write(f"**Bandwidth**: {anomaly['bandwidth_mbps']:.1f} Mbps")
                                st.write(f"**Packet Loss**: {anomaly['packet_loss_pct']:.2f}%")
                                st.write(f"**CPU Usage**: {anomaly['cpu_usage_pct']:.1f}%")
                            
                            with col2:
                                st.write(f"**Memory Usage**: {anomaly['memory_usage_pct']:.1f}%")
                                st.write(f"**Active Connections**: {int(anomaly['active_connections'])}")
                                st.write(f"**Error Rate**: {anomaly['error_rate_pct']:.2f}%")
                                st.write(f"**Anomaly Score**: {anomaly['iso_score']:.3f}")
            else:
                st.info("No anomaly detection data available. Please collect some data first.")
    
    with tab4:
        st.header("Alert Management & Thresholds")
        
        # Alert threshold configuration
        st.subheader("Alert Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Latency Threshold (ms)", 
                          value=analyzer.alert_system.alert_thresholds['latency_ms'],
                          key="latency_threshold")
            
            st.number_input("Bandwidth Threshold (Mbps)", 
                          value=analyzer.alert_system.alert_thresholds['bandwidth_mbps'],
                          key="bandwidth_threshold")
            
            st.number_input("Packet Loss Threshold (%)", 
                          value=analyzer.alert_system.alert_thresholds['packet_loss_pct'],
                          key="packet_loss_threshold")
        
        with col2:
            st.number_input("CPU Usage Threshold (%)", 
                          value=analyzer.alert_system.alert_thresholds['cpu_usage_pct'],
                          key="cpu_threshold")
            
            st.number_input("Memory Usage Threshold (%)", 
                          value=analyzer.alert_system.alert_thresholds['memory_usage_pct'],
                          key="memory_threshold")
            
            st.number_input("Error Rate Threshold (%)", 
                          value=analyzer.alert_system.alert_thresholds['error_rate_pct'],
                          key="error_threshold")
        
        if st.button("Update Thresholds"):
            analyzer.alert_system.alert_thresholds.update({
                'latency_ms': st.session_state.latency_threshold,
                'bandwidth_mbps': st.session_state.bandwidth_threshold,
                'packet_loss_pct': st.session_state.packet_loss_threshold,
                'cpu_usage_pct': st.session_state.cpu_threshold,
                'memory_usage_pct': st.session_state.memory_threshold,
                'error_rate_pct': st.session_state.error_threshold
            })
            st.success("Thresholds updated successfully!")
        
        # Alert history (simulated)
        st.subheader("Recent Alerts")
        
        # Generate some sample alerts for demonstration
        sample_alerts = [
            {"timestamp": datetime.now() - timedelta(minutes=5), 
             "type": "threshold", "metric": "latency_ms", "severity": "high", 
             "value": 180, "threshold": 150},
            {"timestamp": datetime.now() - timedelta(minutes=15), 
             "type": "anomaly", "severity": "medium", 
             "description": "Unusual network pattern detected"},
            {"timestamp": datetime.now() - timedelta(hours=1), 
             "type": "threshold", "metric": "cpu_usage_pct", "severity": "critical", 
             "value": 95, "threshold": 85}
        ]
        
        for alert in sample_alerts:
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(alert['severity'], "🟡")
            
            with st.expander(f"{severity_emoji} {alert['severity'].upper()} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                if alert['type'] == 'threshold':
                    st.write(f"**Metric**: {alert['metric']}")
                    st.write(f"**Value**: {alert['value']}")
                    st.write(f"**Threshold**: {alert['threshold']}")
                else:
                    st.write(f"**Description**: {alert['description']}")
                
                st.write(f"**Time**: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
    