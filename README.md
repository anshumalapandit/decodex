# Decodex: Regulatory Market Surveillance Analytics

**A comprehensive financial market analytics platform for detecting suspicious trading patterns, network anomalies, and counterparty risks.**

---

## 📋 Overview

Decodex is an advanced regulatory surveillance system designed to analyze trading data and identify potential market manipulation, suspicious trading patterns, and counterparty concentration risks. The platform reconstructs market structure through network analysis and applies machine learning-based risk scoring to flag suspicious activities.

### Key Features

- **Network Analysis**: Build and visualize trading networks to identify circular trading patterns and counterparty relationships
- **Structural Metrics**: Calculate reciprocity ratios, counterparty concentration, and triangular loops
- **Risk Assessment**: Composite risk scoring combining multiple risk factors
- **Interactive Dashboards**: Streamlit-powered web interface for real-time analytics
- **Suspicious Entity Detection**: Identify and analyze outlier traders and transactions
- **AI-Powered Insights**: Integration with Grok API for advanced pattern interpretation

---

## 🏗️ Project Structure

```
Decodex/
├── app.py                          # Main Streamlit application
├── analysis.py                     # Data loading and preprocessing
├── metrics.py                      # Structural analysis metrics calculation
├── risk_model.py                   # Risk scoring and aggregation
├── ai_insights.py                  # Grok API integration for AI analysis
├── model_decodex.ipynb            # Comprehensive Jupyter notebook with full analysis
├── requirements.txt                # Python dependencies
├── Trades/                         # Input trade data files (.xls format)
├── Orders/                         # Input order data files (.xls format)
├── suspicious_orders/              # Flagged suspicious orders
├── suspicious_trades/              # Flagged suspicious trades
└── validation/                     # Validation datasets
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pandas, numpy, streamlit, networkx, plotly
- Excel file support (xlrd)

### Installation Steps

1. **Clone or navigate to the project directory**
```bash
cd Decodex
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare Data Files**
   - Place trade data files (.xls) in the `Trades/` folder
   - Place order data files (.xls) in the `Orders/` folder
   - Ensure files contain columns: `BUY CLIENT CODE`, `SELL CLIENT CODE`, `TRADE_QUANTITY`, `TRADE_RATE`, `TRADE_TIME`

4. **Set up Environment Variables (Optional - for AI Insights)**
Create a `.env` file in the project root:
```
GROK_API_KEY=your_api_key_here
```

---

## 🚀 Usage

### Running the Web Dashboard

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` with the following tabs:

#### 1. **Overview Tab**
   - Total trades, unique buyers/sellers, scrip count metrics
   - Trade quantity trends over time
   - Price and quantity distribution histograms

#### 2. **Execution Layer & Network Tab**
   - Reciprocity ratio analysis
   - Circular trading detection (triangular loops)
   - Network density metrics
   - Interactive trading network visualization
   - Price timeline analysis

#### 3. **Intent Layer & Order Patterns Tab**
   - Order execution analysis
   - Trading intent reconstruction
   - Pattern matching and anomaly detection

#### 4. **Risk Assessment Tab**
   - Composite risk scoring
   - High-risk entity flagging
   - Risk factor decomposition

#### 5. **Suspicious Entity Analysis Tab**
   - Detailed analysis of flagged entities
   - Historical pattern comparison
   - Peer benchmarking

#### 6. **Prevention & Governance Tab**
   - Recommended actions
   - Control recommendations
   - Compliance guidelines

### Jupyter Notebook Analysis

Run comprehensive analysis in `model_decodex.ipynb`:

```bash
jupyter notebook model_decodex.ipynb
```

The notebook includes detailed exploratory data analysis, metric calculations, and visualization examples.

---

## 📊 Core Modules

### `analysis.py` - Data Processing
**Functions:**
- `load_trades_file(file_path)`: Loads Excel trade data
- `prepare_trades_dataframe(df)`: Standardizes column names and formats

**Supported Columns:**
- BUY CLIENT CODE → buyer
- SELL CLIENT CODE → seller
- TRADE_QUANTITY → quantity
- TRADE_RATE → price
- TRADE_TIME → time

### `metrics.py` - Structural Analysis
**Key Metrics:**
- **Counterparty Concentration**: Ratio of volume with top counterparty to total volume
- **Reciprocity**: Bidirectional trading frequency (A trades with B and B trades with A)
- **Triangular Loops**: Detects 3-node circular trading patterns
- **Network Density**: Overall interconnectedness of the trading network

**Main Functions:**
- `compute_counterparty_concentration(df)`: Client concentration analysis
- `compute_reciprocity(df)`: Reciprocal trading detection
- `build_trade_graph(df)`: NetworkX graph construction
- `detect_triangular_loops(G)`: Circular pattern identification

### `risk_model.py` - Risk Scoring
**Composite Risk Score Formula:**
```
Risk Score = 0.4 × Concentration + 0.3 × Reciprocity + 0.3 × Loop Presence
```

**Function:**
- `compute_risk_score(concentration_df, reciprocity_ratio, loop_count)`: Aggregates metrics into risk scores

### `ai_insights.py` - AI Integration
**Functionality:**
- Sends analysis summaries to Grok API (x.ai)
- Generates regulatory interpretation and insights
- Leverages LLM for advanced pattern analysis

**Function:**
- `generate_structural_insight(summary_text)`: Calls Grok API for analysis

---

## 🎯 Analysis Workflow

1. **Data Ingestion**
   - Load trade and order data from Excel files
   - Validate schema and data quality

2. **Structural Decomposition**
   - Separate trades into execution and intent layers
   - Analyze buyer-seller relationships

3. **Metric Computation**
   - Calculate counterparty concentration
   - Measure reciprocity patterns
   - Detect circular trading loops
   - Assess network structure

4. **Risk Aggregation**
   - Combine metrics using weighted scoring
   - Rank entities by risk

5. **Suspicious Entity Flagging**
   - Identify outliers
   - Generate alerts for high-risk patterns

6. **AI-Powered Insights**
   - Leverage Grok API for advanced interpretation
   - Generate regulatory narrative

---

## 📈 Suspicious Trading Patterns Detected

### 1. **Counterparty Concentration Risk**
   - High volume concentrated with single counterparty
   - Potential for market manipulation or collusion

### 2. **Reciprocal Trading**
   - A consistently buys from B, and B consistently buys from A
   - May indicate artificial liquidity or price manipulation

### 3. **Triangular Loops**
   - A → B → C → A circular trading chains
   - Classic sign of wash trading or collusion
   - Self-dealing through intermediaries

### 4. **Abnormal Network Patterns**
   - Unusual density or clustering
   - Isolated networks of traders
   - Hub-and-spoke manipulation structures

---

## 🔐 Data Files

### Input Files
- **Trades/**: Trade execution records with buyer, seller, quantity, price, and time
- **Orders/**: Order placement and execution data

### Output Files
- **suspicious_orders/**: Flagged order records
- **suspicious_trades/**: Flagged trade records
- **validation/**: Validation datasets and benchmarks

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| streamlit | 1.32.2 | Web dashboard |
| networkx | 3.2.1 | Network analysis |
| plotly | 5.20.0 | Interactive visualizations |
| matplotlib | 3.8.4 | Static visualizations |
| scipy | 1.12.0 | Scientific computing |
| requests | 2.31.0 | HTTP requests (Grok API) |
| python-dotenv | 1.0.1 | Environment variable management |

---

## 🔄 Workflow Example

```python
# 1. Load data
df_trades = prepare_trades_dataframe(load_trades_file("Trades/SCRIP_2007APR13.xls"))

# 2. Compute metrics
concentration_df = compute_counterparty_concentration(df_trades)
reciprocity = compute_reciprocity(df_trades)
G = build_trade_graph(df_trades)
triangles = detect_triangular_loops(G)

# 3. Calculate risk scores
risk_results = compute_risk_score(concentration_df, reciprocity, len(triangles))

# 4. Generate AI insights
summary = f"Reciprocity: {reciprocity}, Loops: {len(triangles)}, ..."
insights = generate_structural_insight(summary)
```

---

## 🎨 Visualization Features

- **Trading Network Graphs**: Interactive visualization of buyer-seller relationships
- **Time Series Charts**: Price and volume trends
- **Distribution Plots**: Quantity and price histograms
- **Network Metrics**: Density, clustering coefficient, centrality measures
- **Risk Heatmaps**: Entity-level risk scoring

---

## 📝 Notes

- **Data Format**: Ensure Excel files use standard column naming
- **Time Handling**: TRADE_TIME should be parseable as datetime
- **Network Size**: Large networks (>50 nodes) may have optimized visualization
- **API Integration**: Grok API key is optional; system functions without AI insights
- **Baseline Validation Phase**: This is the initial validation phase of development

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "File not found" | Ensure .xls files are in Trades/ and Orders/ folders |
| Column name errors | Verify column names match expected format in analysis.py |
| API connection error | Check GROK_API_KEY in .env file |
| Network visualization fails | Reduce network size or check node count |

---

## 📄 License

This project is part of regulatory surveillance analytics advisory services.

---

## 👥 Development

**Current Phase**: Baseline Validation Phase

**Key Technologies**:
- Network Analysis (NetworkX)
- Statistical Analysis (Pandas, Scipy, NumPy)
- Data Visualization (Plotly, Matplotlib)
- Web Framework (Streamlit)

---

## 💡 Future Enhancements

- [ ] Machine learning-based anomaly detection
- [ ] Real-time streaming data support
- [ ] Multi-currency analysis
- [ ] Advanced temporal pattern detection
- [ ] Regulatory reporting templates
- [ ] Performance optimization for large datasets

---

## 📞 Support

For issues or questions, please refer to the comprehensive Jupyter notebook (`model_decodex.ipynb`) which contains detailed examples and explanations.

