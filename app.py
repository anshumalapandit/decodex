import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from analysis import load_trades_file, prepare_trades_dataframe
from metrics import compute_counterparty_concentration, compute_reciprocity, build_trade_graph, detect_triangular_loops
from risk_model import compute_risk_score


st.set_page_config(layout="wide", page_title="Regulatory Surveillance Analytics")
st.title("Regulatory Market Surveillance Analytics")
st.markdown("**Independent Surveillance Analytics Advisory** | Baseline Validation Phase")

tabs = st.tabs([
    "Overview",
    "Execution Layer & Network",
    "Intent Layer & Order Patterns",
    "Risk Assessment",
    "Suspicious Entity Analysis",
    "Prevention & Governance"
])


# Sidebar file selection
st.sidebar.header("Data Selection")
trade_folder = "Trades"
order_folder = "Orders"

trade_files = [f for f in os.listdir(trade_folder) if f.endswith(".xls")]
order_files = [f for f in os.listdir(order_folder) if f.endswith(".xls")]

selected_trade = st.sidebar.selectbox("Select Trades File", trade_files)
selected_order = st.sidebar.selectbox("Select Orders File", order_files)

trade_path = os.path.join(trade_folder, selected_trade)
order_path = os.path.join(order_folder, selected_order)

df_trades = prepare_trades_dataframe(load_trades_file(trade_path))
df_orders = load_trades_file(order_path)


# ================= OVERVIEW =================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", len(df_trades))
    col2.metric("Unique Buyers", df_trades["buyer"].nunique())
    col3.metric("Unique Sellers", df_trades["seller"].nunique())
    col4.metric("Total Scrips", df_trades["scrip"].nunique() if "scrip" in df_trades.columns else "N/A")

    st.subheader("Trading Activity Over Time")
    df_trades_sorted = df_trades.sort_values("time")
    fig_volume = px.area(df_trades_sorted, x="time", y="quantity", title="Trade Quantity Trend")
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Price Movement Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig_price_hist = px.histogram(df_trades, x="price", nbins=40, title="Price Distribution")
        st.plotly_chart(fig_price_hist, use_container_width=True)
    with col2:
        fig_quantity_hist = px.histogram(df_trades, x="quantity", nbins=40, title="Quantity Distribution")
        st.plotly_chart(fig_quantity_hist, use_container_width=True)


# ================= EXECUTION LAYER & NETWORK =================
with tabs[1]:
    reciprocity_ratio = compute_reciprocity(df_trades)
    G = build_trade_graph(df_trades)
    triangles = detect_triangular_loops(G)

    col1, col2, col3 = st.columns(3)
    col1.metric("Reciprocity Ratio", round(reciprocity_ratio, 3))
    col2.metric("Triangle Loops (Circular Trading)", len(triangles))
    col3.metric("Network Density", round(nx.density(G), 3))

    st.subheader("Price Trend")
    fig_price = px.line(df_trades_sorted, x="time", y="price", title="Price Timeline")
    st.plotly_chart(fig_price, use_container_width=True)

    # Trade Network Graph
    st.subheader("Client Trading Network")
    st.info(f"Shows {G.number_of_nodes()} nodes (clients) with {G.number_of_edges()} connections")
    
    if G.number_of_nodes() < 50:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        edges = G.edges()
        
        edge_x = []
        edge_y = []
        for edge in edges:
            edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo='text',
            marker=dict(size=8, color='#FF6B6B')
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title="Client Trade Network", showlegend=False, hovermode='closest', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    # Triangular Loops Detection
    if triangles:
        st.subheader("Detected Circular Trading Clusters")
        st.warning(f"Found {len(triangles)} triangular loops indicating potential circular trading")
        for i, triangle in enumerate(list(triangles)[:5]):
            st.write(f"**Loop {i+1}:** {' → '.join(triangle)} → {triangle[0]}")

    # Trade Counterparty Matrix
    st.subheader("Buyer-Seller Interaction Heatmap")
    pivot_trades = pd.crosstab(df_trades["buyer"], df_trades["seller"], values=df_trades["quantity"], aggfunc="sum").fillna(0)
    
    if pivot_trades.shape[0] > 0 and pivot_trades.shape[1] > 0:
        # Show top counterparty interactions if matrix is too large
        if pivot_trades.shape[0] > 25 or pivot_trades.shape[1] > 25:
            # Get top buyers and sellers
            top_buyers = pivot_trades.sum(axis=1).nlargest(20).index
            top_sellers = pivot_trades.sum(axis=0).nlargest(20).index
            pivot_trades = pivot_trades.loc[top_buyers, top_sellers]
            st.info(f"Showing top 20 buyers × top 20 sellers (out of {pivot_trades.shape[0]} × {pivot_trades.shape[1]})")
        
        fig_heatmap = px.imshow(pivot_trades, title="Trade Quantity Heatmap (Buyer vs Seller)", aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Not enough data for heatmap")


# ================= INTENT LAYER & ORDER PATTERNS =================
with tabs[2]:
    df_orders.columns = df_orders.columns.str.strip().str.upper()

    st.subheader("Order Activity by Client")
    if "CLIENT CODE" in df_orders.columns:
        order_agg = df_orders.groupby("CLIENT CODE").agg({
            "ORDER_NUMBER": "count"
        }).reset_index()
        order_agg.columns = ["CLIENT CODE", "ORDERS"]
        order_agg = order_agg.sort_values("ORDERS", ascending=False)
        
        st.dataframe(order_agg.head(20))

        col1, col2 = st.columns(2)
        with col1:
            fig_orders = px.bar(order_agg.head(15), x="CLIENT CODE", y="ORDERS", title="Top 15 Clients by Order Count")
            st.plotly_chart(fig_orders, use_container_width=True)
        with col2:
            if "ORDER QUANTITY" in df_orders.columns:
                qty_agg = df_orders.groupby("CLIENT CODE")["ORDER QUANTITY"].sum().reset_index()
                qty_agg = qty_agg.sort_values("ORDER QUANTITY", ascending=False).head(15)
                fig_qty = px.bar(qty_agg, x="CLIENT CODE", y="ORDER QUANTITY", title="Top 15 Clients by Total Quantity")
                st.plotly_chart(fig_qty, use_container_width=True)

    # Order Type Distribution
    st.subheader("Order Type Distribution")
    if "ORDER TYPE" in df_orders.columns:
        order_type_counts = df_orders["ORDER TYPE"].value_counts().reset_index()
        order_type_counts.columns = ["ORDER TYPE", "COUNT"]
        fig_order_type = px.pie(order_type_counts, names="ORDER TYPE", values="COUNT", title="Order Types")
        st.plotly_chart(fig_order_type, use_container_width=True)

    # Buy/Sell Ratio
    st.subheader("Buy vs Sell Orders")
    if "BUY/SELL FLAG" in df_orders.columns:
        buy_sell_counts = df_orders["BUY/SELL FLAG"].value_counts().reset_index()
        buy_sell_counts.columns = ["DIRECTION", "COUNT"]
        fig_buy_sell = px.bar(buy_sell_counts, x="DIRECTION", y="COUNT", title="Buy vs Sell Orders")
        st.plotly_chart(fig_buy_sell, use_container_width=True)


# ================= RISK ASSESSMENT =================
with tabs[3]:
    concentration_df = compute_counterparty_concentration(df_trades)
    risk_df = compute_risk_score(concentration_df, reciprocity_ratio, len(triangles))

    st.subheader("Risk Scoring Results (Top 20 Entities)")
    risk_display = risk_df.head(20).copy()
    st.dataframe(risk_display.style.highlight_max(color='#FF6B6B'))

    # Risk Score Distribution
    st.subheader("Risk Score Distribution")
    fig_risk_dist = px.histogram(risk_df, x="risk_score", nbins=20, title="Risk Score Distribution")
    st.plotly_chart(fig_risk_dist, use_container_width=True)

    # High Risk Entities
    st.subheader("High Risk Entities")
    high_risk = risk_df[risk_df["risk_score"] > risk_df["risk_score"].quantile(0.75)].copy()
    fig_high_risk = px.bar(high_risk.head(15), x="client", y="risk_score", color="risk_score", 
                           title="Top 15 High Risk Entities", color_continuous_scale="Reds")
    st.plotly_chart(fig_high_risk, use_container_width=True)


# ================= SUSPICIOUS ENTITY ANALYSIS =================
with tabs[4]:
    st.subheader("Flagged Suspicious Entities")
    
    # Load suspicious orders
    suspicious_folder = "suspicious_orders"
    suspicious_files = [f for f in os.listdir(suspicious_folder) if f.endswith(".csv")]
    
    if suspicious_files:
        suspicious_data = []
        for file in suspicious_files:
            df_susp = pd.read_csv(os.path.join(suspicious_folder, file))
            suspicious_data.append(df_susp)
        
        if suspicious_data:
            df_suspicious = pd.concat(suspicious_data, ignore_index=True)
            st.write(f"**Total Flagged Entities:** {len(df_suspicious)}")
            
            st.subheader("Suspicious Order Details")
            st.dataframe(df_suspicious)

            st.subheader("Suspicious Entity Risk Profile")
            if "big_ord_client_id" in df_suspicious.columns:
                suspicious_clients = df_suspicious["big_ord_client_id"].unique()
                matching_risks = risk_df[risk_df["client"].isin(suspicious_clients)]
                st.dataframe(matching_risks)

    # Correlation Analysis
    st.subheader("Pattern Correlation Analysis")
    if len(df_trades) > 0:
        correlation_cols = df_trades.select_dtypes(include=[np.number]).columns
        if len(correlation_cols) > 1:
            corr_matrix = df_trades[correlation_cols].corr()
            fig_corr = px.imshow(corr_matrix, title="Feature Correlation", color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)


# ================= PREVENTION & GOVERNANCE =================
with tabs[5]:
    st.subheader("Detection Rules & Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Circular Trading Pattern")
        st.info(f"✓ Triangular loops detected: {len(triangles)}\n\n" +
                "**Mechanism:** A → B → C → A\n\n" +
                "**Evidence:** Closed trading loops")
    
    with col2:
        st.write("### Synchronization Indicator")
        st.info(f"✓ Reciprocity ratio: {round(reciprocity_ratio, 3)}\n\n" +
                "**Pattern:** Sustained buy-sell pairs\n\n" +
                "**Threshold:** >0.3 = coordination signal")

    st.subheader("Recommended Actions")
    high_risk_entities = risk_df[risk_df["risk_score"] > risk_df["risk_score"].quantile(0.80)]["client"].tolist()
    
    st.warning(f"**{len(high_risk_entities)} entities exceed 80th percentile risk threshold**")
    
    # Escalation priority
    st.subheader("Escalation Priority (Max 12 Entities)")
    num_escalate = min(12, len(risk_df))
    escalation_df = risk_df.nlargest(num_escalate, "risk_score")[["client", "risk_score", "counterparty_concentration"]].copy()
    escalation_df.insert(0, "rank", range(1, len(escalation_df) + 1))
    escalation_df["status"] = "HIGH PRIORITY"
    
    st.dataframe(escalation_df, use_container_width=True)

    st.success(f"{num_escalate} entities recommended for escalation review")
    st.subheader("Regulatory Prevention Recommendations")

    st.markdown("""
    - Monitor persistent reciprocal clusters.
    - Flag high counterparty concentration > threshold.
    - Investigate shared infrastructure across client codes.
    - Apply enhanced surveillance during price spikes.
    - Use persistence-based escalation, not volume-based triggers.
    """)