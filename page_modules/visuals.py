import streamlit as st
import plotly.express as px

def render_visuals():
    """Render the enhanced visualizations page"""
    st.header("📈 Advanced Visualizations")
    
    if st.session_state.transactions.empty:
        st.warning("Please upload bank statements first to view visualizations.")
        return
    
    txns = st.session_state.transactions.copy()
    
    # Enhanced visualization tabs with better styling and spacing
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 15px 25px !important;
        margin-right: 12px !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "💰 Spending Analysis", 
        "📊 Category Breakdown", 
        "📅 Time Series", 
        "🔍 Custom Analysis"
    ])
    
    with viz_tab1:
        st.subheader("💰 Spending Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Debit vs Credit comparison with enhanced styling
            type_totals = txns.groupby('Type')['Amount'].sum()
            fig = px.bar(
                x=type_totals.index, 
                y=type_totals.values, 
                title="Total Amount by Transaction Type",
                color=type_totals.index,
                color_discrete_map={'debit': '#ff6b6b', 'credit': '#4ecdc4'},
                height=400
            )
            fig.update_layout(
                xaxis_title="Transaction Type", 
                yaxis_title="Amount (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top transactions with enhanced styling
            top_transactions = txns.nlargest(10, 'Amount')
            fig = px.bar(
                top_transactions, 
                x='Amount', 
                y='Description', 
                orientation='h', 
                title="Top 10 Transactions by Amount",
                height=400
            )
            fig.update_layout(
                yaxis_title="Description", 
                xaxis_title="Amount (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        st.subheader("📊 Category Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category pie chart with enhanced styling
            category_totals = txns.groupby('Category')['Amount'].sum()
            fig = px.pie(
                values=category_totals.values, 
                names=category_totals.index,
                title="Spending Distribution by Category",
                height=400
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category bar chart with enhanced styling
            fig = px.bar(
                x=category_totals.index, 
                y=category_totals.values,
                title="Amount by Category",
                height=400
            )
            fig.update_layout(
                xaxis_title="Category", 
                yaxis_title="Amount (₹)",
                xaxis=dict(tickangle=45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        st.subheader("📅 Time Series Analysis")
        
        # Daily spending trend with enhanced styling
        daily_totals = txns.groupby(txns['Date'].dt.date)['Amount'].sum().reset_index()
        fig = px.line(
            daily_totals, 
            x='Date', 
            y='Amount', 
            title="Daily Transaction Amount Trend",
            height=400
        )
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Amount (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly breakdown
        txns_copy = txns.copy()
        txns_copy['month'] = txns_copy['Date'].dt.to_period('M')
        monthly_totals = txns_copy.groupby('month')['Amount'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=[str(m) for m in monthly_totals.index], 
                y=monthly_totals.values,
                title="Monthly Transaction Totals",
                height=400
            )
            fig.update_layout(
                xaxis_title="Month", 
                yaxis_title="Amount (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                x=[str(m) for m in monthly_totals.index], 
                y=monthly_totals.values,
                title="Monthly Transaction Trend",
                height=400
            )
            fig.update_layout(
                xaxis_title="Month", 
                yaxis_title="Amount (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab4:
        st.subheader("🔍 Custom Analysis")
        
        # Interactive filters for custom analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_categories = st.multiselect(
                "Select Categories", 
                txns['Category'].unique().tolist(),
                default=txns['Category'].unique().tolist()
            )
        
        with col2:
            selected_types = st.multiselect(
                "Select Transaction Types",
                txns['Type'].unique().tolist(),
                default=txns['Type'].unique().tolist()
            )
        
        with col3:
            min_amount = st.number_input("Minimum Amount", value=0.0)
        
        # Apply filters
        custom_txns = txns[
            (txns['Category'].isin(selected_categories)) &
            (txns['Type'].isin(selected_types)) &
            (txns['Amount'] >= min_amount)
        ]
        
        if not custom_txns.empty:
            # Custom visualization with enhanced styling
            fig = px.scatter(
                custom_txns, 
                x='Date', 
                y='Amount', 
                color='Category', 
                size='Amount',
                title="Custom Filtered Transactions",
                hover_data=['Description', 'Type'],
                height=500
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics with enhanced cards
            st.subheader("📊 Custom Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Amount", f"₹{custom_txns['Amount'].sum():,.2f}")
            with col2:
                st.metric("Average Amount", f"₹{custom_txns['Amount'].mean():,.2f}")
            with col3:
                st.metric("Max Amount", f"₹{custom_txns['Amount'].max():,.2f}")
            with col4:
                st.metric("Transaction Count", len(custom_txns))
        else:
            st.info("No transactions match the selected criteria")
