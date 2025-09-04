import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Smart Wallet Transaction Classifier",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Category colors for consistent theming
CATEGORY_COLORS = {
    'Food': '#FF6B6B',
    'Transport': '#4ECDC4', 
    'Subscription': '#45B7D1',
    'Shopping': '#96CEB4',
    'Utilities': '#FECA57'
}

@st.cache_resource
def load_ai_model():
    model_path = "wallet_ai/transaction_classifier.pkl"
    
    try:
        if not os.path.exists(model_path):
            return None, "Model file not found"
        
        with open(model_path, "rb") as f:
            ai_model = pickle.load(f)
        
        return ai_model, "Model loaded successfully"
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def classify_transaction(ai_model, description):
    try:
        clean_text = description.lower().strip()
        features = ai_model['tfidf_vectorizer'].transform([clean_text])
        prediction = ai_model['model'].predict(features)[0]
        confidence = ai_model['model'].predict_proba(features)[0].max()
        category = ai_model['categories'][prediction]
        return category, confidence
    except:
        return "Unknown", 0.0

def initialize_session_state():
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []

def add_transaction(amount, merchant, date, description, category, confidence):
    """Add transaction to session state"""
    transaction = {
        'Date': date,
        'Merchant': merchant,
        'Description': description,
        'Amount': float(amount),
        'Category': category,
        'Confidence': f"{confidence:.1%}",
        'Timestamp': datetime.now()
    }
    st.session_state.transactions.append(transaction)

def create_transaction_summary():
    """Create summary statistics from transactions"""
    if not st.session_state.transactions:
        return None
    
    df = pd.DataFrame(st.session_state.transactions)
    summary = {
        'total_amount': df['Amount'].sum(),
        'total_transactions': len(df),
        'categories_used': df['Category'].nunique(),
        'avg_transaction': df['Amount'].mean()
    }
    return summary, df

def create_pie_chart(df):
    """Create pie chart of spending by category"""
    if df.empty:
        return None
    
    # Group by category and sum amounts
    category_totals = df.groupby('Category')['Amount'].sum().reset_index()
    
    # Create pie chart
    fig = px.pie(
        category_totals, 
        values='Amount', 
        names='Category',
        title="💰 Spending by Category",
        color='Category',
        color_discrete_map=CATEGORY_COLORS,
        hole=0.3  # Donut style
    )
    
    # Customize appearance
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=12
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1, y=0.5),
        font=dict(size=14),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load AI model
    ai_model, load_message = load_ai_model()
    
    if ai_model is None:
        st.error(f"{load_message}")
        st.stop()
    
    # Header
    st.title("Smart Wallet Transaction Classifier")
    st.markdown("### AI-powered transaction categorization with spending analytics")
    st.markdown("---")
    
    # Main layout: Input form + Quick stats
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" New Transaction")
        
        # Transaction input form
        with st.form("transaction_form", clear_on_submit=True):
            # Form inputs
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                amount = st.number_input(
                    "💰 Amount (RM)", 
                    min_value=0.01, 
                    value=10.00, 
                    step=0.01,
                    format="%.2f"
                )
                
                date_input = st.date_input(
                    "📅 Date", 
                    value=date.today()
                )
            
            with form_col2:
                merchant = st.text_input(
                    "🏪 Merchant", 
                    placeholder="e.g., Starbucks",
                    max_chars=50
                )
            
            description = st.text_input(
                "📝 Description", 
                placeholder="e.g., Morning coffee and pastry",
                max_chars=100
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "Classify Transaction", 
                type="primary",
                use_container_width=True
            )
        
        # Process form submission
        if submitted:
            if description.strip():
                # Get AI prediction
                category, confidence = classify_transaction(ai_model, description)
                
                # Add to transaction history
                add_transaction(amount, merchant, date_input, description, category, confidence)
                
                # Show prediction result
                color = CATEGORY_COLORS.get(category, '#999999')
                st.success(f" **{category}** ({confidence:.1%} confidence)")
                
            else:
                st.warning(" Please enter a transaction description")
    
    with col2:
        st.subheader(" Quick Stats")
        
        # Get summary data
        summary_data = create_transaction_summary()
        
        if summary_data:
            summary, df = summary_data
            
            # Display metrics
            st.metric("💰 Total Spent", f"RM{summary['total_amount']:.2f}")
            st.metric("📝 Transactions", f"{summary['total_transactions']}")
            st.metric("📁 Categories", f"{summary['categories_used']}")
            st.metric("📈 Avg Transaction", f"RM{summary['avg_transaction']:.2f}")
            
        else:
            st.info("💡 Add transactions to see statistics")
    
    # Analytics section
    
    if st.session_state.transactions:
        # Create two columns for table and chart
        table_col, chart_col = st.columns([3, 2])
        
        with table_col:
            st.subheader(" Transaction History")
            
            # Create display dataframe
            df = pd.DataFrame(st.session_state.transactions)
            display_df = df[['Date', 'Merchant', 'Description', 'Amount', 'Category', 'Confidence']].copy()
            
            # Format amount column
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"RM{x:.2f}")
            
            # Display table with styling
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=300
            )
            
            # Clear history button
            if st.button(" Clear History", type="secondary"):
                st.session_state.transactions = []
                st.rerun()
        
        with chart_col:
            st.subheader(" Spending Breakdown")
            
            # Create and display pie chart
            _, df = create_transaction_summary()
            fig = create_pie_chart(df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Category breakdown table
            st.markdown("**Category Totals:**")
            category_totals = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
            
            for category, total in category_totals.items():
                color = CATEGORY_COLORS.get(category, '#999999')
                percentage = (total / category_totals.sum()) * 100
                st.markdown(
                    f"<div style='background-color: {color}; padding: 5px; margin: 2px; border-radius: 5px; color: white;'>"
                    f"<strong>{category}</strong>: RM{total:.2f} ({percentage:.1f}%)</div>", 
                    unsafe_allow_html=True
                )
    else:
        # Empty state
        st.info(" **Get Started:** Enter your first transaction above to see analytics and spending patterns!")
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
