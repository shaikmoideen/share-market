import streamlit as st
import pandas as pd
import os

from config import PORTFOLIO, NSE_STOCK_MASTER, DATA_FOLDER
from utils.stock_search import search_stock

st.set_page_config(
    page_title="Manage Portfolio",
    layout="wide"
)

st.title("⚙️ Manage Portfolio")

os.makedirs(DATA_FOLDER, exist_ok=True)

# ---------------------------------
# Load Existing Portfolio
# ---------------------------------

if os.path.exists(PORTFOLIO):
    df = pd.read_csv(PORTFOLIO)
else:
    df = pd.DataFrame(columns=[
        "Stock",
        "Buy Price",
        "Quantity"
    ])

# ---------------------------------
# Load NSE Master for Validation
# ---------------------------------

if os.path.exists(NSE_STOCK_MASTER):
    nse_df = pd.read_csv(NSE_STOCK_MASTER)

    valid_symbols = (
        nse_df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda x: x + ".NS")
        .tolist()
    )
else:
    valid_symbols = []

st.markdown("---")

# ---------------------------------
# Add New Stock
# ---------------------------------

st.subheader("➕ Add New Stock")

col1, col2, col3 = st.columns(3)

with col1:
    search_query = st.text_input(
        "Search Stock",
        placeholder="Type stock like TCS, Reliance, Infosys...",
        key="portfolio_stock_search"
    )

    stock_name = None

    if search_query:
        suggestions = search_stock(search_query)

        if suggestions:
            selected_option = st.selectbox(
                "Select Matching Stock",
                [f"{s['name']} ({s['symbol']})" for s in suggestions],
                key="portfolio_stock_select"
            )

            selected_data = next(
                s for s in suggestions
                if f"{s['name']} ({s['symbol']})" == selected_option
            )

            stock_name = selected_data["symbol"]
        else:
            st.warning("No matching stocks found.")

    if search_query and not stock_name:
        st.info("Please select a stock from suggestions.")

with col2:
    buy_price = st.number_input(
        "Buy Price",
        min_value=0.0,
        step=1.0
    )

with col3:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        step=1
    )

stock_name = locals().get("stock_name", None)

if st.button("💾 Save Stock"):
    if buy_price <= 0:
        st.warning("Buy Price must be greater than 0.")
        st.stop()
        
    if not stock_name:
        st.warning("Please enter stock symbol.")
        st.stop()

    if stock_name in df["Stock"].values:
        st.warning("Stock already exists in portfolio.")
        st.stop()
    
    # Normalize input
    # if not stock_name.endswith(".NS"):
    #     stock_name = stock_name + ".NS"

    # Debug (optional)
    # st.write("Checking:", stock_name)

    # if valid_symbols and stock_name not in valid_symbols:
    #     st.warning(
    #         f"Invalid stock symbol: {stock_name}\n\nExample: TCS, INFY, RELIANCE"
    #     )
    #     st.stop()
    
    st.write("Sample valid symbols:", valid_symbols[:5])

    new_row = pd.DataFrame([{
        "Stock": stock_name,
        "Buy Price": buy_price,
        "Quantity": quantity
    }])

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        PORTFOLIO,
        index=False
    )

    st.success("Stock added successfully!")
    st.rerun()

st.markdown("---")

# ---------------------------------
# Existing Portfolio Table
# ---------------------------------

st.subheader("👜 Your Portfolio Stocks")

if not df.empty:

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    # ---------------------------------
    # Edit Existing Stock
    # ---------------------------------

    st.subheader("✏️ Edit Existing Stock")

    selected_stock = st.selectbox(
        "Select Stock to Edit",
        df["Stock"].tolist()
    )

    selected_row = df[
        df["Stock"] == selected_stock
    ].iloc[0]

    new_buy_price = st.number_input(
        "Update Buy Price",
        value=float(selected_row["Buy Price"]),
        step=1.0,
        key="edit_price"
    )

    new_quantity = st.number_input(
        "Update Quantity",
        value=int(selected_row["Quantity"]),
        step=1,
        key="edit_quantity"
    )

    if st.button("Update Stock"):
        
        if new_buy_price <= 0:
            st.warning(
                "Buy Price must be greater than 0."
            )
            st.stop()

        if new_quantity <= 0:
            st.warning(
                "Quantity must be greater than 0."
            )
            st.stop()
            
        df.loc[
            df["Stock"] == selected_stock,
            "Buy Price"
        ] = new_buy_price
        
        df.loc[
            df["Stock"] == selected_stock,
            "Quantity"
        ] = new_quantity

        df.to_csv(
            PORTFOLIO,
            index=False
        )

        st.success("Stock updated successfully!")
        st.rerun()

    st.markdown("---")

    # ---------------------------------
    # Delete Stock
    # ---------------------------------

    st.subheader("🗑 Delete Stock")

    delete_stock = st.selectbox(
        "Select Stock to Delete",
        df["Stock"].tolist(),
        key="delete_stock"
    )

    if st.button("Delete Selected Stock"):

        df = df[
            df["Stock"] != delete_stock
        ]

        df.to_csv(
            PORTFOLIO,
            index=False
        )

        st.success(
            f"{delete_stock} deleted successfully!"
        )
        st.rerun()

else:
    st.info("No stocks added yet.")