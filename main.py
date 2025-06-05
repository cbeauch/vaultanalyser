# source .venv/bin/activate
# streamlit run main.py

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from hyperliquid.vaults import fetch_vault_details, fetch_vaults_data
from metrics.drawdown import (
    calculate_max_drawdown_on_accountValue,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from metrics.enhanced_metrics import calculate_all_enhanced_metrics_with_timestamps

# Page config
st.set_page_config(page_title="Xin Vault Analyzer", page_icon="📊", layout="wide")

# Header with optional banner image
import os

# Check if banner image exists
banner_path = "assets/banner.png"  # You can change this to banner.jpg, banner.svg, etc.
alternative_banner_path = "assets/logo.png"  # Alternative image name

if os.path.exists(banner_path):
    # Display banner image
    st.image(banner_path, use_container_width=True)
    # Smaller title below banner
    st.markdown("<h2 style='text-align: center; margin-top: -10px;'>📊 Xin Vault Analyzer</h2>", unsafe_allow_html=True)
elif os.path.exists(alternative_banner_path):
    # Display alternative image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(alternative_banner_path, use_container_width=True)
    st.markdown("<h2 style='text-align: center; margin-top: 10px;'>📊 Xin Vault Analyzer</h2>", unsafe_allow_html=True)
else:
    # Default title without image
    st.title("📊 Xin Vault Analyzer")

# Update time display
try:
    with open("./cache/vaults_cache.json", "r") as f:
        cache = json.load(f)
        last_update = datetime.fromisoformat(cache["last_update"])
        st.caption(f"🔄 Last update: {last_update.strftime('%Y-%m-%d %H:%M')} UTC")
except (FileNotFoundError, KeyError, ValueError):
    st.warning("⚠️ Cache not found. Data will be fetched fresh.")
st.markdown("---")  # Add a separator line


def check_date_file_exists(directory="./cache"):
    """
    Checks if the `date.json` file exists in the specified directory.

    :param directory: Directory where the file is expected to be located (default: /cache).
    :return: True if the file exists, otherwise False.
    """
    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Check existence
    return os.path.exists(file_path)


def create_date_file(directory="./cache"):
    """
    Creates a `date.json` file in the specified directory with the current date.

    :param directory: Directory where the file will be created (default: /cache).
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Content to write
    current_date = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # Write to the file
    with open(file_path, "w") as file:
        json.dump(current_date, file, indent=4)
    print(f"`date.json` file created in {file_path}")


def read_date_file(directory="./cache"):
    """
    Reads and returns the date saved in the `date.json` file from the specified directory.

    :param directory: Directory where the file is located (default: /cache).
    :return: The date as a string or None if the file doesn't exist.
    """
    # Full file path
    file_path = os.path.join(directory, "date.json")

    # Check if the file exists
    if not os.path.exists(file_path):
        print("`date.json` file not found.")
        return None

    # Read the file
    with open(file_path, "r") as file:
        data = json.load(file)
        return data.get("date")


# Layout for 3 columns


def slider_with_label_and_toggle(label, col, min_value, max_value, default_value, step, key):
    """Create a slider with a custom centered title and toggle."""
    col.markdown(f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True)
    
    # Add toggle switch
    toggle_enabled = col.toggle("Enable Filter", key=f"toggle_{key}", value=False)
    
    if not min_value < max_value:
        col.markdown(
            f"<p style='text-align: center;'>No choice available ({min_value} for all)</p>", unsafe_allow_html=True
        )
        return None, toggle_enabled

    if default_value < min_value:
        default_value = min_value

    if default_value > max_value:
        default_value = max_value

    # Show slider only if toggle is enabled, otherwise show disabled slider
    if toggle_enabled:
        value = col.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=step,
            label_visibility="hidden",
            key=key,
        )
    else:
        # Show disabled slider with default value
        value = col.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=step,
            label_visibility="hidden",
            key=key,
            disabled=True,
        )
    
    return value, toggle_enabled


def calculate_average_daily_gain(rebuilded_pnl, days_since):
    """
    Calculates the average daily gain percentage.

    :param rebuilded_pnl: List of cumulative PnL values ($).
    :param days_since: Number of days (int).
    :return: Average daily gain percentage (float).
    """
    if len(rebuilded_pnl) < 2 or days_since <= 0:
        return 0  # Not enough data to calculate

    initial_value = rebuilded_pnl[0]
    final_value = rebuilded_pnl[-1]

    # Avoid division by zero
    if initial_value == 0:
        return 0  # Cannot calculate if the initial value is 0

    average_daily_gain_pct = ((final_value - initial_value) / (initial_value * days_since)) * 100
    return average_daily_gain_pct


def calculate_total_gain_percentage(rebuilded_pnl):
    """
    Calculates the total percentage change since the beginning.

    :param rebuilded_pnl: List of cumulative PnL values ($).
    :return: Total percentage change (float).
    """
    if len(rebuilded_pnl) < 2:
        return 0  # Not enough data to calculate

    initial_value = rebuilded_pnl[0]
    final_value = rebuilded_pnl[-1]

    # Avoid division by zero
    if initial_value == 0:
        return 0  # Cannot calculate if the initial value is 0

    total_gain_pct = ((final_value - initial_value) / initial_value) * 100
    return total_gain_pct


limit_vault = False


DATAFRAME_CACHE_FILE = "./cache/dataframe.pkl"

cache_used = False
try:
    final_df = pd.read_pickle(DATAFRAME_CACHE_FILE)
    cache_used = True
except (FileNotFoundError, KeyError, ValueError):
    pass

if not cache_used:

    # Get vaults data (will use cache if valid)
    vaults = fetch_vaults_data()

    # Limit to the first 50 vaults if needed
    if limit_vault:
        vaults = vaults[:50]

    # Process vault details from cache
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing vault details...")
    total_steps = len(vaults)
    indicators = []
    progress_i = 1

    for vault in vaults:
        progress_bar.progress(progress_i / total_steps)
        progress_i = progress_i + 1
        status_text.text(f"Processing vault details ({progress_i}/{total_steps})...")

        details = fetch_vault_details(vault["Leader"], vault["Vault"])

        nb_followers = 0
        if details and "followers" in details:
            nb_followers = sum(1 for f in details["followers"] if float(f["vaultEquity"]) >= 0.01)

        if details and "portfolio" in details:
            if details["portfolio"][3][0] == "allTime":
                data_source_pnlHistory = details["portfolio"][3][1].get("pnlHistory", [])
                data_source_accountValueHistory = details["portfolio"][3][1].get("accountValueHistory", [])
                rebuilded_pnl = []

                balance = start_balance_amount = 1000000
                nb_rekt = 0
                last_rekt_idx = -10

                # Recalculate the balance without considering deposit movements
                for idx, value in enumerate(data_source_pnlHistory):
                    if idx == 0:
                        continue

                    # Capital at time T
                    final_capital = float(data_source_accountValueHistory[idx][1])
                    # Cumulative PnL at time T
                    final_cumulated_pnl = float(data_source_pnlHistory[idx][1])
                    # Cumulative PnL at time T -1
                    previous_cumulated_pnl = float(data_source_pnlHistory[idx - 1][1]) if idx > 0 else 0
                    # Non-cumulative PnL at time T
                    final_pnl = final_cumulated_pnl - previous_cumulated_pnl
                    # Capital before the gain/loss
                    initial_capital = final_capital - final_pnl

                    if initial_capital <= 0:
                        if last_rekt_idx + 1 != idx:
                            rebuilded_pnl = []
                            balance = start_balance_amount
                            nb_rekt = nb_rekt + 1
                        last_rekt_idx = idx
                        continue
                    # Gain/loss ratio
                    ratio = final_capital / initial_capital

                    # Verify timestamp consistency
                    if data_source_pnlHistory[idx][0] != data_source_accountValueHistory[idx][0]:
                        print("Just to check, normally not happening")
                        exit()

                    # Update the simulated balance
                    balance = balance * ratio

                    rebuilded_pnl.append(balance)

                # Calculate existing metrics
                existing_metrics = {
                    "Max DD %": calculate_max_drawdown_on_accountValue(rebuilded_pnl),
                    "Rekt": nb_rekt,
                    "Act. Followers": nb_followers,
                    "Sharpe Ratio": calculate_sharpe_ratio(rebuilded_pnl),
                    "Sortino Ratio": calculate_sortino_ratio(rebuilded_pnl),
                    "Av. Daily Gain %": calculate_average_daily_gain(rebuilded_pnl, vault["Days Since"]),
                    "Gain %": calculate_total_gain_percentage(rebuilded_pnl),
                }
                
                # Calculate enhanced metrics using timestamp data
                enhanced_metrics = calculate_all_enhanced_metrics_with_timestamps(
                    data_source_pnlHistory,
                    data_source_accountValueHistory,
                    rebuilded_pnl,
                    vault["Days Since"]
                )
                
                # Combine all metrics
                metrics = {**existing_metrics, **enhanced_metrics}
                
                # Unpacks the metrics dictionary
                indicator_row = {"Name": vault["Name"], **metrics}
                indicators.append(indicator_row)

    progress_bar.empty()
    status_text.empty()

    st.toast("Vault details OK!", icon="✅")

    # Step 4: Merge indicators with the main table
    indicators_df = pd.DataFrame(indicators)
    vaults_df = pd.DataFrame(vaults)
    del vaults_df["Leader"]
    final_df = vaults_df.merge(indicators_df, on="Name", how="left")

    final_df.to_pickle(DATAFRAME_CACHE_FILE)


# Filters
st.subheader(f"Vaults available ({len(final_df)})")
filtered_df = final_df


# Filter by 'Name' (free text filter with toggle)
st.markdown("<h3 style='text-align: center;'>Filter by Name</h3>", unsafe_allow_html=True)
name_filter_enabled = st.toggle("Enable Name Filter", key="toggle_name_filter", value=False)

if name_filter_enabled:
    name_filter = st.text_input(
        "Name Filter", "", placeholder="Enter names separated by ',' to filter (e.g., toto,tata)...", key="name_filter"
    )
else:
    name_filter = st.text_input(
        "Name Filter", "", placeholder="Enter names separated by ',' to filter (e.g., toto,tata)...", key="name_filter", disabled=True
    )

# Apply the name filter only if enabled and not empty
if name_filter_enabled and name_filter.strip():
    name_list = [name.strip() for name in name_filter.split(",")]  # List of names to search for
    pattern = "|".join(name_list)  # Create a regex pattern with logical "or"
    filtered_df = filtered_df[filtered_df["Name"].str.contains(pattern, case=False, na=False, regex=True)]

# Organize sliders into rows of 3
sliders = [
    {"label": "Min Sharpe Ratio", "column": "Sharpe Ratio", "max": False, "default": 0.4, "step": 0.1},
    {"label": "Min Sortino Ratio", "column": "Sortino Ratio", "max": False, "default": 0.5, "step": 0.1},
    {"label": "Max Rekt accepted", "column": "Rekt", "max": True, "default": 0, "step": 1},
    {"label": "Max DD % accepted", "column": "Max DD %", "max": True, "default": 15, "step": 1},
    {"label": "Min Days Since accepted", "column": "Days Since", "max": False, "default": 100, "step": 1},
    {"label": "Min TVL accepted", "column": "Total Value Locked", "max": False, "default": 0, "step": 1},
    {"label": "Min APR accepted", "column": "APR %", "max": False, "default": 0, "step": 1},
    {"label": "Min Followers", "column": "Act. Followers", "max": False, "default": 0, "step": 1},
    {"label": "Min Win Rate %", "column": "Win Rate %", "max": False, "default": 40, "step": 5},
    {"label": "Min Profit Factor", "column": "Profit Factor", "max": False, "default": 1.0, "step": 0.1},
    {"label": "Max Volatility %", "column": "Volatility %", "max": True, "default": 100, "step": 5},
    {"label": "Min Calmar Ratio", "column": "Calmar Ratio", "max": False, "default": 0, "step": 0.1},
]

for i in range(0, len(sliders), 3):
    cols = st.columns(3)
    for slider, col in zip(sliders[i : i + 3], cols):
        column = slider["column"]
        if column in filtered_df.columns:  # Check if column exists
            # Clean the data - remove NaN, infinity values
            clean_data = filtered_df[column].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_data) > 0:
                # Get safe min/max values
                col_min = float(clean_data.min())
                col_max = float(clean_data.max())
                
                # Ensure min < max and handle edge cases
                if col_min >= col_max:
                    col_max = col_min + 1
                
                # Ensure default value is within range
                safe_default = max(col_min, min(col_max, float(slider["default"])))
                
                try:
                    value, toggle_enabled = slider_with_label_and_toggle(
                        slider["label"],
                        col,
                        min_value=col_min,
                        max_value=col_max,
                        default_value=safe_default,
                        step=float(slider["step"]),
                        key=f"slider_{column}",
                    )
                    # Only apply filter if toggle is enabled and value is not None
                    if toggle_enabled and value is not None:
                        if slider["max"]:
                            filtered_df = filtered_df[filtered_df[column] <= value]
                        else:
                            filtered_df = filtered_df[filtered_df[column] >= value]
                except Exception as e:
                    # If slider still fails, show error message
                    col.error(f"Error with {column}: {str(e)}")
            else:
                # No valid data for this column
                col.warning(f"No valid data for {column}")

# Multi-Column Sorting Section
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🔄 Multi-Column Sorting</h3>", unsafe_allow_html=True)

# Get numeric columns for sorting (exclude non-numeric columns)
numeric_columns = []
for col in filtered_df.columns:
    if col not in ["Name", "Vault", "Link"] and pd.api.types.is_numeric_dtype(filtered_df[col]):
        numeric_columns.append(col)

# Add "None" option for unused sort levels
sort_options = ["None"] + numeric_columns

# Create sorting controls in 3 columns
sort_col1, sort_col2, sort_col3 = st.columns(3)

# Set default primary sort to APR if it exists and no sort has been selected yet
default_primary_index = 0
if "APR" in sort_options and "primary_sort" not in st.session_state:
    default_primary_index = sort_options.index("APR")

with sort_col1:
    st.markdown("<h4 style='text-align: center;'>Primary Sort</h4>", unsafe_allow_html=True)
    primary_sort = st.selectbox("Primary Sort Column", options=sort_options, index=default_primary_index, key="primary_sort")
    if primary_sort != "None":
        primary_order = st.radio("Primary Order", ["Descending (High to Low)", "Ascending (Low to High)"], key="primary_order")

with sort_col2:
    st.markdown("<h4 style='text-align: center;'>Secondary Sort</h4>", unsafe_allow_html=True)
    secondary_sort = st.selectbox("Secondary Sort Column", options=sort_options, index=0, key="secondary_sort")
    if secondary_sort != "None":
        secondary_order = st.radio("Secondary Order", ["Descending (High to Low)", "Ascending (Low to High)"], key="secondary_order")

with sort_col3:
    st.markdown("<h4 style='text-align: center;'>Tertiary Sort</h4>", unsafe_allow_html=True)
    tertiary_sort = st.selectbox("Tertiary Sort Column", options=sort_options, index=0, key="tertiary_sort")
    if tertiary_sort != "None":
        tertiary_order = st.radio("Tertiary Order", ["Descending (High to Low)", "Ascending (Low to High)"], key="tertiary_order")

# Apply multi-column sorting
sort_columns = []
sort_ascending = []

if tertiary_sort != "None":
    sort_columns.append(tertiary_sort)
    sort_ascending.append(tertiary_order == "Ascending (Low to High)")

if secondary_sort != "None":
    sort_columns.append(secondary_sort)
    sort_ascending.append(secondary_order == "Ascending (Low to High)")

if primary_sort != "None":
    sort_columns.append(primary_sort)
    sort_ascending.append(primary_order == "Ascending (Low to High)")

# Apply sorting if any sort columns are selected
if sort_columns:
    filtered_df = filtered_df.sort_values(by=sort_columns, ascending=sort_ascending)

# Pagination Section
st.markdown("---")

# Add a column with clickable links
filtered_df["Link"] = filtered_df["Vault"].apply(lambda vault: f"https://app.hyperliquid.xyz/vaults/{vault}")

# Reset index for continuous ranking
filtered_df = filtered_df.reset_index(drop=True)

# Pagination settings
ITEMS_PER_PAGE = 50
total_items = len(filtered_df)
total_pages = max(1, (total_items - 1) // ITEMS_PER_PAGE + 1)

# Page selection
st.title(f"📊 Vaults Results ({total_items} vaults found)")

if total_items > 0:
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col2:
        if st.button("⬅️ Previous", disabled=(st.session_state.get('current_page', 1) <= 1)):
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            st.session_state.current_page = max(1, st.session_state.current_page - 1)
            st.rerun()
    
    with col3:
        # Initialize current page if not exists
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
            
        # Page selector
        current_page = st.selectbox(
            f"Page (1-{total_pages})",
            options=list(range(1, total_pages + 1)),
            index=st.session_state.current_page - 1,
            key="page_selector"
        )
        
        # Update session state if page changed
        if current_page != st.session_state.current_page:
            st.session_state.current_page = current_page
            st.rerun()
    
    with col4:
        if st.button("Next ➡️", disabled=(st.session_state.current_page >= total_pages)):
            st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
            st.rerun()
    
    # Calculate pagination bounds
    start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    
    # Display page info
    st.caption(f"Showing vaults {start_idx + 1}-{end_idx} of {total_items} (Page {st.session_state.current_page} of {total_pages})")
    
    # Get page data
    page_df = filtered_df.iloc[start_idx:end_idx].copy()
    
    # Add ranking column
    page_df.insert(0, "Rank", range(start_idx + 1, end_idx + 1))
    
    # Display the paginated table
    st.dataframe(
        page_df,
        use_container_width=True,
        height=min(600, (len(page_df) * 35) + 50),  # Cap height to prevent very tall tables
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Link": st.column_config.LinkColumn(
                "Vault Link",
                display_text="View Vault",
                width="medium"
            )
        },
    )
    
    # Bottom pagination controls (duplicate for convenience)
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col2:
        if st.button("⬅️ Previous ", disabled=(st.session_state.current_page <= 1), key="prev_bottom"):
            st.session_state.current_page = max(1, st.session_state.current_page - 1)
            st.rerun()
    
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 10px;'><strong>Page {st.session_state.current_page} of {total_pages}</strong></div>", unsafe_allow_html=True)
    
    with col4:
        if st.button("Next ➡️ ", disabled=(st.session_state.current_page >= total_pages), key="next_bottom"):
            st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
            st.rerun()

else:
    st.info("🔍 No vaults match your current filters. Try adjusting the filter criteria.")
