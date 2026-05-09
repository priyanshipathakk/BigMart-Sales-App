import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# --- 1. Initialize Master Website Memory ---
# This single variable will hold your uploaded data AND your new single items
if 'master_dataset' not in st.session_state:
    st.session_state['master_dataset'] = pd.DataFrame()

# --- 2. Load the Models ---
@st.cache_resource
def load_models():
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return xgb_model, rf_model

xgb_model, rf_model = load_models()

# --- 3. Page Title ---
st.title("🛒 BigMart Sales Prediction App")
st.write("Upload your base dataset first, then add single items to it, or run bulk predictions.")
st.divider()

# --- 4. Create Navigation Tabs ---
tab1, tab2 = st.tabs(["📌 Single Item Entry", "📁 Base Data Upload & Bulk Predict"])

# ==========================================
# TAB 2: BULK UPLOAD (Do this first so Tab 1 has a base to add to)
# ==========================================
with tab2:
    st.header("1. Upload Base Dataset")
    st.write("Upload your `test.csv` file here. This will serve as the base dataset that new single items are added to.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Load the uploaded file into our Master Dataset
    if uploaded_file is not None:
        if st.button("Load File into Memory", type="primary"):
            st.session_state['master_dataset'] = pd.read_csv(uploaded_file)
            st.success("✅ Base dataset successfully loaded into memory!")
            
    # Bulk Prediction Logic
    st.divider()
    st.subheader("2. Run Bulk Prediction")
    bulk_model_choice = st.radio("Choose Algorithm for Bulk Prediction:", ["XGBoost", "Random Forest"])
    
    if st.button("Generate Bulk Predictions for Master Dataset", key="bulk_btn"):
        if st.session_state['master_dataset'].empty:
            st.warning("Please upload and load a dataset first!")
        else:
            with st.spinner("Processing data and running AI models..."):
                process_data = st.session_state['master_dataset'].copy()
                
                # Basic cleaning for prediction
                if 'Item_Weight' in process_data.columns:
                    process_data['Item_Weight'] = process_data['Item_Weight'].fillna(process_data['Item_Weight'].mean())
                if 'Outlet_Size' in process_data.columns:
                    process_data['Outlet_Size'] = process_data['Outlet_Size'].fillna('Medium') 
                if 'Item_Fat_Content' in process_data.columns:
                    process_data['Item_Fat_Content'] = process_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
                
                mapping_dict = {
                    'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1},
                    'Outlet_Size': {'High': 0, 'Medium': 1, 'Small': 2},
                    'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
                    'Outlet_Type': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
                }
                for col, mapping in mapping_dict.items():
                    if col in process_data.columns:
                        process_data[col] = process_data[col].map(mapping)
                    
                item_type_list = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]
                if 'Item_Type' in process_data.columns:
                    process_data['Item_Type'] = process_data['Item_Type'].apply(lambda x: item_type_list.index(x) if x in item_type_list else 0)
                
                # Assume 0 if identifiers are missing for prediction purposes
                process_data['Item_Identifier'] = 0
                process_data['Outlet_Identifier'] = 0
                
                features = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
                
                # Filter only the features needed for the model
                try:
                    model_input = process_data[features]
                    if bulk_model_choice == "XGBoost":
                        predictions = xgb_model.predict(model_input)
                    else:
                        predictions = rf_model.predict(model_input)
                    
                    # Update the MASTER dataset with predictions
                    st.session_state['master_dataset']['Predicted_Outlet_Sales'] = predictions
                    st.success("✅ Bulk Prediction Complete! Check the dataset below.")
                except Exception as e:
                    st.error(f"Error during prediction. Ensure your uploaded CSV has the correct columns. Error: {e}")

# ==========================================
# TAB 1: SINGLE ITEM & ADD TO MASTER
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Item Details")
        item_identifier = st.text_input("Item ID", "FDA15")
        item_weight = st.number_input("Weight", value=9.3)
        item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
        item_visibility = st.number_input("Visibility", value=0.016)
        item_type = st.selectbox("Item Type", [
            "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", 
            "Frozen Foods", "Fruits and Vegetables", "Hard Drinks", 
            "Health and Hygiene", "Household", "Meat", "Others", 
            "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
        ])
        item_mrp = st.number_input("MRP ($)", value=249.8)

    with col2:
        st.subheader("Outlet Details")
        outlet_identifier = st.text_input("Outlet ID", "OUT049")
        outlet_year = st.number_input("Establishment Year", min_value=1980, max_value=2026, value=1999)
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_location = st.selectbox("Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

    with col3:
        st.subheader("Model Selection")
        model_choice = st.radio("Choose Algorithm:", ["XGBoost", "Random Forest"], key="single_model")

    st.divider()

    if st.button("Predict Sales & Add to Master Dataset", type="primary"):
        
        # 1. Create data for the AI Model
        model_data = pd.DataFrame({
            'Item_Identifier': [0], 'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content], 'Item_Visibility': [item_visibility],
            'Item_Type': [item_type], 'Item_MRP': [item_mrp],
            'Outlet_Identifier': [0], 'Outlet_Establishment_Year': [outlet_year],
            'Outlet_Size': [outlet_size], 'Outlet_Location_Type': [outlet_location],
            'Outlet_Type': [outlet_type]
        })
        
        mapping_dict = {
            'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1},
            'Outlet_Size': {'High': 0, 'Medium': 1, 'Small': 2},
            'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
            'Outlet_Type': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
        }
        for col, mapping in mapping_dict.items():
            model_data[col] = model_data[col].map(mapping)
            
        item_type_list = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]
        model_data['Item_Type'] = model_data['Item_Type'].apply(lambda x: item_type_list.index(x))

        # 2. Run Prediction
        if model_choice == "XGBoost":
            prediction = xgb_model.predict(model_data)[0]
        else:
            prediction = rf_model.predict(model_data)[0]

        st.success(f"Calculation Complete using {model_choice}!")
        st.metric(label="Predicted Item Outlet Sales", value=f"${prediction:.2f}")
        
        # 3. Save the new record as a DataFrame row
        new_record = pd.DataFrame([{
            'Item_Identifier': item_identifier,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location,
            'Outlet_Type': outlet_type,
            'Predicted_Outlet_Sales': round(prediction, 2)
        }])
        
        # 4. Append directly to the MASTER dataset
        if st.session_state['master_dataset'].empty:
            st.session_state['master_dataset'] = new_record
        else:
            st.session_state['master_dataset'] = pd.concat([st.session_state['master_dataset'], new_record], ignore_index=True)


# ==========================================
# ALWAYS DISPLAY THE MASTER DATASET AT THE BOTTOM
# ==========================================
if not st.session_state['master_dataset'].empty:
    st.divider()
    st.subheader("📊 Your Active Master Dataset")
    st.write("This contains your uploaded data AND any single items you have added. It updates in real-time.")
    
    # Show the dataframe
    st.dataframe(st.session_state['master_dataset'])
    
    # Provide a unified download button
    csv_data = st.session_state['master_dataset'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Updated Dataset",
        data=csv_data,
        file_name='Updated_BigMart_Dataset.csv',
        mime='text/csv',
    )
