import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# --- 1. Initialize Website Memory ---
# This creates a blank dataset that remembers data as long as the website is open
if 'my_dataset' not in st.session_state:
    st.session_state['my_dataset'] = pd.DataFrame()

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
st.write("Predict sales for a single item, build a custom dataset, or upload bulk data.")
st.divider()

# --- 4. Create Navigation Tabs ---
tab1, tab2 = st.tabs(["📌 Single Item & Build Dataset", "📁 Bulk Data Upload"])

# ==========================================
# TAB 1: SINGLE ITEM & BUILD DATASET
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
        model_choice = st.radio("Choose Algorithm:", ["XGBoost", "Random Forest"])

    st.divider()

    # The New "Predict and Save" Button
    if st.button("Predict Sales & Add to Dataset", type="primary"):
        
        # 1. Create the data for the AI Model (Needs to be converted to numbers)
        model_data = pd.DataFrame({
            'Item_Identifier': [0], 'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content], 'Item_Visibility': [item_visibility],
            'Item_Type': [item_type], 'Item_MRP': [item_mrp],
            'Outlet_Identifier': [0], 'Outlet_Establishment_Year': [outlet_year],
            'Outlet_Size': [outlet_size], 'Outlet_Location_Type': [outlet_location],
            'Outlet_Type': [outlet_type]
        })
        
        # Mapping to numbers for the AI
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

        # 2. Run the Prediction
        if model_choice == "XGBoost":
            prediction = xgb_model.predict(model_data)[0]
        else:
            prediction = rf_model.predict(model_data)[0]

        st.success(f"Calculation Complete using {model_choice}!")
        st.metric(label="Predicted Item Outlet Sales", value=f"${prediction:.2f}")
        
        # 3. Save the human-readable data to the Website's Memory
        new_record = pd.DataFrame({
            'Item_Identifier': [item_identifier],
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Identifier': [outlet_identifier],
            'Outlet_Establishment_Year': [outlet_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location],
            'Outlet_Type': [outlet_type],
            'Predicted_Sales': [round(prediction, 2)] # Add the prediction!
        })
        
        # Append the new record to our running dataset in session_state
        st.session_state['my_dataset'] = pd.concat([st.session_state['my_dataset'], new_record], ignore_index=True)

    # Display the built dataset if it has data in it
    if not st.session_state['my_dataset'].empty:
        st.divider()
        st.subheader("📝 Your Custom Dataset")
        st.write("Every time you predict a single item, it is saved here. You can download this custom list anytime.")
        st.dataframe(st.session_state['my_dataset'])
        
        # Download button for the custom dataset
        custom_csv = st.session_state['my_dataset'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Your Custom Dataset",
            data=custom_csv,
            file_name='My_Custom_BigMart_Data.csv',
            mime='text/csv',
        )

# ==========================================
# TAB 2: BULK UPLOAD (From previous step)
# ==========================================
with tab2:
    st.header("Upload Dataset for Bulk Prediction")
    st.write("Upload your `test.csv` file here. The AI will process all rows and generate predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.write("Preview of your uploaded data:")
        st.dataframe(test_data.head())
        
        bulk_model_choice = st.radio("Choose Algorithm for Bulk Prediction:", ["XGBoost", "Random Forest"])
        
        if st.button("Generate Bulk Predictions", type="primary", key="bulk_btn"):
            with st.spinner("Cleaning data and running AI models..."):
                process_data = test_data.copy()
                
                process_data['Item_Weight'] = process_data['Item_Weight'].fillna(process_data['Item_Weight'].mean())
                process_data['Outlet_Size'] = process_data['Outlet_Size'].fillna('Medium') 
                process_data['Item_Fat_Content'] = process_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
                
                mapping_dict = {
                    'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1},
                    'Outlet_Size': {'High': 0, 'Medium': 1, 'Small': 2},
                    'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
                    'Outlet_Type': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
                }
                for col, mapping in mapping_dict.items():
                    process_data[col] = process_data[col].map(mapping)
                    
                item_type_list = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]
                process_data['Item_Type'] = process_data['Item_Type'].apply(lambda x: item_type_list.index(x) if x in item_type_list else 0)
                
                process_data['Item_Identifier'] = 0
                process_data['Outlet_Identifier'] = 0
                
                features = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
                process_data = process_data[features]
                
                if bulk_model_choice == "XGBoost":
                    predictions = xgb_model.predict(process_data)
                else:
                    predictions = rf_model.predict(process_data)
                
                test_data['Predicted_Outlet_Sales'] = predictions
                
                st.success("✅ Bulk Prediction Complete!")
                st.dataframe(test_data)
                
                csv = test_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name='BigMart_Bulk_Predictions.csv',
                    mime='text/csv',
                )