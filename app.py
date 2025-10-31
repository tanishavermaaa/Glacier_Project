
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.cluster import KMeans  

st.set_page_config(layout="wide", page_title="Global Glacier Health Index")

@st.cache_data  
def load_data(filepath):
    
    try:
        df = pd.read_csv(filepath)
   
        df = df.rename(columns={
            'glac_id': 'RGI_ID',
            'melt_rate': 'Melt_Rate',
            'lat': 'Latitude',
            'lon': 'Longitude'
        })
        

        required_cols = ['RGI_ID', 'Latitude', 'Longitude', 'Melt_Rate']

        df = df.dropna(subset=required_cols)

        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Melt_Rate'] = pd.to_numeric(df['Melt_Rate'], errors='coerce')
        

        df = df.dropna(subset=required_cols)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=required_cols)
        
        return df
        
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Make sure 'glacier_final_cleaned_ready.csv' is in your GitHub repository.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        st.write("Please ensure 'glacier_final_cleaned_ready.csv' is in the correct format.")
        return None


def run_ml_clustering(df):
    """
    This is the "brains" of your project, updated for the new file.
    We will perform K-Means Clustering based on the single feature
    we have: 'Melt_Rate'.
    """
    df_ml = df.copy()

    features = ['Melt_Rate']

    scaler = MinMaxScaler()
 
    data_scaled = scaler.fit_transform(df_ml[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    

    df_ml['ML_Cluster'] = kmeans.fit_predict(data_scaled)


    cluster_centers = kmeans.cluster_centers_

    sorted_cluster_indices = np.argsort(cluster_centers.flatten())
    
    # Create the mapping
    cluster_mapping = {
        sorted_cluster_indices[0]: "Critical",  # Lowest (most negative) melt rate
        sorted_cluster_indices[1]: "Moderate",  # Middle melt rate
        sorted_cluster_indices[2]: "Healthy"    # Highest (closest to 0) melt rate
    }
    
    # Map the cluster numbers (0, 1, 2) to our new text labels
    df_ml['ML_Category'] = df_ml['ML_Cluster'].map(cluster_mapping)
    
    return df_ml

def create_map(df):
    """
    This function creates the interactive Folium map
    (Step 6 from your PDF).
    """
    # Create a base map, centered roughly on the world
    m = folium.Map(location=[30, 0], zoom_start=2)

    # Define colors for our categories
    color_map = {
        "Healthy": "green",
        "Moderate": "orange",
        "Critical": "red"
    }

    # Loop through every glacier in our data
    for idx, row in df.iterrows():
        # Get the color based on its ML Category
        color = color_map.get(row['ML_Category'], 'gray') # Default to gray

        # Create the popup text that appears when you click
        # **MODIFICATION**: Using your new dataset's columns
        popup_html = f"""
        <b>Glacier ID:</b> {row['RGI_ID']}<br>
        <b>ML Category:</b> {row['ML_Category']}<br>
        <hr>
        <b>Melt Rate:</b> {row['Melt_Rate']:.2e} (scientific notation)<br>
        <b>Location:</b> ({row['Latitude']:.3f}, {row['Longitude']:.3f})
        """

        # Add a circle marker to the map
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    return m

# --- 4. Main Application ---
# This is what runs when you start the app.

def main():
    st.title("❄️ Global Glacier Health Index (GHI) ❄️")
    st.write("An AI-powered system to assess glacier conditions worldwide, based on your project proposal and research paper.")
    # **MODIFICATION**: Updated info text
    st.info("Now using your provided `glacier_final_cleaned_ready.csv` file. This analysis is based on K-Means Clustering of 'Melt_Rate'.")

    # --- Load Data ---
    # **MODIFICATION**: Load the correct file
    df_raw = load_data("glacier_final_cleaned_ready.csv")
    
    if df_raw is not None and not df_raw.empty:
        # --- Run Analysis ---
        # 1. Run ML Models (K-Means) (from project summary)
        df_final = run_ml_clustering(df_raw)
        
        # --- Build the UI ---
        
        col1, col2 = st.columns([3, 2]) # Map is 3 parts wide, stats are 2
        
        with col1:
            st.subheader("Interactive Glacier Health Map")
            glacier_map = create_map(df_final)
            st_folium(glacier_map, width= '100%')

        with col2:
            st.subheader("Project Analytics")
            
            st.write(f"""
            **K-Means Clustering Analysis (Steps 4 & 5):**
            We analyzed **{len(df_final)}** glaciers from your dataset.
            
            The ML model grouped them into 3 clusters based on their
            'Melt_Rate', which we then labeled:
            """)

            # Show ML Category distribution
            st.subheader("ML-Driven Cluster Distribution")
            ml_counts = df_final['ML_Category'].value_counts()
            st.bar_chart(ml_counts)
            
            st.write("---")
            st.write(f"**Total Glaciers:** {len(df_final)}")
            st.write(f"**Healthy:** {ml_counts.get('Healthy', 0)}")
            st.write(f"**Moderate:** {ml_counts.get('Moderate', 0)}")
            st.write(f"**Critical:** {ml_counts.get('Critical', 0)}")


        # --- Show the Raw Data ---
        st.subheader("Glacier Data & Calculated Scores")
        st.write("This table shows the data from `glacier_final_cleaned_ready.csv` plus our ML-driven cluster.")
        
        # **MODIFICATION**: Show the columns we are actually using
        display_cols = ['RGI_ID', 'Latitude', 'Longitude', 'Melt_Rate', 'ML_Category']
        
        # We need to format Melt_Rate to be readable
        df_display = df_final[display_cols].copy()
        df_display['Melt_Rate'] = df_display['Melt_Rate'].apply(lambda x: f"{x:.2e}")
        
        st.dataframe(df_display)
    
    elif df_raw is not None and df_raw.empty:
        st.error("The loaded data is empty after cleaning. Please check 'glacier_final_cleaned_ready.csv' for valid data.")

# This line tells Python to run the "main()" function when you execute the file
if __name__ == "__main__":
    main()



