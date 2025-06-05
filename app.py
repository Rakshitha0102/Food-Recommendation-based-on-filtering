import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('food_recommendation_merged1.csv')  

st.set_page_config(page_title="Food Recommendation System", layout="wide")

st.title("Food Recommendation System")


st.sidebar.header("Recommendation Type")
rec_type = st.sidebar.radio(
    "Select Recommendation System",
    ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Recommendation System"]
)
st.subheader("Explore Our Food Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("food.jpeg", use_column_width=True) 

with col2:
    st.image("health.jpeg", use_column_width=True)  

with col3:
    st.image("french.jpeg",use_column_width=True)


def content_based_recommendation(cuisine_type, veg_non, num_recommendations=5):

    filtered_data = data[(data['C_Type'] == cuisine_type) & (data['Veg_Non_encoded'] == veg_non)]

    if filtered_data.empty:
        st.warning("No data found for the selected criteria.")
        return pd.DataFrame(columns=['Name', 'Cuisine', 'Rating'])
    filtered_data = filtered_data.groupby('Name').agg({'Rating': 'mean', 'C_Type': 'first'}).reset_index()
    
    tfidf = TfidfVectorizer(stop_words='english')
    filtered_data['Name'] = filtered_data['Name'].fillna('') 
    tfidf_matrix = tfidf.fit_transform(filtered_data['Name'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(range(len(filtered_data)), index=filtered_data['Name']).drop_duplicates()

    try:
        idx = indices.sample(1).iloc[0]  
    except ValueError:
        st.warning("Unable to select a sample from the filtered data.")
        return pd.DataFrame(columns=['Name', 'Cuisine', 'Rating'])

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]  
    food_indices = [i[0] for i in sim_scores if i[0] < len(filtered_data)]
    
    recommendations = filtered_data.iloc[food_indices][['Name', 'C_Type', 'Rating']]

    recommendations = recommendations.rename(columns={'C_Type': 'Cuisine'})
    
    return recommendations

def collaborative_recommendation(user_id, num_recommendations=5):
    data_unique = data.drop_duplicates(subset=['Food_ID', 'User_ID'], keep='first')
    
    pivot_table = (
        data_unique.groupby(['Food_ID', 'User_ID'])
        .agg({'Rating': 'mean'})
        .reset_index()
        .pivot(index='Food_ID', columns='User_ID', values='Rating')
        .fillna(0)
    )

    user_ratings = pivot_table.values.T  
    cosine_sim = cosine_similarity(user_ratings)

    if user_id not in pivot_table.columns:
        st.warning(f"User ID {user_id} not found in the dataset.")
        return pd.DataFrame(columns=['Name', 'Cuisine', 'Rating'])
    
    user_index = pivot_table.columns.get_loc(user_id)

    user_similarities = cosine_sim[user_index]
    similar_users = user_similarities.argsort()[-num_recommendations-1:-1][::-1]
 
    recommended_foods = set()  
    for similar_user in similar_users:
        similar_user_ratings = pivot_table.iloc[:, similar_user]
        top_rated_foods = similar_user_ratings[similar_user_ratings > 0].index
        recommended_foods.update(top_rated_foods)  
    
    recommended_foods = list(recommended_foods)
    recommendations = (
        data[data['Food_ID'].isin(recommended_foods)]
        .groupby('Food_ID')
        .agg({'Name': 'first', 'C_Type': 'first', 'Rating': 'mean'})
        .reset_index()
    )
    recommendations = recommendations.rename(columns={'C_Type': 'Cuisine'})
    
    return recommendations

def hybrid_recommendation(user_id, cuisine_type, veg_non, num_recommendations=5):
    content_recs = content_based_recommendation(cuisine_type, veg_non, num_recommendations)
    collaborative_recs = collaborative_recommendation(user_id, num_recommendations)
    

    hybrid_recs = pd.concat([content_recs, collaborative_recs]).drop_duplicates(subset=['Name', 'Cuisine', 'Rating']).head(num_recommendations)
    
    hybrid_recs = hybrid_recs[['Name', 'Cuisine', 'Rating']]
    
    return hybrid_recs

if rec_type == "Content-Based Filtering":

    col1, col2 = st.columns([1, 2])
    with col1:
        cuisine_type = st.selectbox("Select Cuisine Type:", data['C_Type'].unique())
        veg_non = st.radio("Veg or Non-Veg:", ["Veg", "Non-Veg"])
        veg_non_encoded = 1 if veg_non == "Veg" else 0
    with col2:
        if cuisine_type:
            st.subheader("Recommendations")
            recommendations = content_based_recommendation(cuisine_type, veg_non_encoded)
            st.dataframe(recommendations, use_container_width=True)

elif rec_type == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID:", min_value=int(data['User_ID'].min()), max_value=int(data['User_ID'].max()))
    if user_id:
        st.subheader("Recommendations")
        recommendations = collaborative_recommendation(user_id)
        st.dataframe(recommendations, use_container_width=True)

elif rec_type == "Hybrid Recommendation System":
    # Layout
    col1, col2 = st.columns([1, 2])
    with col1:
        user_id = st.number_input("Enter User ID:", min_value=int(data['User_ID'].min()), max_value=int(data['User_ID'].max()))
        cuisine_type = st.selectbox("Select Cuisine Type:", data['C_Type'].unique())
        veg_non = st.radio("Veg or Non-Veg:", ["Veg", "Non-Veg"])
        veg_non_encoded = 1 if veg_non == "Veg" else 0
    with col2:
        if user_id and cuisine_type:
            st.subheader("Recommendations")
            recommendations = hybrid_recommendation(user_id, cuisine_type, veg_non_encoded)
            st.dataframe(recommendations, use_container_width=True)

