import streamlit as st

st.button('Re-run')

st.title('Predict visual novel popularity')

import joblib
clf = joblib.load('model_fitted.joblib')

vn_name = "Muv-Luv Alternative"

import pandas as pd
releases = pd.read_csv("releases.csv", low_memory=False)
st.write(releases.head(20))

def get_vn_id(name=None):
    # Return -1 if no name are provided
    if name == None:
        return -1

    # Only check with lowercase name to improve precision
    name = name.lower()

    for ind in releases.index: 
        id = releases['id'][ind]
        original_title = releases['title'][ind]
        title = original_title.lower()

        # If the title contains the provided name, return its id
        if name in title:
            st.write("Visual novel found:", original_title)
            return id

    # Return -1 when no vn is founded
    return -1

# User input
user_input = st.text_input("User input:", vn_name)
# ID of visual novel
id = get_vn_id(user_input)
cleaned_with_id = pd.read_csv('cleaned_with_id.csv', low_memory=False)
if id != -1:
    try:
        # Prediction
        df_prov = cleaned_with_id.query('id == ' + str(id)).drop(columns=['id', 'c_popularity'])
        value = df_prov.iloc[0,:].to_list()
        # Result
        result = clf.predict([value])

        st.write(result)
    except IndexError:
        st.write('Could not find the provided visual novel in database.')
else:
    st.write('Could not find the provided visual novel in database.')
