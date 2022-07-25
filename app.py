import streamlit as st
from PIL import Image
import pickle
classifier = pickle.load(open('Capstone_svm.pkl', 'rb'))
def welcome():
    return 'welcome all'

def prediction(tweet):  
   
    prediction = classifier.predict(
        [[tweet]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
    global result
      # giving the webpage a title
    st.title("Tweet type Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    tweet= st.text_input("tweet", "Type Here")
    if st.button("Predict"):
        st.success('The output is {}'.format( prediction(tweet)))
     
    if __name__=='__main__':
        main()