import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# #Set title


image = Image.open('EDA.png')
st.image(image,use_column_width=True)

st.title('Interactive Machine Learning Application')
#set subtitle

st.write(""" ## **A simple Data Application for Machine Learning Purposes** """)
st.write(""" ### Let's explore different classifiers and datasets """)

def main():
    activities=['Background Information','EDA','Visualization','Model']
    option=st.sidebar.selectbox('Selection option:',activities)

    if option=='EDA':
        st.subheader("Exploratory Data Analysis")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")

        if data is not None:
            data.seek(0)
            df=pd.read_csv(data, low_memory=False)
            st.dataframe(df.head(50))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns=st.multiselect('Select preferred columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display summary"):
                st.write(df.describe().T)

            if st.checkbox("Display Null Values"):
                st.write(df.isnull().sum())

            if st.checkbox("Display the data type"):
                st.write(df.dtypes)

            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr())

# Dealing with the visualization part
    elif option=='Visualization':
        st.subheader("Data Visualization")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")

        if data is not None:
            data.seek(0)
            df=pd.read_csv(data, low_memory=False)
            st.dataframe(df.head(50))

            if st.checkbox('Select multiple columns to plot'):
                selected_columns=st.multiselect('Select your preferred columns', df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                #map = ()
                #st.write(map)
                fig, ax = plt.subplots()
                ax = (sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                st.pyplot(fig)
                #plt.savefig(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                if st.checkbox('Display Pairplot'):
                    fig, ax = plt.subplots()
                    ax = (sns.pairplot(df1,diag_kind='kde'))
                    st.pyplot(fig)
                if st.checkbox('Display Pie Chart'):
                    all_columns = df.columns.to_list()
                    pie_columns=st.selectbox("select column to display",df.columns)
                    fig, pieChart = plt.subplots()
                    pieChart = (df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot(fig)

        # DEALING WITH MODEL BUILDING

    elif option=='Model':
        st.subheader("Model Building")

        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            data.seek(0)
            df=pd.read_csv(data, low_memory=False)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns'):
                new_data=st.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected", df.columns)
                df1=df[new_data]
                st.dataframe(df1)

                #Dividing my data into x and y variables
                X=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]

            seed=st.sidebar.slider('Seed',1,200)

            classifier_name=st.sidebar.selectbox('Select your preferred classifier:',('KNN','SVM','LR','naive_bayes','decision tree'))

            def add_parameter(name_of_clf):
                params=dict()
                if name_of_clf=='SVM':
                    C=st.sidebar.slider('C',0.01,15.0)
                    params['C']=C
                else:
                    name_of_clf=='KNN'
                    K=st.sidebar.slider('K',1,15)
                    params['K']=K
                    return params

            #calling the Function

            params=add_parameter(classifier_name)

            #defing a function for our classifiers
            def get_classifier(name_of_clf,params):
                clf= None
                if name_of_clf=='SVM':
                    clf=SVC(C=params['C'])
                elif name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf=='LR':
                    clf=LogisticRegression()
                elif name_of_clf=='naive_bayes':
                    clf=GaussianNB()
                elif name_of_clf=='decision tree':
                    clf=DecisionTreeClassifier()
                else:
                    st.warning('Select your choice of algorithm')
                return clf

            clf=get_classifier(classifier_name,params)

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)

            clf.fit(X_train,y_train)

            y_pred=clf.predict(X_test)
            st.write('Predictions:',y_pred)

            accuracy=accuracy_score(y_test,y_pred)

            st.write('Name of classifier:', classifier_name)
            st.write('Accuracy',accuracy)

#DEALING WITH THE ABOUT PAGE

    elif option=='Background Information':
        st.markdown(' This is an interactive web page for our ML project, feel free to use it. The analysis in here is to demonstrate our work to our stakeholders in an interactive way by building a web app for our machine learning algorithms using different option')
        st.markdown('Look to your left for navigating through available options')

        st.balloons()

if __name__ == '__main__':
    main()
