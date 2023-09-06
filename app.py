import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import bias_variance_decomp
import numpy as np

# Use a dark background style for plots
plt.style.use('dark_background')

# Function to generate custom data
def generate_data(n_classes, n_samples, pattern='Linear'):
    X = np.zeros((n_classes*n_samples, 2))
    y = np.zeros(n_classes*n_samples, dtype='uint8')
    for j in range(n_classes):
        ix = range(n_samples*j, n_samples*(j+1))
        if pattern == 'Spiral':
            r = np.linspace(0.0, 1, n_samples) # radius
            t = np.linspace(j*4, (j+1)*4, n_samples) + np.random.randn(n_samples)*0.2 # theta
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        elif pattern == 'Linear':
            X[ix] = np.random.rand(n_samples, 2) * [j * 2, 1] + np.random.randn(n_samples, 2) * 0.2
        elif pattern == 'Concentric Circle':
            t = np.linspace(0, 2*np.pi, n_samples)
            r = j/float(n_classes) + np.random.randn(n_samples)*0.1
            X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
        elif pattern == 'Blob':
            t = np.linspace(0, 2*np.pi, n_samples)
            r = 0.8 + np.random.randn(n_samples)*0.1
            X[ix] = np.c_[r*np.cos(t), r*np.sin(t)] + np.random.randn(n_samples, 2)*0.2
        elif pattern == 'Crescent':
            half_samples = int(n_samples / 2)
            theta = np.linspace(j * np.pi, (j + 2) * np.pi, n_samples)
            r = np.linspace(1.0, 2.5, half_samples)
            r = np.concatenate((r, np.linspace(2.5, 1.0, half_samples)))
            X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
        elif pattern == 'Normal':
            for j in range(n_classes):
                ix = range(n_samples*j, n_samples*(j+1))
                X[ix] = np.random.randn(n_samples, 2) * 0.5 + np.random.randn(2) * j * 2
                y[ix] = j
            return X, y
        elif pattern == 'Random':
            X[ix] = np.random.randn(n_samples, 2)*0.5 + np.random.randn(2)*j*2
        else:
            raise ValueError('Invalid pattern: {}'.format(pattern))
        y[ix] = j
    return X, y

# Function to plot decision boundary and calculate model evaluation metrics
def keffect(k):
    X, y = generate_data(num_classes, num_data_points, pattern=pattern)
    
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    mse, bias, var = bias_variance_decomp(knn, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
    
    # Create a meshgrid for decision boundary plotting
    a = np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=selected_step)
    b = np.arange(start=X_train[:,1].min()-1, stop=X_train[:,1].max()+1, step=selected_step)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    labels = knn.predict(input_array)
    
    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#FFF')
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=selected_alpha, cmap='Set1')
    scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='Set1', edgecolors='black')
    ax.set_title('K-Nearest Neighbors (K = {})'.format(k), color='white')
    ax.set_xlabel('Feature 1', color='white')
    ax.set_ylabel('Feature 2', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    result = [accuracy, mse, bias, var]
    return fig, result

# Function to plot bias-variance tradeoff
def plot_bias_variance_tradeoff(start_value, end_value):
    X, y = generate_data(num_classes, num_data_points, pattern=pattern)

    ks = range(start_value, end_value)
    mse, bias, var = [], [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        mse_k, bias_k, var_k = bias_variance_decomp(knn, X, y, X, y, loss='mse', num_rounds=200, random_seed=1)
        mse.append(mse_k)
        bias.append(bias_k)
        var.append(var_k)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(ks, mse, label='MSE', color='crimson')
    ax.plot(ks, bias, label='Bias', color='magenta')
    ax.plot(ks, var, label='Variance', color='cyan')
    ax.legend()
    ax.set_title('Bias-Variance Tradeoff', color='white')
    ax.set_xlabel('Number of Neighbors (K)', color='white')
    ax.set_ylabel('Error', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xticks(list(range(start_value, end_value, 5)) + [end_value])
    ax.set_facecolor('#000')

    return fig

# Create a streamlit app to interact with the functions
st.set_page_config(page_title='K-Nearest Neighbors', layout='wide')
st.title('K-Nearest Neighbors')

with st.sidebar:
    # Set up Streamlit sidebar

    # [fig_width, fig_height] = [st.sidebar.slider(label, 1, 20, default) for label, default in [("Figure Width", 10), ("Figure Height", 6)]]
    # selected_alpha = st.sidebar.slider('Select the transparency of the decision boundary', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    # selected_step = st.sidebar.slider('Select the stepsize for the grid', min_value=1, max_value=10, value=5, step=1) * 0.01/5
    fig_width, fig_height = 10, 6
    selected_alpha = 0.5
    selected_step = 0.01
    
    pattern = st.selectbox('Select a dataset pattern', ['Linear', 'Concentric Circle', 'Spiral', 'Blob', 'Crescent', 'Normal', 'Random'])
    num_classes = st.slider('Select the number of classes', min_value=2, max_value=10, value=2, step=1)
    num_data_points = st.slider('Select the number of data points', min_value=20, max_value=200, value=40, step=20)
    
    st.write("---")
    
    selected_k = st.slider(label="Select the number of neighbors (K)", min_value=1, max_value=50, value=3, step=1)
    range_slider = st.slider(
        label="Select a range for bias-variance tradeoff",
        min_value=1,
        max_value=50,
        value=(1, 20),
        step=1
    )
    start_value, end_value = range_slider

if st.button('Run'):
    # st.write('Decision Boundary')
    fig, result = keffect(min(selected_k, num_data_points))
    st.write(fig)

    st.write('Model evaluation metrics')
    st.write('Accuracy:', round(result[0], 3))
    st.write('MSE:', round(result[1], 3))
    st.write('Bias:', round(result[2], 3))
    st.write('Variance:', round(result[3], 3))
    
    fig2 = plot_bias_variance_tradeoff(min(start_value, num_data_points), min(end_value, num_data_points))
    st.write(fig2)

# if st.button('Get Bias-Variance Tradeoff'):
    ## st.write('Bias-Variance Tradeoff')
    # fig2 = plot_bias_variance_tradeoff(min(start_value, num_data_points), min(end_value, num_data_points))
    # st.write(fig2)
