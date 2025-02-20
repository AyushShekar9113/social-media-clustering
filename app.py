import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI-Powered Social Media Clustering", layout="wide")
st.title("📊 AI-Powered Social Media Text Clustering")

# ---- GENERATE RANDOM SOCIAL MEDIA DATA ----
st.sidebar.header("Step 1: Generate Social Media Data")
n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 500)

# Sample phrases for social media posts
sample_texts = [
    "AI is changing the world! 🚀", "Love this new tech update! 🔥", "Streaming my gaming session now 🎮",
    "Just finished a great book 📚", "Can't believe this happened! 😲", "Morning coffee ☕ and deep thoughts",
    "Another day, another grind 💪", "Crypto market is crazy right now! 📈", "Weekend vibes 😎",
    "Breaking news: Major event happening now! 📰", "AI vs Humans debate 🤖", "Excited for the new movie release 🎬",
    "Workout complete! 🏋️", "Nature is so beautiful 🌿", "Politics is heating up again 🔥",
    "Who's watching the game tonight? 🏀", "Cooking a new recipe today! 🍲", "Productivity hacks you need! 🏆",
    "Deep learning is the future of AI 🤯", "Hiking in the mountains today 🏔️"
]

# Generate random social media posts
random_texts = [random.choice(sample_texts) for _ in range(n_samples)]
st.write("📂 **Generated Social Media Posts:**", pd.DataFrame(random_texts, columns=["Post"]).head())

# ---- TEXT PREPROCESSING ----
st.sidebar.header("Step 2: Preprocess Text")

vectorizer = TfidfVectorizer(max_features=1000)
data_vectorized = vectorizer.fit_transform(random_texts).toarray()

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_vectorized)

# ---- AUTOENCODER TRAINING ----
st.sidebar.header("Step 3: Train Autoencoder")

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(data_scaled.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu')  # Latent space
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(data_scaled.shape[1], activation='sigmoid')
])

autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.output))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train Autoencoder
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
batch_size = st.sidebar.slider("Batch Size", 32, 512, 256)

if st.sidebar.button("Train Autoencoder"):
    with st.spinner("Training Autoencoder..."):
        autoencoder.fit(data_scaled, data_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        st.success("Training Complete!")

# Extract Latent Features
latent_features = encoder.predict(data_scaled)

# ---- K-MEANS CLUSTERING ----
st.sidebar.header("Step 4: Apply K-Means Clustering")
n_clusters = st.sidebar.slider("Number of Clusters", 3, 10, 5)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(latent_features)

# Calculate Evaluation Metrics
silhouette_avg = silhouette_score(latent_features, clusters)

st.subheader(f"🚀 Clustering Results (K={n_clusters})")
st.write(f"✅ **Silhouette Score:** {silhouette_avg:.4f}")

# ---- VISUALIZATION ----
st.subheader("📍 Cluster Visualization")

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_features)
latent_df = pd.DataFrame(latent_2d, columns=['Feature_1', 'Feature_2'])
latent_df["Cluster"] = clusters

fig = px.scatter(latent_df, x="Feature_1", y="Feature_2", color=latent_df["Cluster"].astype(str),
                 title="Clusters in Latent Space", color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig)

# ---- RECONSTRUCTION LOSS ----
st.sidebar.header("Step 5: Compute Reconstruction Loss")
if st.sidebar.button("Calculate Loss"):
    reconstructed_data = autoencoder.predict(data_scaled)
    mse_loss = np.mean(np.square(data_scaled - reconstructed_data))
    st.sidebar.write(f"🛠 **Reconstruction Loss:** {mse_loss:.4f}")
