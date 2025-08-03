
# 🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendations

This project focuses on customer segmentation and personalized product recommendations using machine learning techniques. It helps e-commerce businesses understand their customer base, group them based on behavior (RFM model), and suggest relevant products using collaborative filtering.

---

## 📌 Problem Statement

E-commerce platforms serve a wide range of customers with diverse behaviors and preferences. However, sending the same promotions to all customers leads to poor engagement and wasted marketing efforts. This project solves this challenge by clustering customers based on their purchasing behavior (Recency, Frequency, Monetary - RFM) and recommending products using item-based collaborative filtering. This enables businesses to offer personalized experiences, increase customer retention, and boost revenue.

---

## 💡 Key Features

- 📊 **Customer Segmentation** using KMeans clustering on RFM features  
- 🔍 **Cluster Profiling** to identify High-Value, Loyal, At-Risk customers  
- 🛒 **Product Recommendations** using cosine similarity  
- 🧠 **Item-based Collaborative Filtering** for real-time suggestions  
- 📈 **Silhouette Score Evaluation** for optimal cluster quality  
- 🧼 **Data Cleaning Pipeline** to remove duplicates, invalid transactions, and outliers  
- ☁️ **Streamlit Web App** for interactive use

---

## 🚀 Deployment

The project is deployed and accessible as a **Streamlit Web App**:  
👉 [Launch App](https://mntyf3csgfdbjauil4mdx5.streamlit.app) 

---

## 🧪 How It Works

### 🧹 Data Cleaning
- Removed duplicates
- Excluded negative quantities and prices
- Removed canceled transactions
- Parsed and filtered dates

### 📊 RFM Segmentation
- `Recency`: Days since last purchase
- `Frequency`: Number of unique invoices
- `Monetary`: Total spend

KMeans was applied to segment customers into 4 behavioral clusters:
- 👑 High-Value
- 🔁 Loyal/Regular
- 💤 Medium Value
- ⚠️ At-Risk

### 🧠 Product Recommendation Engine
- Created a user–product matrix using `CustomerID` and `StockCode`
- Calculated cosine similarity between products
- Returned top 5 similar items for any input product

---

## 🖥️ Streamlit Interface

- 📌 Sidebar with "Customer Clustering" and "Product Recommendation"
- 📈 Visual charts for RFM segmentation
- 🔍 Real-time product suggestion engine
- 📁 Upload new data for live predictions

---

## 🛠️ Installation & Requirements

### 🔗 Clone the repo
```bash
git clone https://github.com/Pavan-Kumar-Dirisala/Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce.git
cd Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce
````

### ⚙️ Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the app locally

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
├── app.py                        # Streamlit main app
├── clustering_model.pkl         # Saved KMeans model
├── product_similarity.pkl       # Cosine similarity matrix (excluded from repo)
├── cleaned_data.csv             # Cleaned customer transaction dataset
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignored files
```

---

## ✅ .gitignore Example

```
*.pkl
.ipynb_checkpoints/
__pycache__/
.DS_Store
.env
```

---

## ✍️ Author

* **Pavan Kumar Dirisala**
  B.Tech CSE – Artificial Intelligence & Intelligent Process Automation
  [GitHub](https://github.com/Pavan-Kumar-Dirisala)

