# AAI540-Final-G3
Final GitHub Repo for Group 3 of AAI-540 (Spring 2026) Final Project


# 📊 AI Model Routing System — End-to-End ML Pipeline

## Overview

This project implements an intelligent routing system that selects the most appropriate AI model for a given request based on:

* token usage
* latency requirements
* quality constraints
* domain characteristics

The system trains machine learning routers to predict which model should handle a request, and compares different routing approaches.

The pipeline includes:

✔ Data preprocessing
✔ Cloud storage and Athena integration
✔ Feature engineering
✔ Model training (Random Forest, XGBoost)
✔ Model evaluation
✔ Model comparison

---

## 🎯 Project Objective

The goal is to design and evaluate machine learning models that can route inference requests to optimal AI models under operational constraints.

The routers learn from labeled synthetic request data and predict the best model to serve each request.

Two routing models are trained and compared:

* Random Forest Router
* XGBoost Router

Their performance is evaluated using classification metrics and prediction behavior.

---

## 🧩 Pipeline Structure

The notebooks are organized sequentially to reflect the full data and model lifecycle.

---

### 🔹 Data Preparation

**00_data_preprocessing_all_sources.ipynb**

Cleans and standardizes all raw datasets.

Tasks:

* handle missing values
* normalize numeric formats
* standardize column names
* prepare datasets for Athena ingestion

---

### 🔹 Cloud Infrastructure Setup

**01_infra_configure_s3_bucket.ipynb**

Configures AWS environment.

Tasks:

* initialize SageMaker session
* create or access S3 bucket
* define storage paths
* upload datasets to S3

---

### 🔹 Data Warehouse (AWS Athena Tables)

Each notebook creates an Athena table from a dataset stored in S3.

* 02_athena_create_table_aimodelpoll.ipynb
* 03_athena_create_table_llmachievements.ipynb
* 04_athena_create_table_model_profiles.ipynb
* 05_athena_create_table_lifearch_models.ipynb
* 06_athena_create_table_llm_pricing.ipynb
* 07_athena_create_table_llm_leaderboard.ipynb
* 08_athena_create_table_overview_models.ipynb
* 09_athena_create_table_simplebench.ipynb

Purpose:

* structured querying
* centralized data access
* analytics integration

---

### 🔹 Model Training — Primary Router

**10_router_training_random_forest.ipynb**

Trains the baseline routing model.

Features include:

* prompt tokens
* output tokens
* total tokens
* latency requirement
* quality requirement
* domain
* strict latency flag

Outputs:

* trained Random Forest router
* classification metrics
* feature importance
* confusion matrix
* predicted model labels

Model saved to:
S3 /models/

---

### 🔹 Model Training — Alternative Router

**11_router_training_xgboost.ipynb**

Trains a gradient boosted decision tree router using the same features.

Purpose:

* compare with Random Forest
* test alternative learning approach

Outputs:

* trained XGBoost router
* prediction performance plots

---

### 🔹 Model Comparison

**12_router_model_comparison.ipynb**

Compares both routers on identical test data.

Evaluates:

✔ Accuracy
✔ Macro F1
✔ Weighted F1
✔ Top-3 Accuracy
✔ Confusion matrices
✔ Feature importance
✔ Model prediction distribution
✔ Agreement rate
✔ Statistical comparison (McNemar test)

This notebook provides the final evaluation used in the project report.

---

## 🧠 Machine Learning Models

### Random Forest Router

* ensemble of decision trees
* robust baseline
* interpretable feature importance

### XGBoost Router

* gradient boosted trees
* optimized for predictive accuracy
* handles nonlinear interactions efficiently

Both models perform multi-class classification where each class represents a target AI model.

---

## 📂 Data Sources

The project integrates multiple AI benchmarking and metadata datasets:

* AI model performance benchmarks
* model pricing data
* capability comparisons
* leaderboard rankings
* architecture metadata
* synthetic request dataset (training labels)

All datasets are stored in S3 and accessed via Athena.

---

## ☁ AWS Architecture

The system uses:

* Amazon SageMaker — model training
* Amazon S3 — data storage
* AWS Athena — data querying

Workflow:

Raw Data → Preprocessing → S3 Storage → Athena Tables → Feature Engineering → Model Training → Evaluation

---

## ▶ Execution Order

Run notebooks sequentially:

00_data_preprocessing_all_sources
01_infra_configure_s3_bucket

02–09 Athena table creation

10_router_training_random_forest
11_router_training_xgboost

12_router_model_comparison

---

## 📈 Evaluation Approach

Models are compared purely on prediction performance.

Metrics used:

* Accuracy
* Macro F1
* Weighted F1
* Top-K accuracy
* Confusion matrices
* Prediction agreement
* Statistical significance testing

This ensures a fair comparison without external cost or latency simulation.

---

## 📊 Key Outputs

The pipeline produces:

* trained routing models
* labeled request dataset
* evaluation metrics
* performance visualizations
* comparison tables

These outputs support analytical conclusions about routing effectiveness.

---

## 🧪 Reproducibility

To reproduce results:

1. Use AWS SageMaker environment
2. Run notebooks in order
3. Ensure datasets exist in S3
4. Restart kernel after installing dependencies
5. Use consistent random seeds

---

## 📚 Project Outcome

The project demonstrates how machine learning can automate AI model selection under operational constraints.

It provides:

✔ a reproducible ML pipeline
✔ comparative model analysis
✔ practical routing framework

---

## 👤 Authors

Project developed as part of an academic machine learning system design study.

---

## 📄 License

Educational use.
