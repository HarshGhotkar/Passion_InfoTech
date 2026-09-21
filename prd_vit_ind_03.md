# Product Requirements Document (PRD)

## Project Details
- **Project Name:** VIT-IND-03: Smart Manufacturing: Acoustic AI-Based Metal Parts Quality Inspection & Predictive Maintenance
- **Domain:** Metal Parts & Industrial Quality Inspection
- **Thrust Area:** Smart Manufacturing
- **Document Role:** Senior Technical Product Manager

---

## 1. Product Vision & Objectives

### 1.1 Core Problem
Manufacturing industries currently rely on visual inspections and periodic manual maintenance schedules. This approach suffers from several key limitations:
- **Invisible Defects:** Internal cracks and micro-fractures in metal components develop beneath the surface before catastrophic failure and remain visually undetectable.
- **Time Inefficiency:** Manual inspection is time-consuming and often requires planned shutdowns.
- **Reactive Breakdowns:** Unexpected machine breakdowns lead to substantial production losses.
- **Sensor Limitations:** Traditional sensors frequently miss early-stage degradation signals.

### 1.2 Solution Concept & Vision
To build an **AI-powered Acoustic Predictive Quality Inspection Platform** that serves as a non-invasive structural health monitoring system. By leveraging Acoustic Machine Learning, Signal Processing, and IoT Sensors, the system will listen to machine vibrations and acoustic signatures to detect structural cracks, material fatigue, and abnormal conditions, thereby predicting failure probabilities before they occur. 

### 1.3 SDG Alignment
This product aligns directly with the UN Sustainable Development Goals:
- **SDG 9 (Industry, Innovation and Infrastructure):** Fosters smart industrial infrastructure, promoting resilient innovation through AI-based non-invasive monitoring.
- **SDG 12 (Responsible Consumption and Production):** Ensures efficient resource utilization by maximizing industrial asset life, reducing part wastage, and minimizing unnecessary component replacements through predictive action.

---

## 2. Target Architecture
The system employs a multi-tiered architecture to ensure real-time responsiveness and scalable data processing in an industrial environment. 

### Application Architecture Layers
- **1. Sensor Layer (Acquisition):** Captures industrial acoustic and vibration signals. (IoT Sensors, Microphones, Temperature & Pressure sensors)
- **2. Edge Layer (IoT Edge Processing):** Performs local, real-time noise filtering and feature extraction to reduce latency. (Raspberry Pi, Edge AI)
- **3. Data Layer (Storage):** Centralized storage for machine configuration, raw acoustic data, and processed ML features. (PostgreSQL, MongoDB)
- **4. Signal Processing Layer:** Converts raw signals into clean acoustic datasets via Digital Signal Processing (DSP), extracting features like MFCCs, FFTs, and Spectrograms.
- **5. AI Intelligence Layer (Processing):** Houses Acoustic ML models, Anomaly Detection models, and Failure Prediction (LSTM) models for defect detection and component life estimation. 
- **6. Analytics Layer (Presentation):** Provides actionable insights, dashboard visualizations, and industrial reports for Maintenance Engineers and Plant Managers. (Power BI, React)
- **7. Communication Layer:** Facilitates robust and lightweight device-to-cloud messaging. (MQTT Protocol)
- **8. Deployment Layer:** Manages containerized microservices for industrial scaling. (Kubernetes)

---

## 3. Core Features

The platform will deliver the following core features (F01–F07):

| Feature ID | Feature Name | Description |
| :--- | :--- | :--- |
| **F01** | **Real-time Machine Health Monitoring** | Continuous telemetry collection and display of machine status, maintaining a "Factory Reliability Index" and a real-time health score for individual components. |
| **F02** | **Acoustic Defect Detection** | Spectrogram classification and frequency pattern analysis to detect abnormal sound patterns indicating material defects (e.g., fatigue, corrosion). |
| **F03** | **Crack Prediction Engine** | Deep learning models specifically tuned to detect micro-crack signatures in metal impacts or bearing vibrations. |
| **F04** | **Predictive Maintenance Scheduler** | Analyzes component lifecycle and deterioration rates to suggest optimal maintenance windows, predicting remaining useful life to prevent unexpected downtime. |
| **F05** | **Digital Maintenance History** | A comprehensive digital ledger logging all past maintenance actions, component replacements, and technician notes. |
| **F06** | **Production Quality Analytics** | Aggregation of quality metrics, providing defect percentage reductions and total production counts mapped against acoustic data. |
| **F07** | **AI-based Failure Alerts** | Real-time threshold-based alerting mechanism to notify personnel of critical anomalies or high failure probability scenarios (e.g., immediate shutdown warnings). |

---

## 4. Tech Stack

The engineering execution will leverage the following primary technologies:

- **Signal Processing & Audio Analysis:** Python, Librosa, SciPy (for noise filtering, MFCC, Spectrogram generation)
- **AI / Deep Learning Models:** TensorFlow, PyTorch (for CNN acoustic classifiers, LSTM failure predictors)
- **IoT & Telemetry:** MQTT (lightweight device messaging)
- **Edge Hardware:** Raspberry Pi (Edge AI execution)
- **Databases:** PostgreSQL (Relational Master Data), MongoDB (Unstructured/Raw Signal JSON Data)
- **Frontend / Dashboard:** React
- **Business Intelligence:** Power BI
- **Infrastructure:** Kubernetes

---

## 5. Phase 1 Milestones (Sprint 1 Roadmap)

The immediate goal for Phase 1 is to establish the core data pipeline, encompassing sensor ingestion, acoustic signal processing, and foundational database schemas.

### Sprint 1 Goals & Deliverables

- **Step 1: Database Schema Initialization**
  - **Objective:** Establish the foundational RDBMS schemas based on the target architecture.
  - **Tasks:**
    - Create Industrial Asset Master Tables (`TBL_MACHINE_MASTER`, `TBL_COMPONENT_MASTER`, `TBL_PRODUCTION_LINE_MASTER`).
    - Create Sensor Data Tables (`TBL_SENSOR_MASTER`, `TBL_SENSOR_READING_LOG`).
    - Create Acoustic Signal Processing Tables (`TBL_AUDIO_SIGNAL_DATA`, `TBL_SIGNAL_FEATURE_STORE`).
  - **Deliverable:** Configured PostgreSQL instance with verified DDL scripts.

- **Step 2: MQTT Broker & IoT Ingestion Setup**
  - **Objective:** Enable a lightweight messaging channel for simulated or initial sensor data.
  - **Tasks:**
    - Deploy an MQTT broker (e.g., Mosquitto).
    - Write a Python script to simulate edge devices publishing raw acoustic/vibration telemetry to MQTT topics.
    - Create a subscriber service to ingest MQTT payloads and write to `TBL_SENSOR_READING_LOG`.
  - **Deliverable:** End-to-end data flow from simulated sensor to database.

- **Step 3: Basic Acoustic Signal Processing Pipeline**
  - **Objective:** Process raw audio files into machine learning-ready features.
  - **Tasks:**
    - Develop a Python module utilizing **Librosa** to ingest raw audio snippets (.wav/.raw).
    - Implement noise filtering routines (e.g., bandpass filters using SciPy).
    - Extract preliminary features: MFCCs (Mel-frequency cepstral coefficients) and Spectrograms.
    - Store the resulting feature vectors in `TBL_SIGNAL_FEATURE_STORE` as JSON.
  - **Deliverable:** A reproducible Python script/pipeline that converts an input audio file into structured DB records.

- **Step 4: Sprint Review & Documentation**
  - **Objective:** Validate Phase 1 deliverables against requirements.
  - **Tasks:**
    - Test end-to-end throughput: Simulated Sensor -> MQTT -> DB -> Python processing -> Feature DB.
    - Document API contracts and data models for the AI team to use in Sprint 2.
  - **Deliverable:** Sprint 1 sign-off and readiness for AI Model Development (Phase 2).
