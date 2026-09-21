-- PostgreSQL Database Schema for Industrial Asset and Sensor Data Management
-- Project: VIT-IND-03

-- Enable the uuid-ossp extension to generate UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 0. TBL_PRODUCTION_LINE_MASTER
CREATE TABLE TBL_PRODUCTION_LINE_MASTER (
    Production_Line_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Line_Name VARCHAR(255) NOT NULL,
    Factory_Name VARCHAR(255),
    Location VARCHAR(255),
    Supervisor_ID UUID,
    Status VARCHAR(50)
);

-- 1. TBL_MACHINE_MASTER
CREATE TABLE TBL_MACHINE_MASTER (
    Machine_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Machine_Name VARCHAR(255) NOT NULL,
    Machine_Type VARCHAR(100),
    Manufacturer VARCHAR(255),
    Model_Number VARCHAR(100),
    Installation_Date DATE,
    Production_Line_ID UUID,
    Criticality_Level VARCHAR(50), -- e.g., 'High', 'Medium', 'Low'
    Machine_Status VARCHAR(50), -- e.g., 'Active', 'Inactive', 'Maintenance'
    CONSTRAINT fk_production_line
        FOREIGN KEY(Production_Line_ID) 
        REFERENCES TBL_PRODUCTION_LINE_MASTER(Production_Line_ID)
        ON DELETE SET NULL
);

-- 2. TBL_COMPONENT_MASTER
CREATE TABLE TBL_COMPONENT_MASTER (
    Component_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Machine_ID UUID NOT NULL,
    Component_Name VARCHAR(255) NOT NULL,
    Material_Type VARCHAR(100),
    Batch_Number VARCHAR(100),
    Manufacturing_Date DATE,
    Expected_Life_Cycle INTEGER, -- Expected life cycle in hours or cycles
    CONSTRAINT fk_machine
        FOREIGN KEY(Machine_ID) 
        REFERENCES TBL_MACHINE_MASTER(Machine_ID)
        ON DELETE CASCADE
);

-- 3. TBL_SENSOR_MASTER
CREATE TABLE TBL_SENSOR_MASTER (
    Sensor_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Machine_ID UUID NOT NULL,
    Sensor_Type VARCHAR(100) NOT NULL, -- e.g., 'Acoustic Sensor', 'Vibration Sensor'
    Manufacturer VARCHAR(255),
    Installation_Date DATE,
    Calibration_Date DATE,
    Status VARCHAR(50), -- e.g., 'Active', 'Offline'
    CONSTRAINT fk_machine_sensor
        FOREIGN KEY(Machine_ID) 
        REFERENCES TBL_MACHINE_MASTER(Machine_ID)
        ON DELETE CASCADE
);

-- 4. TBL_AUDIO_SIGNAL_DATA
-- Optimized to store metadata about the raw acoustic recordings
CREATE TABLE TBL_AUDIO_SIGNAL_DATA (
    Audio_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Sensor_ID UUID NOT NULL,
    Recording_Time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    File_Path VARCHAR(500) NOT NULL, -- Path to the raw .wav or audio file in object storage
    Duration_Seconds REAL,
    Frequency_Range VARCHAR(100),
    Noise_Level REAL,
    CONSTRAINT fk_sensor
        FOREIGN KEY(Sensor_ID) 
        REFERENCES TBL_SENSOR_MASTER(Sensor_ID)
        ON DELETE CASCADE
);

-- 5. TBL_SIGNAL_FEATURE_STORE
-- Optimized using JSONB for storing extracted ML features like MFCCs, Spectrograms, etc.
CREATE TABLE TBL_SIGNAL_FEATURE_STORE (
    Feature_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Audio_ID UUID NOT NULL,
    Feature_Type VARCHAR(100) NOT NULL, -- e.g., 'MFCC', 'Spectral Energy'
    Feature_Vector JSONB NOT NULL, -- JSONB used for optimized querying and storage of ML feature arrays/objects
    Extraction_Method VARCHAR(255),
    Created_Date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_audio
        FOREIGN KEY(Audio_ID) 
        REFERENCES TBL_AUDIO_SIGNAL_DATA(Audio_ID)
        ON DELETE CASCADE
);

-- Indexes for performance optimization
CREATE INDEX idx_machine_id_comp ON TBL_COMPONENT_MASTER(Machine_ID);
CREATE INDEX idx_machine_id_sensor ON TBL_SENSOR_MASTER(Machine_ID);
CREATE INDEX idx_sensor_id_audio ON TBL_AUDIO_SIGNAL_DATA(Sensor_ID);
CREATE INDEX idx_audio_id_feature ON TBL_SIGNAL_FEATURE_STORE(Audio_ID);
CREATE INDEX idx_feature_type ON TBL_SIGNAL_FEATURE_STORE(Feature_Type);
-- GIN index for optimized querying within the JSONB payload
CREATE INDEX idx_feature_vector_gin ON TBL_SIGNAL_FEATURE_STORE USING GIN (Feature_Vector);

-- 6. TBL_SENSOR_READING_LOG
CREATE TABLE TBL_SENSOR_READING_LOG (
    Reading_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Sensor_ID UUID NOT NULL,
    Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    Signal_Value REAL,
    Unit VARCHAR(50),
    Sampling_Rate REAL,
    Data_Quality REAL,
    CONSTRAINT fk_sensor_log
        FOREIGN KEY(Sensor_ID) 
        REFERENCES TBL_SENSOR_MASTER(Sensor_ID)
        ON DELETE CASCADE
);

-- 7. TBL_DEFECT_CATEGORY_MASTER
CREATE TABLE TBL_DEFECT_CATEGORY_MASTER (
    Defect_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Defect_Name VARCHAR(100) NOT NULL,
    Description TEXT,
    Severity_Level VARCHAR(50) -- e.g., 'High', 'Medium', 'Low'
);

-- 8. TBL_AI_MODEL_MASTER
CREATE TABLE TBL_AI_MODEL_MASTER (
    Model_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Model_Name VARCHAR(255) NOT NULL,
    Algorithm_Type VARCHAR(100), -- e.g., 'CNN', 'LSTM'
    Framework VARCHAR(100), -- e.g., 'TensorFlow', 'PyTorch'
    Version VARCHAR(50),
    Deployment_Type VARCHAR(100)
);

-- 9. TBL_MODEL_TRAINING_HISTORY
CREATE TABLE TBL_MODEL_TRAINING_HISTORY (
    Training_ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    Model_ID UUID NOT NULL,
    Dataset_Size INTEGER,
    Training_Date DATE,
    Accuracy REAL,
    Precision REAL,
    Recall REAL,
    F1_Score REAL,
    CONSTRAINT fk_model_history
        FOREIGN KEY(Model_ID)
        REFERENCES TBL_AI_MODEL_MASTER(Model_ID)
        ON DELETE CASCADE
);

-- Additional Indexes
CREATE INDEX idx_sensor_id_log ON TBL_SENSOR_READING_LOG(Sensor_ID);
CREATE INDEX idx_model_id_history ON TBL_MODEL_TRAINING_HISTORY(Model_ID);
